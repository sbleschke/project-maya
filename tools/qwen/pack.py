"""tools/qwen/pack.py - a Maya pack for an EXL3 (exllamav3) Qwen3.8-Flash-Next model, beside the model's files.

    python tools/qwen/pack.py --model ~/models/Qwen3.8-Flash-Next-exl3-3.18bpw
    python tools/qwen/pack.py --model <dir> --check <a Qwen3.8-Flash-Next GGUF, or a fine-tune's> --check-only

As tools/exl3_pack.py does for GLM-5.3-Flash (its layout: <model>/maya-exl3.gguf, pack/index.txt, dense.bin, exl3.txt,
native_experts.txt, tokenizer/), with the names, layouts and conversions llama.cpp's converter gives this model
(third_party/llama.cpp/conversion/qwen.py, qwen4exp.py), checked against a real GGUF by --check:

  * the norms get Gemma's + 1, except the GDN's own (`linear_attn.norm`); A_log -> ssm_a = -exp(A_log);
  * the GDN's V heads go from grouped by K head to tiled, the order the engine's GDN kernels read
    ([G0_v0 .. G0_v2, G1_v0 ..] -> [G0_v0, G1_v0, .. G0_v1 ..]).  The float tensors are reordered as they are written.
    The EXL3 matrices (in_proj_qkv's V columns, in_proj_z's columns, out_proj's inputs) are reordered in
    <model>/maya-qwen-gdn.safetensors, about 1.8 GB and the only weights the pack copies: a V head is 128 wide, exactly
    one rotation block, so moving whole heads moves whole runs of 8 trellis tiles and of suh / svh, and the matrix is
    the same matrix, bit for bit, with no decode and no re-quantization;
  * the indexer's fused index_qk_proj (q | k) is decoded and rounded to BF16: the engine's indexer reads BF16 (K 7 here);
  * the small F16 projections the engine reads as BF16 (the hyper-connections' mixes, the router, the PLE's) are rounded
    to BF16 too, as iq_pack.py's --compat-bf16 rounds them for a GGUF.

pack/ngram.txt names the n-gram table for PleTable (--ple-gguf <pack>/ngram.txt): its file, K and offsets, the shards
checked to be one run of rows in row order.  The file may be swapped for another quant's (turboderp's card says so):
pack again afterwards.

The NextN (MTP) block stays out of the index, as iq_pack.py leaves a GGUF's out: block_count here is the trunk's 48.
pack/mtp/ is the drafter built from it (--mtp <pack>/mtp), dense.txt / dense.bin as tools/mtp_from_gguf.py writes them
(its matrices in Q8_0) but its experts as they are: experts_exl3.bin, [gate | up | down], a part [trellis | suh | svh],
with their bitrates in experts_exl3.txt, run by the trunk's EXL3 kernels: no decode and no second quantization.  All its
norms take the + 1, its two input norms too.  The vision tower is not packed (the Qwen path is text only).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import re
import struct
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import exl3_pack as X  # noqa: E402

ARCH = "Qwen4ExpForConditionalGeneration"
SIDECAR = "maya-qwen-gdn.safetensors"
TILED = "maya.tiled."                       # the sidecar's names: this + the model's name of the matrix
NGRAM_FILE = "ngram_embedding.safetensors"
L = "model.language_model."


def tiled(n_k: int, n_v: int) -> np.ndarray:
    """new head i = old head tiled(..)[i]: llama.cpp's _reorder_v_heads."""
    return np.arange(n_v).reshape(n_k, n_v // n_k).T.reshape(-1)


def gdn_geometry(t: dict) -> tuple:
    n_k, n_v = t["linear_num_key_heads"], t["linear_num_value_heads"]
    k_dim, v_dim = t["linear_key_head_dim"], t["linear_value_head_dim"]
    if n_v % n_k or v_dim % 128:
        raise SystemExit("the GDN's V heads (%d of %d values, %d K heads) do not move as whole rotation blocks"
                         % (n_v, v_dim, n_k))
    return n_k, n_v, k_dim, v_dim, tiled(n_k, n_v)


# ------------------------------------------------------------------ the sidecar
def sidecar_plan(st: X.SafeTensors, t: dict) -> list:
    """(name, dtype, shape, producer) of every reordered tensor."""
    n_k, n_v, k_dim, v_dim, heads = gdn_geometry(t)
    tiles = v_dim // 16
    v_tiles = (heads[:, None] * tiles + np.arange(tiles)[None, :]).reshape(-1)        # a matrix's V heads, in tiles
    v_vals = (heads[:, None] * v_dim + np.arange(v_dim)[None, :]).reshape(-1)         # ... in values
    qk = 2 * n_k * k_dim                                                              # q | k before them in qkv
    out = []

    def add(hf, axis, order_tiles, order_vals):
        for x in ("trellis", "suh", "svh", "mul1"):
            if not st.has(hf + "." + x):
                raise SystemExit("%s: no %s (the packer takes the mul1 codebook's matrices)" % (hf, x))
        tr = hf + ".trellis"
        shape = st.t[tr][2]
        if axis == 0:
            scales = (hf + ".suh", hf + ".svh")
            take = lambda tr=tr, s=shape: st.raw(tr).view("<i2").reshape(s)[order_tiles]  # noqa: E731
        else:
            scales = (hf + ".svh", hf + ".suh")
            take = lambda tr=tr, s=shape: st.raw(tr).view("<i2").reshape(s)[:, order_tiles]  # noqa: E731
        out.append((TILED + tr, "I16", shape, take))
        out.append((TILED + scales[0], "F16", st.t[scales[0]][2], lambda n=scales[0]: st.raw(n).view("<f2")[order_vals]))
        out.append((TILED + scales[1], "F16", st.t[scales[1]][2], lambda n=scales[1]: st.raw(n).view("<f2")))
        out.append((TILED + hf + ".mul1", "I32", [], lambda n=hf + ".mul1": st.raw(n).view("<i4")))

    for il, kind in enumerate(t["layer_types"][:t["num_hidden_layers"]]):
        if kind != "linear_attention":
            continue
        A = L + "layers.%d.linear_attn." % il
        add(A + "in_proj_qkv", 1, np.concatenate((np.arange(qk // 16), qk // 16 + v_tiles)),
            np.concatenate((np.arange(qk), qk + v_vals)))
        add(A + "in_proj_z", 1, v_tiles, v_vals)
        add(A + "out_proj", 0, v_tiles, v_vals)
    return out


def write_sidecar(st: X.SafeTensors, d: pathlib.Path, t: dict) -> None:
    plan = sidecar_plan(st, t)
    size = {"I16": 2, "F16": 2, "I32": 4}
    header = {"__metadata__": {"format": "maya_qwen_gdn_tiled", "source": "tools/qwen/pack.py"}}
    at = 0
    for name, ty, shape, _ in plan:
        nb = size[ty] * int(np.prod(shape, dtype=np.int64))
        header[name] = {"dtype": ty, "shape": list(shape), "data_offsets": [at, at + nb]}
        at += nb
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * ((-len(raw) - 8) % 8)
    with X.replace_atomic(d / SIDECAR) as f:
        f.write(struct.pack("<Q", len(raw)) + raw)
        for _, _, _, fn in plan:
            f.write(np.ascontiguousarray(fn()).tobytes())
    print("%s: the GDN's V heads in tiled order, %d tensors, %.2f GB" % (SIDECAR, len(plan), at / 1e9))


# ------------------------------------------------------------------ the plan
class QwenPlan(X.Plan):
    def _hc(self, gname: str, hf: str, with_inject: bool = True) -> None:
        st = self.st
        self.bf16(gname + "_down.weight", None, lambda: st.f32(hf + ".input_mix_weight_down.weight"))
        self.bf16(gname + "_up.weight", None, lambda: st.f32(hf + ".input_mix_weight_up.weight"))
        if with_inject:
            self.bf16(gname + "_inject.weight", None, lambda: st.f32(hf + ".block_inject_weight.weight"))
        self.f32(gname + "_norm.weight", None, lambda: st.f32(hf + ".hc_norm.weight") + 1)

    def _gamma(self, gname: str, hf: str) -> None:
        self.f32(gname, None, lambda: self.st.f32(hf) + 1)

    def _decoded(self, hf: str) -> np.ndarray:
        """An EXL3 matrix's weight as float32 rows (n x k, the GGUF layout)."""
        st = self.st
        shape = st.t[hf + ".trellis"][2]
        tr = st.raw(hf + ".trellis").view(np.int16).reshape(shape)
        return X.exl3_rows(tr, st.f32(hf + ".suh"), st.f32(hf + ".svh"), X.tile_k2(shape[2], 2, hf), 2)

    def build(self, mtp: bool = False) -> None:
        st, t = self.st, self.cfg
        n_layers = t["num_hidden_layers"]
        n_k, n_v, k_dim, v_dim, heads = gdn_geometry(t)
        channels = 2 * n_k * k_dim + n_v * v_dim
        qk = 2 * n_k * k_dim
        n_q = t["indexer_n_heads"] * t["indexer_head_dim"]
        ple = {i - 1 for i in t["ple_layer_ids"]}                  # (1-based in the config)
        d_conv = t["linear_conv_kernel_dim"]
        g = lambda n: (lambda: st.f32(n))  # noqa: E731

        self.bf16("token_embd.weight", None, None)                 # the BF16 embedding, copied raw (write_all)
        self.mat("output.weight", "lm_head")
        self._hc("output_hc", L + "hyper_connection_mixer", with_inject=False)
        for il in range(n_layers):
            P, H = "blk.%d." % il, L + "layers.%d." % il
            self._hc(P + "hc_attn", H + "attn_hyper_connection")
            self._hc(P + "hc_ffn", H + "mlp_hyper_connection")
            if t["layer_types"][il] == "linear_attention":
                A = H + "linear_attn."
                self.mat(P + "attn_qkv.weight", TILED + A + "in_proj_qkv")
                self.mat(P + "attn_gate.weight", TILED + A + "in_proj_z")
                self.mat(P + "ssm_out.weight", TILED + A + "out_proj")
                self.bf16(P + "ssm_alpha.weight", None, lambda A=A: st.f32(A + "in_proj_a.weight")[heads])
                self.bf16(P + "ssm_beta.weight", None, lambda A=A: st.f32(A + "in_proj_b.weight")[heads])
                self.f32(P + "ssm_a", None, lambda A=A: -np.exp(st.f32(A + "A_log"))[heads])
                self.f32(P + "ssm_dt.bias", None, lambda A=A: st.f32(A + "dt_bias")[heads])
                self.f32(P + "ssm_norm.weight", None, g(A + "norm.weight"))
                self.f32(P + "ssm_conv1d.weight", [d_conv, channels], lambda A=A: self._conv(A, qk, n_v, v_dim, heads, d_conv))
            else:
                A = H + "self_attn."
                self.mat(P + "attn_q.weight", A + "q_proj")
                self.mat(P + "attn_k.weight", A + "k_proj")
                self.mat(P + "attn_v.weight", A + "v_proj")
                self.mat(P + "attn_output.weight", A + "o_proj")
                self._gamma(P + "attn_q_norm.weight", A + "q_norm.weight")
                self._gamma(P + "attn_k_norm.weight", A + "k_norm.weight")
                self._gamma(P + "indexer.q_norm.weight", A + "indexer.q_layernorm.weight")
                self._gamma(P + "indexer.k_norm.weight", A + "indexer.k_layernorm.weight")
                I = A + "indexer.index_qk_proj"
                self.bf16(P + "indexer.q_proj.weight", None, lambda I=I: self._decoded(I)[:n_q])
                self.bf16(P + "indexer.k_proj.weight", None, lambda I=I: self._decoded(I)[n_q:])
            if il in ple:
                E = H + "ple."
                self.bf16(P + "ple_key.weight", None, g(E + "key_proj.weight"))
                self.bf16(P + "ple_value.weight", None, g(E + "value_proj.weight"))
                self.f16(P + "ple_conv1d.weight", [d_conv, channels],
                         lambda E=E: st.f32(E + "conv1d.weight").reshape(channels, d_conv))
                for nm in ("key", "query", "conv"):
                    self._gamma(P + "ple_norm_%s.weight" % nm, E + "norm_%s.weight" % nm)
            M = H + "mlp."
            self.bf16(P + "ffn_gate_inp.weight", None, g(M + "gate.weight"))
            self.bf16(P + "ffn_gate_inp_shexp.weight", None, lambda M=M: st.f32(M + "shared_expert_gate.weight").reshape(-1))
            for r in X.ROLES:
                self.mat(P + "ffn_%s_shexp.weight" % r, M + "shared_expert.%s_proj" % r)
            self.layers.append(self._experts(il, M, t["num_experts"]))

    def _conv(self, A: str, qk: int, n_v: int, v_dim: int, heads: np.ndarray, d_conv: int) -> np.ndarray:
        w = self.st.f32(A + "conv1d.weight").reshape(-1, d_conv)   # torch [channels, 1, d_conv] = GGUF [d_conv, channels]
        v = w[qk:].reshape(n_v, v_dim, d_conv)[heads].reshape(-1, d_conv)
        return np.concatenate((w[:qk], v))


# ------------------------------------------------------------------ the GGUF's keys
def tokenizer_meta(d: pathlib.Path, t: dict) -> dict:
    """The tokenizer as llama.cpp's converter types it (--check compares it with a real GGUF's)."""
    tj = json.loads((d / "tokenizer.json").read_text(encoding="utf-8"))
    tc = json.loads((d / "tokenizer_config.json").read_text(encoding="utf-8"))
    if tj["model"]["type"] != "BPE":
        raise SystemExit("tokenizer.json is %s, not BPE" % tj["model"]["type"])
    vocab = dict(tj["model"]["vocab"])
    added = {a["content"]: a for a in tj.get("added_tokens", [])}
    for a in added.values():
        vocab[a["content"]] = a["id"]
    rev = {i: s for s, i in vocab.items()}
    tokens, types = [], []
    for i in range(t["vocab_size"]):
        s = rev.get(i)
        if s is None:
            tokens.append("[PAD%d]" % i)
            types.append(5)                       # UNUSED
        else:
            tokens.append(s)
            # NORMAL; CONTROL (the special ones, and <|fim_prefix|> .. <|file_sep|>, not special in tokenizer.json:
            # they would be matched in plain text otherwise); USER_DEFINED (<think>, <tool_call> ...)
            control = s in added and (added[s].get("special") or (s.startswith("<|") and s.endswith("|>")))
            types.append(1 if s not in added else 3 if control else 4)
    meta = {
        "tokenizer.ggml.model": "gpt2",
        "tokenizer.ggml.pre": "qwen35",
        "tokenizer.ggml.tokens": tokens,
        "tokenizer.ggml.token_type": types,
        "tokenizer.ggml.merges": [m if isinstance(m, str) else " ".join(m) for m in tj["model"]["merges"]],
        "tokenizer.ggml.eos_token_id": added[tc["eos_token"]]["id"],
        "tokenizer.ggml.padding_token_id": added[tc["pad_token"]]["id"],
        "tokenizer.ggml.bos_token_id": t["bos_token_id"],
        "tokenizer.ggml.add_bos_token": bool(tc.get("add_bos_token", False)),
    }
    if (d / "chat_template.jinja").exists():
        meta["tokenizer.chat_template"] = (d / "chat_template.jinja").read_text(encoding="utf-8")
    return meta


def model_meta(cfg: dict, table: dict) -> list:
    """llama.cpp's qwen4exp keys.  block_count is the trunk's: the NextN block is not in this pack."""
    t = cfg["text_config"]
    P = "qwen4exp."
    n = t["num_hidden_layers"]
    kinds = t["layer_types"][:n]
    rope = t["rope_parameters"]
    return [
        ("general.architecture", "qwen4exp", "string"),
        ("general.name", "Qwen3.8-Flash-Next EXL3", "string"),
        (P + "block_count", n, "u32"),
        (P + "context_length", t["max_position_embeddings"], "u32"),
        (P + "embedding_length", t["hidden_size"], "u32"),
        (P + "attention.head_count", t["num_attention_heads"], "u32"),
        (P + "attention.head_count_kv", t["num_key_value_heads"], "u32"),
        (P + "rope.dimension_sections", list(rope["mrope_section"]) + [0], "array:i32"),
        (P + "rope.freq_base", float(rope["rope_theta"]), "f32"),
        (P + "attention.layer_norm_rms_epsilon", float(t["rms_norm_eps"]), "f32"),
        (P + "expert_count", t["num_experts"], "u32"),
        (P + "expert_used_count", t["num_experts_per_tok"], "u32"),
        (P + "attention.key_length", t["head_dim"], "u32"),
        (P + "attention.value_length", t["head_dim"], "u32"),
        (P + "expert_feed_forward_length", t["moe_intermediate_size"], "u32"),
        (P + "expert_shared_feed_forward_length", t["shared_expert_intermediate_size"], "u32"),
        (P + "nextn_predict_layers", 0, "u32"),
        (P + "ssm.conv_kernel", t["linear_conv_kernel_dim"], "u32"),
        (P + "ssm.state_size", t["linear_key_head_dim"], "u32"),
        (P + "ssm.group_count", t["linear_num_key_heads"], "u32"),
        (P + "ssm.time_step_rank", t["linear_num_value_heads"], "u32"),
        (P + "ssm.inner_size", t["linear_value_head_dim"] * t["linear_num_value_heads"], "u32"),
        (P + "attention.recurrent_layers", [x == "linear_attention" for x in kinds], "array:bool"),
        (P + "full_attention_interval", t["full_attention_interval"], "u32"),
        (P + "rope.dimension_count", int(t["head_dim"] * rope["partial_rotary_factor"]), "u32"),
        (P + "hyper_connection.count", t["hc_count"], "u32"),
        (P + "hyper_connection.low_rank", t["hc_lowrank"], "u32"),
        (P + "attention.indexer.head_count", t["indexer_n_heads"], "u32"),
        (P + "attention.indexer.key_length", t["indexer_head_dim"], "u32"),
        (P + "attention.indexer.top_k", t["indexer_budget"], "u32"),
        (P + "attention.compress_ratios", [t["indexer_compress_ratio"] if x == "full_attention" else 0 for x in kinds],
         "array:i32"),
        (P + "ple.layers", [i - 1 for i in t["ple_layer_ids"]], "array:i32"),
        (P + "ple.ngram_size", t["ngram_size"], "u32"),
        (P + "ple.heads_per_ngram", t["heads_per_ngram"], "u32"),
        (P + "ple.conv_kernel", t["ple_conv_kernel_size"], "u32"),
        (P + "ple.eos_token_id", t["eos_token_id"], "u32"),
        (P + "ple.image_token_id", cfg["image_token_id"], "u32"),
        (P + "embedding_length_per_layer_input", table["row_dim"], "u32"),
        (P + "ple.layer_multipliers", table["layer_multipliers"], "array:u64"),
        (P + "ple.head_offsets", table["head_offsets"], "array:u64"),
        (P + "ple.head_vocab_sizes", table["head_vocab_sizes"], "array:u64"),
    ]


# ------------------------------------------------------------------ the n-gram table
def ngram_table(st: X.SafeTensors, d: pathlib.Path, t: dict) -> dict:
    """The table's geometry, checked: one run of rows in row order, heads covering them."""
    path = d / NGRAM_FILE
    if not path.is_file():
        raise SystemExit("%s is missing: the model's n-gram table" % path)
    with path.open("rb") as f:
        header = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
    md = header.get("__metadata__", {})
    if md.get("format") != "exl3_ngram_trellis" or md.get("codebook") != "mul1":
        raise SystemExit("%s: format %r codebook %r, not exl3_ngram_trellis with mul1"
                         % (NGRAM_FILE, md.get("format"), md.get("codebook")))
    il = t["ple_layer_ids"][0] - 1
    base = L + "layers.%d.ple.ple_embedding.ngram_embedding." % il
    shards = []
    for name in header:
        m = re.fullmatch(re.escape(base) + r"shard_(\d+)\.trellis", name)
        if m:
            shards.append((int(m.group(1)), name))
    shards.sort()
    if not shards and base + "trellis" in header:              # (exllamav3 1.4's quants: one tensor, no shards)
        shards = [(0, base + "trellis")]
    elif [s[0] for s in shards] != list(range(t["split_ngram_parts"])):
        raise SystemExit("%s: %d shards, expected %d in a row" % (NGRAM_FILE, len(shards), t["split_ngram_parts"]))
    at, rows, words = None, 0, None
    for _, name in shards:
        fn, dt, shape, off, nb = st.t[name]
        if fn != NGRAM_FILE or dt != "I16" or len(shape) != 2:
            raise SystemExit("%s: not an I16 [rows, words] tensor of %s" % (name, NGRAM_FILE))
        if at is not None and off != at:
            raise SystemExit("%s: the shards are not one run of rows in row order (at %d, expected %d)" % (name, off, at))
        if words not in (None, shape[1]):
            raise SystemExit("%s: its rows are %d words, the others %d" % (name, shape[1], words))
        at, words, rows = off + nb, shape[1], rows + shape[0]
    row_dim = int(md["row_dim"])
    K = (words - 1) * 16 // row_dim
    if row_dim != 160 or not 1 <= K <= 8 or words != 1 + row_dim * K // 16 or K != int(md["K"]) or rows != int(md["rows"]):
        raise SystemExit("%s: rows of %d words are not a %d-wide table at K %s" % (NGRAM_FILE, words, row_dim, md["K"]))
    ints = lambda n: st.raw(base + n).view("<i8").tolist()  # noqa: E731
    offsets, sizes, mult = ints("head_offsets"), ints("head_vocab_sizes"), ints("layer_multipliers")
    # (the rows are rounded up to a multiple of make_ngram_vocab_size_divisible_by: the last few belong to no hash)
    div = t["make_ngram_vocab_size_divisible_by"]
    if offsets != [sum(sizes[:i]) for i in range(len(sizes))] or rows != (sum(sizes) + div - 1) // div * div:
        raise SystemExit("%s: the heads (%d rows) do not cover its %d rows" % (NGRAM_FILE, sum(sizes), rows))
    return {"K": K, "row_dim": row_dim, "row_bytes": words * 2, "rows": rows, "heads": len(sizes),
            "table_off": st.t[shards[0][1]][3], "head_bias_off": st.t[base + "head_bias"][3],
            "head_offsets_off": st.t[base + "head_offsets"][3], "head_vocab_sizes_off": st.t[base + "head_vocab_sizes"][3],
            "layer_multipliers_off": st.t[base + "layer_multipliers"][3], "head_offsets": offsets,
            "head_vocab_sizes": sizes, "layer_multipliers": mult}


def write_ngram_txt(table: dict, d: pathlib.Path, out: pathlib.Path) -> None:
    try:
        file = os.path.relpath(d / NGRAM_FILE, out)
    except ValueError:                                         # (another drive)
        file = str(d / NGRAM_FILE)
    with X.replace_atomic(out / "ngram.txt") as fb:
        fb.write(("# maya n-gram table v1 -- generated by tools/qwen/pack.py from %s (--ple-gguf this file)\n" % d.name
                  + "format exl3_ngram_trellis\nfile %s\n" % file.replace("\\", "/")
                  + "".join("%s %d\n" % (k, table[k]) for k in (
                      "K", "row_dim", "row_bytes", "rows", "heads", "table_off", "head_bias_off", "head_offsets_off",
                      "head_vocab_sizes_off", "layer_multipliers_off"))).encode())
    print("ngram.txt: %d rows at K %d, %d bytes a row, %.1f GB" % (table["rows"], table["K"], table["row_bytes"],
                                                                 table["rows"] * table["row_bytes"] / 1e9))


# ------------------------------------------------------------------ the MTP drafter
def write_mtp(plan: QwenPlan, out: pathlib.Path) -> None:
    from mtp_from_gguf import bf16
    from mtp_rt import q8_0
    st, t = plan.st, plan.cfg
    H, FF, NE = t["hidden_size"], t["moe_intermediate_size"], t["num_experts"]
    M, A, S = "mtp.", "mtp.layers.0.self_attn.", "mtp.layers.0.mlp."
    if not st.has(M + "fc_embedding.trellis"):
        print("note: the files carry no MTP block - no drafter")
        return
    out.mkdir(parents=True, exist_ok=True)
    lay = plan._experts(-1, S, NE)
    if lay["cb"] != 2 or lay["k2"][0] != lay["k2"][1]:
        raise SystemExit("the MTP experts are not of the mul1 codebook, or their gate and up differ")
    with X.replace_atomic(out / "experts_exl3.bin") as f:
        for e in range(NE):
            for r in X.ROLES:
                E = S + "experts.%d.%s_proj." % (e, r)
                for x in ("trellis", "suh", "svh"):
                    f.write(st.raw(E + x).tobytes())
    with X.replace_atomic(out / "experts_exl3.txt") as f:
        f.write(("%d %d %d %d %d\n" % (*lay["k2"], lay["gu_bytes"], lay["dn_bytes"])).encode())
    g, one = st.f32, lambda n: st.f32(n) + 1  # noqa: E731
    dense = [
        ("fc_embedding.weight", "q8_0", plan._decoded(M + "fc_embedding")),
        ("fc_hidden.weight", "q8_0", plan._decoded(M + "fc_hidden")),
        ("pre_fc_norm_embedding.weight", "f32", one(M + "pre_fc_norm_embedding.weight")),
        ("pre_fc_norm_hidden.weight", "f32", one(M + "pre_fc_norm_hidden.weight")),
        ("self_attn.q_proj.weight", "q8_0", plan._decoded(A + "q_proj")),
        ("self_attn.k_proj.weight", "q8_0", plan._decoded(A + "k_proj")),
        ("self_attn.v_proj.weight", "q8_0", plan._decoded(A + "v_proj")),
        ("self_attn.o_proj.weight", "q8_0", plan._decoded(A + "o_proj")),
        ("self_attn.q_norm.weight", "f32", one(A + "q_norm.weight")),
        ("self_attn.k_norm.weight", "f32", one(A + "k_norm.weight")),
    ]
    for short, hf in (("attn_hyper_connection", "mtp.layers.0.attn_hyper_connection."),
                      ("mlp_hyper_connection", "mtp.layers.0.mlp_hyper_connection.")):
        dense += [(short + ".hc_norm.weight", "f32", one(hf + "hc_norm.weight")),
                  (short + ".input_mix_weight_down.weight", "bf16", g(hf + "input_mix_weight_down.weight")),
                  (short + ".input_mix_weight_up.weight", "bf16", g(hf + "input_mix_weight_up.weight")),
                  (short + ".block_inject_weight.weight", "bf16", g(hf + "block_inject_weight.weight"))]
    X_ = M + "hyper_connection_mixer."
    dense += [("hyper_connection_mixer.hc_norm.weight", "f32", one(X_ + "hc_norm.weight")),
              ("hyper_connection_mixer.input_mix_weight_down.weight", "bf16", g(X_ + "input_mix_weight_down.weight")),
              ("hyper_connection_mixer.input_mix_weight_up.weight", "bf16", g(X_ + "input_mix_weight_up.weight")),
              ("mlp.gate.weight", "bf16", g(S + "gate.weight")),
              ("mlp.shared_expert.gate_proj.weight", "q8_0", plan._decoded(S + "shared_expert.gate_proj")),
              ("mlp.shared_expert.up_proj.weight", "q8_0", plan._decoded(S + "shared_expert.up_proj")),
              ("mlp.shared_expert.down_proj.weight", "q8_0", plan._decoded(S + "shared_expert.down_proj")),
              ("mlp.shared_expert_gate.weight", "bf16", g(S + "shared_expert_gate.weight").reshape(1, -1))]
    lines, off = [], 0
    with X.replace_atomic(out / "dense.bin") as f:
        for name, kind, a in dense:
            a = np.asarray(a, dtype=np.float32)
            rows, cols = (1, a.size) if a.ndim == 1 else (a.shape[0], int(np.prod(a.shape[1:])))
            raw = q8_0(a.reshape(rows, cols)) if kind == "q8_0" else bf16(a) if kind == "bf16" else a.tobytes()
            pad = (-off) % 256
            f.write(b"\0" * pad)
            off += pad
            f.write(raw)
            lines.append("%s %s %d %d %d %d" % (name, kind, rows, cols, off, len(raw)))
            off += len(raw)
    with X.replace_atomic(out / "dense.txt") as f:
        f.write(("\n".join(lines) + "\n").encode())
    print("mtp/: %d experts at K %s, %d tensors, dense.bin %.2f GB" % (NE, X.k_text(lay["k2"][0]), len(lines), off / 1e9))


# ------------------------------------------------------------------ --check
def check(plan: QwenPlan, st: X.SafeTensors, ref: pathlib.Path, layers: set, meta: list, tok: dict) -> int:
    """This pack against a GGUF of the model (a fine-tune's serves: its weights are close, not the same).  Reordered
    heads, a missed + 1 or a wrong layout show as a cosine far from 1, or as a difference of 1."""
    from _paths import add_gguf_py
    add_gguf_py()
    from gguf import GGUFReader
    from gguf.quants import dequantize
    import gguf_reader as G
    R = {t.name: t for t in GGUFReader(ref).tensors}
    bad = 0

    def cosine(a, b):
        a, b = a.reshape(-1).astype(np.float64), b.reshape(-1).astype(np.float64)
        return float((a * b).sum() / math.sqrt((a * a).sum() * (b * b).sum() + 1e-30))

    def wanted(name):
        m = re.match(r"blk\.(\d+)\.", name)
        return m is None or int(m.group(1)) in layers

    print("keys against the GGUF's:")
    md = G.GGUFFile(ref).metadata
    own = {k: v for k, v, _ in meta}
    own.update(tok)
    for k in sorted(own):
        # (compress_ratios: GGUFs differ, today's converter writes the ratio where this does and older ones zeros; the
        #  engine reads neither, its QSA block is fixed)
        if k in ("general.name", "tokenizer.chat_template", "qwen4exp.block_count", "qwen4exp.nextn_predict_layers",
                 "qwen4exp.attention.compress_ratios"):
            continue
        if k not in md:
            print("  %-46s not in the GGUF" % k)
            continue
        a, b = own[k], md[k]
        if isinstance(a, float):
            same = math.isclose(a, float(b), rel_tol=1e-6)
        else:
            # (a list per layer: the GGUF's may go on for its NextN block)
            same = list(a) == list(b)[:len(a)] if isinstance(a, list) and k.endswith("_layers") else \
                list(a) == list(b) if isinstance(a, list) else a == b
        if not same:
            print("  %-46s differs: %s vs %s" % (k, str(a)[:50], str(b)[:50]))
            bad += 1
    for k in md:
        if k.startswith(("qwen4exp.", "tokenizer.ggml.")) and k not in own:
            print("  %-46s only in the GGUF" % k)
            bad += 1
    print("float tensors:")
    for name, shape, ty, fn in plan.floats:
        if fn is None or not wanted(name):
            continue
        if name not in R:
            print("  %-34s not in the GGUF  FAIL" % name)
            bad += 1
            continue
        a = np.ascontiguousarray(fn(), dtype=np.float32)
        tt = R[name]
        b = dequantize(tt.data, tt.tensor_type).astype(np.float32)
        gs = X.gguf_shape(a.shape, shape)
        if list(gs) != [int(x) for x in tt.shape]:
            print("  %-34s shape %s vs %s  FAIL" % (name, gs, [int(x) for x in tt.shape]))
            bad += 1
            continue
        c = cosine(a, b)
        ok = c > 0.98
        print("  %-34s cos %.5f %s (GGUF %s)" % (name, c, "ok" if ok else "FAIL", tt.tensor_type.name))
        bad += not ok
    print("EXL3 matrices:")
    for name, hf, k, n, k2, cb in plan.mats:
        if not wanted(name):
            continue
        tt = R[name]
        if [k, n] != [int(x) for x in tt.shape]:
            print("  %-34s %d x %d vs %s  FAIL" % (name, k, n, [int(x) for x in tt.shape]))
            bad += 1
            continue
        if name == "output.weight":
            n_c = 512                                          # (248320 outputs: the first rows only)
            tr = st.raw(hf + ".trellis").view(np.int16).reshape(st.t[hf + ".trellis"][2])
            got = X.exl3_rows(tr, st.f32(hf + ".suh"), st.f32(hf + ".svh"), k2, cb, 0, n_c)
            want = dequantize(tt.data.reshape(-1, tt.data.shape[-1])[:n_c], tt.tensor_type).astype(np.float32)
        else:
            got = plan._decoded(hf)
            want = dequantize(tt.data, tt.tensor_type).astype(np.float32)
        c = cosine(got, want)
        ok = c > 0.9
        print("  %-34s K%s cos %.5f %s (GGUF %s)" % (name, X.k_text(k2), c, "ok" if ok else "FAIL", tt.tensor_type.name))
        bad += not ok
    for lay in plan.layers:
        il = lay["layer"]
        if il not in layers:
            continue
        for ri, r in enumerate(X.ROLES):
            hf = L + "layers.%d.mlp.experts.7.%s_proj" % (il, r)
            tt = R["blk.%d.ffn_%s_exps.weight" % (il, r)]
            want = dequantize(tt.data[7], tt.tensor_type).astype(np.float32)
            c = cosine(plan._decoded(hf), want)
            ok = c > 0.85
            print("  blk.%d expert 7 %-5s K%s cos %.5f %s (GGUF %s)" % (il, r, X.k_text(lay["k2"][ri]), c,
                                                                     "ok" if ok else "FAIL", tt.tensor_type.name))
            bad += not ok
    print("check: %s" % ("PASS" if bad == 0 else "%d FAILED" % bad))
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True, help="the EXL3 model directory (config.json, *.safetensors)")
    ap.add_argument("--out", help="the pack directory (default: <model>/pack)")
    ap.add_argument("--check", help="a GGUF of the same model or a fine-tune of it: compare before packing")
    ap.add_argument("--check-layers", default="0,1,3,47", help="the layers --check decodes (default 0,1,3,47)")
    ap.add_argument("--check-only", action="store_true", help="--check without writing the pack")
    a = ap.parse_args()
    d = pathlib.Path(a.model).expanduser().absolute()
    out = pathlib.Path(a.out).expanduser().absolute() if a.out else d / "pack"
    cfg = json.loads((d / "config.json").read_text())
    if cfg.get("architectures", [None])[0] != ARCH:
        print("%s is %s; this packer knows Qwen3.8-Flash-Next (%s)" % (d.name, cfg.get("architectures"), ARCH))
        return 1
    q = cfg.get("quantization_config", {})
    if q.get("quant_method") != "exl3":
        print("%s is not an EXL3 model (quantization_config.quant_method %r)" % (d.name, q.get("quant_method")))
        return 1
    t = cfg["text_config"]
    print("%s: EXL3 %s, %s bpw (head %s), codebook %s" % (d.name, q.get("version"), q.get("bits"), q.get("head_bits"),
                                                          q.get("codebook", "3inst")))
    write_sidecar(X.SafeTensors(d), d, t)
    st = X.SafeTensors(d)                                      # (now with the sidecar)
    table = ngram_table(st, d, t)
    plan = QwenPlan(st, t)
    plan.build()
    meta, tok = model_meta(cfg, table), tokenizer_meta(d, t)
    if a.check:
        rc = check(plan, st, pathlib.Path(a.check).expanduser().absolute(),
                   {int(x) for x in a.check_layers.split(",")}, meta, tok)
        if rc or a.check_only:
            return rc
    X.write_all(plan, d, out, t, 0, None, meta=meta, tok=tok)
    write_ngram_txt(table, d, out)
    write_mtp(plan, out / "mtp")
    for n in plan.notes:
        print("note: " + n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
