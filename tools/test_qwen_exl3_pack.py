"""tools/qwen/pack.py on a small synthetic Qwen3.8-Flash-Next EXL3 model, with no GPU, network or model download: the
V heads' move into tiled order changes no weight, the conversions llama.cpp's converter makes (the norms' + 1 but not
the GDN's own, -exp(A_log), the fused indexer's q | k), the index the engine reads, the tokenizer's types and the n-gram
table's checks.

    python -m unittest tools.test_qwen_exl3_pack -v
"""
import json
import pathlib
import struct
import subprocess
import sys
import tempfile
import unittest

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "qwen"))
import exl3_pack as X  # noqa: E402
import pack as Q  # noqa: E402

L = "model.language_model."
HIDDEN, N_EXPERT, CONV = 256, 2, 4
N_K, N_V, HEAD = 2, 4, 128
QK, V = 2 * N_K * HEAD, N_V * HEAD
CHANNELS = QK + V
HEADS = np.array([0, 2, 1, 3])                 # [G0_v0, G1_v0, G0_v1, G1_v1] of [G0_v0, G0_v1, G1_v0, G1_v1]
TEXT = {
    "num_hidden_layers": 2, "layer_types": ["linear_attention", "full_attention"], "hidden_size": HIDDEN,
    "max_position_embeddings": 1024, "num_attention_heads": 2, "num_key_value_heads": 2, "head_dim": HEAD,
    "rms_norm_eps": 1e-6, "num_experts": N_EXPERT, "num_experts_per_tok": 1, "moe_intermediate_size": 128,
    "shared_expert_intermediate_size": 128, "linear_conv_kernel_dim": CONV, "linear_key_head_dim": HEAD,
    "linear_num_key_heads": N_K, "linear_num_value_heads": N_V, "linear_value_head_dim": HEAD,
    "full_attention_interval": 2, "hc_count": 4, "hc_lowrank": 128, "indexer_n_heads": 1, "indexer_head_dim": 128,
    "indexer_budget": 16, "indexer_compress_ratio": 4, "ple_layer_ids": [2], "ngram_size": 3, "heads_per_ngram": 8,
    "ple_conv_kernel_size": CONV, "eos_token_id": 3, "bos_token_id": 3, "vocab_size": 8, "split_ngram_parts": 2,
    "make_ngram_vocab_size_divisible_by": 128,
    "rope_parameters": {"mrope_section": [1, 1, 0], "rope_theta": 1e7, "partial_rotary_factor": 0.25},
}


def write_st(path: pathlib.Path, tensors: dict, metadata: dict | None = None) -> None:
    """name -> numpy array, written as safetensors."""
    names = {"f2": "F16", "i2": "I16", "i4": "I32", "i8": "I64", "u2": "BF16"}
    header = {"__metadata__": metadata or {}}
    at = 0
    for k, a in tensors.items():
        n = a.nbytes
        header[k] = {"dtype": names[a.dtype.str[1:]], "shape": list(a.shape), "data_offsets": [at, at + n]}
        at += n
    raw = json.dumps(header).encode()
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(raw)) + raw)
        for a in tensors.values():
            f.write(a.tobytes())


class Model:
    def __init__(self, d: pathlib.Path, ngram_gap: bool = False):
        self.r = np.random.default_rng(11)
        self.t = {}
        r = self.r
        fl = lambda *s: (r.standard_normal(s) * 0.1).astype("<f2")  # noqa: E731
        bf = lambda *s: ((r.standard_normal(s) * 0.1).astype("<f4").view("<u4") >> 16).astype("<u2")  # noqa: E731
        self.t[L + "embed_tokens.weight"] = bf(8, HIDDEN)
        self.mat("lm_head", HIDDEN, 128)
        self.hc(L + "hyper_connection_mixer", False)
        H = L + "layers.0."
        A = H + "linear_attn."
        self.mat(A + "in_proj_qkv", HIDDEN, CHANNELS)
        self.mat(A + "in_proj_z", HIDDEN, V)
        self.mat(A + "out_proj", V, HIDDEN)
        self.t[A + "in_proj_a.weight"] = fl(N_V, HIDDEN)
        self.t[A + "in_proj_b.weight"] = fl(N_V, HIDDEN)
        self.t[A + "A_log"] = bf(N_V)
        self.t[A + "dt_bias"] = bf(N_V)
        self.t[A + "norm.weight"] = bf(HEAD)
        self.t[A + "conv1d.weight"] = bf(CHANNELS, 1, CONV)
        H = L + "layers.1."
        A = H + "self_attn."
        for x, n in (("q_proj", 256), ("k_proj", 256), ("v_proj", 256)):
            self.mat(A + x, HIDDEN, n)
        self.mat(A + "o_proj", 256, HIDDEN)
        self.mat(A + "indexer.index_qk_proj", HIDDEN, 256)
        for x in ("q_norm", "k_norm", "indexer.q_layernorm", "indexer.k_layernorm"):
            self.t[A + x + ".weight"] = bf(HEAD)
        E = H + "ple."
        self.t[E + "key_proj.weight"] = fl(CHANNELS, HIDDEN)
        self.t[E + "value_proj.weight"] = fl(HIDDEN, HIDDEN)
        self.t[E + "conv1d.weight"] = fl(CHANNELS, 1, CONV)
        for x in ("key", "query", "conv"):
            self.t[E + "norm_%s.weight" % x] = bf(CHANNELS)
        for il in (0, 1):
            H = L + "layers.%d." % il
            self.hc(H + "attn_hyper_connection")
            self.hc(H + "mlp_hyper_connection")
            self.t[H + "mlp.gate.weight"] = fl(N_EXPERT, HIDDEN)
            self.t[H + "mlp.shared_expert_gate.weight"] = fl(1, HIDDEN)
            for x, k, n in (("gate", HIDDEN, 128), ("up", HIDDEN, 128), ("down", 128, HIDDEN)):
                self.mat(H + "mlp.shared_expert.%s_proj" % x, k, n)
                for e in range(N_EXPERT):
                    self.mat(H + "mlp.experts.%d.%s_proj" % (e, x), k, n)
        write_st(d / "model-00001-of-00001.safetensors", self.t)
        (d / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {k: "model-00001-of-00001.safetensors" for k in self.t}}))
        cfg = {"architectures": [Q.ARCH], "image_token_id": 7, "text_config": TEXT,
               "quantization_config": {"quant_method": "exl3", "version": "1.6.3", "bits": 3.0, "head_bits": 6}}
        (d / "config.json").write_text(json.dumps(cfg))
        (d / "tokenizer.json").write_text(json.dumps({
            "model": {"type": "BPE", "vocab": {"a": 0, "b": 1, "ab": 2}, "merges": ["a b"]},
            "added_tokens": [{"id": 3, "content": "<|endoftext|>", "special": True},
                             {"id": 4, "content": "<|im_end|>", "special": True},
                             {"id": 5, "content": "<|fim_prefix|>", "special": False},
                             {"id": 6, "content": "<think>", "special": False}]}))
        (d / "tokenizer_config.json").write_text(json.dumps({"eos_token": "<|im_end|>", "pad_token": "<|endoftext|>",
                                                             "add_bos_token": False}))
        (d / "chat_template.jinja").write_text("{{ messages }}")
        self.ngram(d, ngram_gap)

    def hc(self, p: str, inject: bool = True) -> None:
        r = self.r
        fl = lambda *s: (r.standard_normal(s) * 0.1).astype("<f2")  # noqa: E731
        self.t[p + ".input_mix_weight_down.weight"] = fl(128, 4 * HIDDEN)
        self.t[p + ".input_mix_weight_up.weight"] = fl(4 * HIDDEN, 128)
        if inject:
            self.t[p + ".block_inject_weight.weight"] = fl(4, 4 * HIDDEN)
        self.t[p + ".hc_norm.weight"] = fl(4 * HIDDEN)

    def mat(self, hf: str, k: int, n: int, K: int = 3) -> None:
        r = self.r
        self.t[hf + ".trellis"] = r.integers(-32768, 32767, (k // 16, n // 16, 16 * K), dtype="<i2", endpoint=True)
        self.t[hf + ".suh"] = (r.standard_normal(k) * 0.1 + 1).astype("<f2")
        self.t[hf + ".svh"] = (r.standard_normal(n) * 0.1 + 1).astype("<f2")
        self.t[hf + ".mul1"] = np.array(0, dtype="<i4")

    def ngram(self, d: pathlib.Path, gap: bool) -> None:
        K, rows = 3, 128                                  # the heads cover 48: rounded up to a multiple of 128
        base = L + "layers.1.ple.ple_embedding.ngram_embedding."
        sizes = np.full(16, 3, dtype="<i8")
        t = {base + "head_bias": np.zeros((16, 160), dtype="<f2"),
             base + "head_offsets": (np.arange(16) * 3).astype("<i8"), base + "head_vocab_sizes": sizes,
             base + "layer_multipliers": np.array([1, 2, 3], dtype="<i8")}
        words = 1 + 10 * K
        t[base + "shard_0.trellis"] = np.zeros((rows // 2, words), dtype="<i2")
        if gap:
            t[base + "other"] = np.zeros(4, dtype="<i2")
        t[base + "shard_1.trellis"] = np.zeros((rows // 2, words), dtype="<i2")
        write_st(d / Q.NGRAM_FILE, t, {"format": "exl3_ngram_trellis", "codebook": "mul1", "K": str(K),
                                       "row_dim": "160", "rows": str(rows)})


def decoded(st: X.SafeTensors, hf: str) -> np.ndarray:
    tr = st.raw(hf + ".trellis").view(np.int16).reshape(st.t[hf + ".trellis"][2])
    return X.exl3_rows(tr, st.f32(hf + ".suh"), st.f32(hf + ".svh"), X.tile_k2(tr.shape[2], 2, hf), 2)


class TestQwenPack(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.d = pathlib.Path(cls.tmp.name)
        Model(cls.d)
        r = subprocess.run([sys.executable, str(HERE / "qwen" / "pack.py"), "--model", str(cls.d)],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        cls.out = r.stdout
        cls.st = X.SafeTensors(cls.d)
        cls.pack = cls.d / "pack"
        cls.rows = {}
        for line in (cls.pack / "index.txt").read_text().splitlines():
            if line[0] != "#":
                f = line.split()
                cls.rows[f[0]] = f
        cls.mats = {}
        for line in (cls.pack / "exl3.txt").read_text().splitlines():
            if line.startswith("mat "):
                f = line.split()
                cls.mats[f[1]] = f

    @classmethod
    def tearDownClass(cls):
        del cls.st                                            # (its maps, before the folder goes)
        cls.tmp.cleanup()

    def dense(self, name: str) -> np.ndarray:
        f = self.rows[name]
        raw = (self.pack / "dense.bin").read_bytes()[int(f[3]):int(f[3]) + int(f[4])]
        return np.frombuffer(raw, dtype={"2": "<f4", "4": "<u2", "5": "<f2"}[f[2]])

    def test_tiled_order(self):
        self.assertEqual(Q.tiled(2, 4).tolist(), HEADS.tolist())
        self.assertEqual(Q.tiled(16, 48)[:4].tolist(), [0, 3, 6, 9])
        self.assertEqual(Q.tiled(16, 48)[16], 1)

    def test_moving_heads_changes_no_weight(self):
        """The sidecar's matrices are the model's with whole heads moved: their decode is the same floats, moved."""
        st = self.st
        A = L + "layers.0.linear_attn."
        rows = lambda hf: decoded(st, hf)  # noqa: E731
        qkv, qkv2 = rows(A + "in_proj_qkv"), rows(Q.TILED + A + "in_proj_qkv")
        v = qkv[QK:].reshape(N_V, HEAD, -1)[HEADS].reshape(V, -1)
        self.assertTrue(np.array_equal(qkv2[:QK], qkv[:QK]))
        self.assertTrue(np.array_equal(qkv2[QK:], v))
        z, z2 = rows(A + "in_proj_z"), rows(Q.TILED + A + "in_proj_z")
        self.assertTrue(np.array_equal(z2, z.reshape(N_V, HEAD, -1)[HEADS].reshape(V, -1)))
        o, o2 = rows(A + "out_proj"), rows(Q.TILED + A + "out_proj")      # (rows n x k: the V heads are the inputs)
        self.assertTrue(np.array_equal(o2, o.reshape(-1, N_V, HEAD)[:, HEADS].reshape(-1, V)))

    def test_conversions(self):
        st = self.st
        A = L + "layers.0.linear_attn."
        self.assertTrue(np.array_equal(self.dense("blk.0.hc_attn_norm.weight"),
                                       st.f32(L + "layers.0.attn_hyper_connection.hc_norm.weight") + 1))
        self.assertTrue(np.array_equal(self.dense("output_hc_norm.weight"),
                                       st.f32(L + "hyper_connection_mixer.hc_norm.weight") + 1))
        self.assertTrue(np.array_equal(self.dense("blk.0.ssm_norm.weight"), st.f32(A + "norm.weight")),
                        "the GDN's own norm takes no + 1")
        self.assertTrue(np.array_equal(self.dense("blk.0.ssm_a"), -np.exp(st.f32(A + "A_log"))[HEADS]))
        self.assertTrue(np.array_equal(self.dense("blk.0.ssm_dt.bias"), st.f32(A + "dt_bias")[HEADS]))
        alpha = (st.f32(A + "in_proj_a.weight")[HEADS]).reshape(-1)
        self.assertTrue(np.array_equal(self.dense("blk.0.ssm_alpha.weight"), X.to_bf16(alpha)[0]))
        conv = st.f32(A + "conv1d.weight").reshape(CHANNELS, CONV)
        want = np.concatenate((conv[:QK], conv[QK:].reshape(N_V, HEAD, CONV)[HEADS].reshape(-1, CONV)))
        self.assertTrue(np.array_equal(self.dense("blk.0.ssm_conv1d.weight"), want.reshape(-1)))
        self.assertEqual(self.rows["blk.0.ssm_conv1d.weight"][7:9], [str(CONV), str(CHANNELS)])
        qk = decoded(st, L + "layers.1.self_attn.indexer.index_qk_proj")
        self.assertTrue(np.array_equal(self.dense("blk.1.indexer.q_proj.weight"), X.to_bf16(qk[:128].reshape(-1))[0]))
        self.assertTrue(np.array_equal(self.dense("blk.1.indexer.k_proj.weight"), X.to_bf16(qk[128:].reshape(-1))[0]))
        self.assertEqual(self.rows["blk.1.indexer.q_proj.weight"][7:9], [str(HIDDEN), "128"])
        E = L + "layers.1.ple."
        self.assertEqual(self.rows["blk.1.ple_conv1d.weight"][2], "5")
        self.assertTrue(np.array_equal(self.dense("blk.1.ple_conv1d.weight").astype(np.float32),
                                       st.f32(E + "conv1d.weight").reshape(-1)))
        self.assertTrue(np.array_equal(self.dense("blk.1.ple_norm_key.weight"), st.f32(E + "norm_key.weight") + 1))

    def test_index_and_exl3_txt(self):
        kinds = {"blk.0.hc_attn_down.weight": "4", "blk.0.hc_attn_inject.weight": "4", "blk.0.ffn_gate_inp.weight": "4",
                 "blk.0.ffn_gate_inp_shexp.weight": "4", "blk.1.ple_key.weight": "4", "blk.1.attn_q_norm.weight": "2",
                 "blk.0.attn_qkv.weight": "x", "blk.0.ssm_out.weight": "x", "output.weight": "x", "token_embd.weight": "0",
                 "blk.1.attn_q.weight": "x"}
        for name, kind in kinds.items():
            self.assertEqual(self.rows[name][2], kind, name)
        self.assertEqual(self.rows["blk.0.ffn_gate_inp_shexp.weight"][7:9], [str(HIDDEN), "0"])
        files = [ln.split()[2] for ln in (self.pack / "exl3.txt").read_text().splitlines() if ln.startswith("file ")]
        for name in ("blk.0.attn_qkv.weight", "blk.0.attn_gate.weight", "blk.0.ssm_out.weight"):
            self.assertEqual(files[int(self.mats[name][2])], Q.SIDECAR, name + " is read from the tiled copy")
        self.assertNotEqual(files[int(self.mats["blk.1.attn_q.weight"][2])], Q.SIDECAR)
        self.assertEqual(self.mats["blk.0.attn_qkv.weight"][6:8], [str(HIDDEN), str(CHANNELS)])
        text = (self.pack / "exl3.txt").read_text()
        self.assertIn("layer 0 2 3 3 3 %d 128 128 %d" % (HIDDEN, HIDDEN), text)
        self.assertIn("layer 1 2 3 3 3 ", text)
        self.assertNotIn("blk.2", (self.pack / "index.txt").read_text())

    def test_gguf_keys_and_tokenizer(self):
        import gguf_reader as G
        md = G.GGUFFile(self.d / X.GGUF_NAME).metadata
        self.assertEqual(md["general.architecture"], "qwen4exp")
        self.assertEqual(md["qwen4exp.block_count"], 2)
        self.assertEqual(md["qwen4exp.nextn_predict_layers"], 0)
        self.assertEqual(list(md["qwen4exp.ple.layers"]), [1], "ple_layer_ids is 1-based")
        self.assertEqual(list(md["qwen4exp.ple.layer_multipliers"]), [1, 2, 3])
        self.assertEqual(md["tokenizer.ggml.pre"], "qwen35")
        self.assertEqual(md["tokenizer.ggml.eos_token_id"], 4)
        self.assertEqual(md["tokenizer.ggml.padding_token_id"], 3)
        self.assertEqual(list(md["tokenizer.ggml.token_type"]), [1, 1, 1, 3, 3, 3, 4, 5])
        self.assertTrue((self.pack / "tokenizer" / "chat_template.jinja").exists())

    def test_ngram_txt(self):
        kv = dict(ln.split(" ", 1) for ln in (self.pack / "ngram.txt").read_text().splitlines() if ln[0] != "#")
        self.assertEqual(kv["format"], "exl3_ngram_trellis")
        self.assertEqual(kv["file"], "../" + Q.NGRAM_FILE)
        self.assertEqual((kv["K"], kv["row_bytes"], kv["rows"]), ("3", "62", "128"))
        base = L + "layers.1.ple.ple_embedding.ngram_embedding."
        self.assertEqual(int(kv["table_off"]), self.st.t[base + "shard_0.trellis"][3])
        self.assertEqual(int(kv["head_bias_off"]), self.st.t[base + "head_bias"][3])


class TestRefusals(unittest.TestCase):
    def test_shards_not_one_run_of_rows(self):
        with tempfile.TemporaryDirectory() as t:
            d = pathlib.Path(t)
            Model(d, ngram_gap=True)
            st = X.SafeTensors(d)
            with self.assertRaisesRegex(SystemExit, "one run of rows"):
                Q.ngram_table(st, d, TEXT)
            del st

    def test_other_architecture(self):
        with tempfile.TemporaryDirectory() as t:
            d = pathlib.Path(t)
            (d / "config.json").write_text(json.dumps({"architectures": ["Glm5NextForConditionalGeneration"]}))
            r = subprocess.run([sys.executable, str(HERE / "qwen" / "pack.py"), "--model", str(d)],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 1)
            self.assertIn("this packer knows", r.stdout)


if __name__ == "__main__":
    unittest.main()
