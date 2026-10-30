"""tools/exl3_logits_oracle.py - exllamav3's own logits for a token sequence, to compare the engine with.

    <exllamav3 venv>/bin/python -I tools/exl3_logits_oracle.py --model <EXL3 dir> --tokens ids.txt --out x3.logits
    build/maya --pack ... --tokens $(cat ids.txt) --max-new 1 --greedy --dump-logits maya.logits
    python tools/exl3_logits_oracle.py --compare x3.logits maya.logits --split 2051

ids.txt: comma-separated token ids.  Both files are as maya's --dump-logits writes them: int32 vocab, rows, then F32
[rows][vocab]; row i is the logits after token i, so the last token's row is not written.  --topk-blocks N replaces the
QSA indexers' budget (a small one makes the selection dominate the logits, a test the real 512 blocks do not give).
"""
from __future__ import annotations

import argparse

import numpy as np


def run(a) -> int:
    import torch
    from exllamav3 import Config, Model
    from exllamav3.modules.qsa_indexer import QSAIndexer
    ids = [int(x) for x in open(a.tokens).read().split(",")]
    model = Model.from_config(Config.from_directory(a.model))
    model.load(max_chunk_size=max(2048, len(ids) + 8), max_output_size=len(ids) + 8, reserve_per_device=[1.0, 1.0, 1.0])
    if a.topk_blocks:
        seen = 0

        def walk(mods):
            nonlocal seen
            for m in mods:
                for v in vars(m).values():
                    for w in v if isinstance(v, (list, tuple)) else [v]:
                        if isinstance(w, QSAIndexer):
                            w.block_topk = a.topk_blocks
                            seen += 1
                walk(getattr(m, "modules", None) or [])
        walk(model.modules)
        print("%d QSA indexers at %d blocks" % (seen, a.topk_blocks))
    with torch.inference_mode():
        y = model.forward(torch.tensor([ids], dtype=torch.long), {"attn_mode": "flash_attn_nc"})
    y = y[0].float().cpu().numpy()[:len(ids) - 1]
    with open(a.out, "wb") as f:
        f.write(np.array([y.shape[1], y.shape[0]], dtype="<i4").tobytes())
        f.write(y.astype("<f4").tobytes())
    print("%s: %d rows" % (a.out, len(y)))
    return 0


def read(path: str) -> np.ndarray:
    vocab = int(np.fromfile(path, dtype="<i4", count=1)[0])
    return np.memmap(path, dtype="<f4", mode="r", offset=8).reshape(-1, vocab)


def log_softmax(a: np.ndarray) -> np.ndarray:
    a = a - a.max()
    return a - np.log(np.exp(a).sum())


def compare(paths, splits) -> int:
    p, q = read(paths[0]), read(paths[1])
    n = min(len(p), len(q))
    edges = [0] + [s for s in splits if s < n] + [n]
    for lo, hi in zip(edges, edges[1:]):
        kl, cos, same = [], [], 0
        for i in range(lo, hi):
            a, b = np.asarray(p[i], dtype=np.float64), np.asarray(q[i], dtype=np.float64)
            la, lb = log_softmax(a), log_softmax(b)
            kl.append(float((np.exp(la) * (la - lb)).sum()))
            cos.append(float(a @ b / np.linalg.norm(a) / np.linalg.norm(b)))
            same += int(a.argmax() == b.argmax())
        print("rows %5d-%-5d top1 %5d/%d (%.2f%%)  KL mean %.5f med %.5f p99 %.4f max %.3f  cos med %.5f min %.4f" % (
            lo, hi, same, hi - lo, 100.0 * same / (hi - lo), np.mean(kl), np.median(kl), np.percentile(kl, 99), np.max(kl),
            np.median(cos), np.min(cos)))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model")
    ap.add_argument("--tokens")
    ap.add_argument("--out")
    ap.add_argument("--topk-blocks", type=int, default=0)
    ap.add_argument("--compare", nargs=2, metavar=("REFERENCE", "OTHER"))
    ap.add_argument("--split", type=int, action="append", default=[], help="report the rows on either side of this one")
    a = ap.parse_args()
    return compare(a.compare, sorted(a.split)) if a.compare else run(a)


if __name__ == "__main__":
    raise SystemExit(main())
