"""tools/exl3_ngram_fixture.py - EXL3 n-gram table fixtures from exllamav3's own codec, for ple_reader_test.

    <venv with torch>/bin/python -I tools/exl3_ngram_fixture.py --codec <exllamav3>/exllamav3/modules/quant/exl3_lib/ngram_codec.py \\
        --out data/exl3-exllamav3

The codec is loaded from its file alone (it needs torch and nothing of exllamav3's extension, so no GPU).  For each K
1..8: random trellis symbols, scales and head bias vectors, packed by exllamav3's pack_rows and decoded by its
dequant_rows.  Each <out>/ngram_k<K>.bin is
    "EXN1", int32 K, rows, words (1 + 10 K)
    int16   packed [rows][words]     the rows as a table stores them
    int32   heads  [rows]            each row's head, sorted (a table's rows are in head order)
    fp16    bias   [16][160]
    fp16    want   [rows][160]       exllamav3's decode, rounded to fp16 as its kernel rounds it
"""
from __future__ import annotations

import argparse
import importlib.util
import pathlib
import struct

import torch

ROWS_PER_HEAD = 2
HEADS = 16
DIM = 160


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--codec", required=True, help="exllamav3's modules/quant/exl3_lib/ngram_codec.py")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    spec = importlib.util.spec_from_file_location("ngram_codec", a.codec)
    codec = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(codec)
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    book = codec.mul1_codebook("cpu")
    gen = torch.Generator().manual_seed(1234)
    n = ROWS_PER_HEAD * HEADS
    heads = torch.arange(HEADS).repeat_interleave(ROWS_PER_HEAD)
    for K in range(1, 9):
        states = torch.randint(0, 65536, (n, DIM), generator=gen, dtype=torch.int64)
        scales = (torch.rand(n, generator=gen) * 0.2 + 1e-3).to(torch.float16)
        bias = ((torch.rand(HEADS, DIM, generator=gen) - 0.5) * 0.05).to(torch.float16)
        packed = codec.pack_rows(states, scales, K)
        # what unpacking recovers is the ring's, not the random states' upper bits: decode what the table holds
        got, got_scales = codec.unpack_rows(packed, K)
        assert torch.equal(got_scales.view(torch.int16), scales.view(torch.int16))
        assert torch.equal(codec.pack_rows(got, got_scales, K), packed)
        want = codec.dequant_rows(packed, K, book, bias[heads]).to(torch.float16)
        with (out / ("ngram_k%d.bin" % K)).open("wb") as f:
            f.write(b"EXN1" + struct.pack("<iii", K, n, packed.shape[1]))
            f.write(packed.numpy().astype("<i2").tobytes())
            f.write(heads.to(torch.int32).numpy().astype("<i4").tobytes())
            f.write(bias.numpy().astype("<f2").tobytes())
            f.write(want.numpy().astype("<f2").tobytes())
        print("ngram_k%d.bin: %d rows of %d words" % (K, n, packed.shape[1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
