#!/usr/bin/env python3
"""tools/test_alloc_recovery.py - the engine survives an allocation failure.

An out-of-memory moment in the prompt path used to end the process (std::exit(1) in the prefill's checks,
std::abort() in the MMQ scratch pool): the client got a 503 and the next request waited for the engine to load
again.  Now the failure ends the request with an error and the engine lives.

Three test hooks make the failures happen on purpose:

  STRATA_MMQ_POOL_FAIL=<n>        the prompt path's MMQ scratch pool cannot allocate the first n times (the case
                                  that used to abort the engine mid-prompt)
  STRATA_PREALLOC_FAIL=<tokens> the prompt path's layout of a request's chunk bigger than <tokens> does not fit
                                  (the case the recovery retries with a smaller chunk)
  STRATA_X3_PROMPT_FAIL=<n>       an EXL3 pack's n-th call for a layer's prompt experts fails, once

A hook is only reached on its own path, so a test whose path the installed model does not take skips: an EXL3 pack never
runs MMQ, and a prompt path with buffers of its own never lays out borrowed ones.

The test needs the installed model and a GPU; without them it skips.
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import time
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PROMPT_TOKENS = 2048
# Qwen3.8-Flash-Next's layers with experts: a chunk calls for each one's prompt experts, the last on a split's last GPU
QWEN_EXL3_EXPERT_LAYERS = 48


def is_exl3_pack(cfg: dict) -> bool:
    """As the engine tells: a --pack (not a --glm-pack) with an exl3.txt."""
    args = cfg.get("args") or []
    return "--glm-pack" not in args and "--pack" in args[:-1] and \
        (pathlib.Path(args[args.index("--pack") + 1]) / "exl3.txt").exists()


def config_path() -> pathlib.Path | None:
    """The installed config (the newest one), with its engine and tokenizer."""
    found = sorted(ROOT.glob("maya-*.json"))
    for path in reversed(found):
        try:
            cfg = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if cfg.get("exe") and pathlib.Path(cfg["exe"]).exists():
            return path
    return None


def gpu_present() -> bool:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20)
        return out.returncode == 0 and out.stdout.strip() != ""
    except (OSError, subprocess.SubprocessError):
        return False


def tokens_of(lines: list[str]) -> list[str]:
    """A request's answer: its T lines."""
    return [line for line in lines if line.startswith("T ")]


class Engine:
    """The engine in its --serve mode, driven over its stdin/stdout (the way the server drives it)."""

    def __init__(self, cfg_path: pathlib.Path, env: dict):
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        import serve.server as SV

        self.log_path = ROOT / "maya-alloc-test.log"
        self.env = dict(os.environ)
        self.env.update(env)
        child = dict(SV.child_env(cfg))
        child.update(self.env)   # the test's hooks reach the engine
        self.log = open(self.log_path, "w", encoding="utf-8")
        self.proc = subprocess.Popen([cfg["exe"], "--serve"] + SV.engine_args(cfg), stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=self.log, text=True,
                                     env=child, cwd=cfg.get("cwd") or str(ROOT), bufsize=1)
        self.cfg = cfg

    def wait_ready(self, timeout: float = 600.0) -> bool:
        end = time.time() + timeout
        while time.time() < end:
            line = self.proc.stdout.readline()
            if not line:
                return False
            if line.startswith("READY"):
                return True
        return False

    def gen(self, ids: list[int], n_new: int = 8) -> list[str]:
        self.proc.stdin.write(f"GEN {n_new} temperature=0 " + ",".join(map(str, ids)) + "\n")
        self.proc.stdin.flush()
        out = []
        while True:
            line = self.proc.stdout.readline()
            if not line:
                return out
            out.append(line.strip())
            if line.startswith("DONE") or line.startswith("ERR"):
                return out

    def alive(self) -> bool:
        return self.proc.poll() is None

    def stop(self) -> None:
        if self.log.closed:    # (stopped already: a test that needs the GPU back before its cleanup)
            return
        try:
            self.proc.stdin.write("QUIT\n")
            self.proc.stdin.flush()
            self.proc.wait(timeout=60)
        except (OSError, subprocess.SubprocessError):
            self.proc.kill()
            self.proc.wait()
        finally:
            self.log.close()
            for pipe in (self.proc.stdin, self.proc.stdout):
                try:
                    pipe.close()
                except OSError:
                    pass

    def log_text(self) -> str:
        """The engine's log (closed first: it is still writing it)."""
        self.log.flush()
        return self.log_path.read_text(encoding="utf-8", errors="replace")


@unittest.skipIf(config_path() is None, "no installed model (run ./maya.sh --setup)")
@unittest.skipIf(not gpu_present(), "no NVIDIA GPU")
class AllocRecovery(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT / "tools"))
        import strata_tokenizer as ST

        cfg_path = config_path()
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        args = cfg.get("args") or []
        pack = pathlib.Path(args[args.index("--glm-pack") + 1]) if "--glm-pack" in args[:-1] else None
        tp = pathlib.Path(cfg.get("tokenizer") or (pack / "tokenizer" if pack else ""))
        tok = ST.Tokenizer.from_dir(tp)
        # a prompt of PROMPT_TOKENS tokens (the same text for every run)
        text = "\n\n".join(p.read_text(encoding="utf-8", errors="replace")
                           for p in [ROOT / "README.md"] + sorted((ROOT / "docs").glob("*.md")) if p.exists())
        ids = tok.encode(text)
        cls.ids = ids[:PROMPT_TOKENS]
        cls.cfg_path = cfg_path
        cls.exl3 = is_exl3_pack(cfg)
        cls.answer = []
        # the engine needs the whole GPU: with another Maya (or a test) running it cannot load, and the test would
        # fail for a reason that is not its own - so it skips
        plain = Engine(cls.cfg_path, {})
        try:
            if not plain.wait_ready():
                raise unittest.SkipTest("the engine did not load (the GPU's memory is held by something else)")
            # (the engine says this before READY)
            cls.own_buffers = "the prompt path allocates its own buffers" in plain.log_text()
            if cls.exl3:
                cls.answer = tokens_of(plain.gen(cls.ids, 8))
        finally:
            plain.stop()

    def engine(self, env: dict) -> Engine:
        e = Engine(self.cfg_path, env)
        self.addCleanup(e.stop)
        self.assertTrue(e.wait_ready(), "the engine did not load")
        return e

    def test_the_scratch_pool_runs_out_and_the_engine_lives(self):
        """The MMQ scratch pool cannot allocate: the request ends with an error, the engine answers the next one."""
        if self.exl3:
            self.skipTest("an EXL3 pack's prompt path does not run MMQ")
        e = self.engine({"STRATA_MMQ_POOL_FAIL": "1"})
        out = e.gen(self.ids, 8)
        self.assertTrue(any(line.startswith("ERR") for line in out),
                        f"the request should have failed with an error, not an answer: {out}")
        self.assertTrue(e.alive(), "the engine died: the allocation failure must not end the process")
        # the next request runs: the engine is not broken (the hook's one failure is used up)
        out2 = e.gen(self.ids, 8)
        self.assertTrue(any(line.startswith("DONE") for line in out2), f"the engine did not answer: {out2}")

    def test_a_chunk_that_does_not_fit_is_read_again_smaller(self):
        """A layout that does not fit: the request is read again with a smaller chunk and answered."""
        if self.own_buffers:
            self.skipTest("this prompt path has buffers of its own: it borrows nothing to lay out")
        e = self.engine({"STRATA_PREALLOC_FAIL": str(PROMPT_TOKENS // 2)})
        out = e.gen(self.ids, 8)
        self.assertTrue(any(line.startswith("DONE") for line in out),
                        f"the request should have been read with a smaller chunk: {out}")
        self.assertTrue(e.alive(), "the engine died")
        self.assertIn("the chunk is", e.log_text(), "the engine did not report the smaller chunk")

    def test_exl3_prompt_experts_fail_and_the_engine_lives(self):
        """A layer's prompt experts fail, on the first layer and on the last: the request ends with an error, the engine
        lives, and the next request's answer is the one an engine that never failed gives."""
        if not self.exl3:
            self.skipTest("not an EXL3 pack")
        self.assertTrue(self.answer, "the plain engine gave no answer to compare with")
        for call in (1, QWEN_EXL3_EXPERT_LAYERS):
            with self.subTest(call=call):
                e = self.engine({"STRATA_X3_PROMPT_FAIL": str(call)})
                try:
                    out = e.gen(self.ids, 8)
                    self.assertTrue(any(line.startswith("ERR") for line in out),
                                    f"the request should have failed with an error, not an answer: {out}")
                    self.assertIn("STRATA_X3_PROMPT_FAIL", e.log_text(), "the hook did not fire")
                    self.assertTrue(e.alive(), "the engine died: the failure must not end the process")
                    out2 = e.gen(self.ids, 8)
                    self.assertTrue(any(line.startswith("DONE") for line in out2), f"the engine did not answer: {out2}")
                    self.assertEqual(tokens_of(out2), self.answer, "the answer after the failure is not the plain one")
                finally:
                    e.stop()


if __name__ == "__main__":
    unittest.main()
