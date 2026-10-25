#!/usr/bin/env python3
"""Hardened code sandbox for RL training (priority 0.5, 2026-10-23).

The release sandbox (`bench/code_sandbox.py`) runs agent code in a fresh folder behind a Python audit hook that
refuses file access outside that folder, subprocesses and socket connects. `test_sandbox_isolation.py` found two
holes an RL policy could in principle learn to use (a benchmark agent never tried):
  1. ctypes: `ctypes.CDLL(None).fopen(world_file)` reads any file through libc, below the audit hook -> the gold.
  2. the environment: `os.environ` / `/proc/self/environ` show the trainer's environment (keys, paths).
This module wraps the release worker; it does not change it. The child process first scrubs its environment and adds
a second audit hook (hooks can be added, never removed), then calls the release `_worker` unchanged:
  - ctypes may not look up (dlsym, `in_dll`, attribute access on a CDLL) any libc/syscall entry point that opens
    files, runs programs, opens sockets or reads the environment (`_DENY_SYM`). This holds whoever asks, so it can't
    be dodged by compiling code under a fake file name or importing a module written into the analysis folder;
    libraries that use ctypes for other symbols (numpy, threadpoolctl/sklearn BLAS lookups) are unaffected;
  - /proc/<pid>/{environ,mem,pagemap} cannot be opened; os.environ holds only a small allowlist; posix.environ
    (the startup snapshot) is emptied;
  - UDP sends and binds are refused (socket.connect is already refused by the release hook).
`test_sandbox_equivalence.py` replays every code turn of the stored benchmark episodes through both sandboxes and
requires identical output, so for legitimate analysis code the hardened sandbox IS the benchmark sandbox.
`install()` swaps it into the frozen bench (`code_sandbox.run_code`, read by run_world at call time).
"""
from __future__ import annotations

import multiprocessing as mp
import os
import re
from typing import Any, Dict, Optional, Tuple

_ENV_KEEP = ("PATH", "LANG", "LC_ALL", "HOME", "TMPDIR", "PYTHONHASHSEED", "OMP_NUM_THREADS",
             "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "MPLCONFIGDIR", "PYTHONPATH")
_PROC_SECRET_RE = re.compile(r"^/proc/(?:\d+|self|thread-self)(?:/task/\d+)?/(?:environ|mem|pagemap)$")


# libc / syscall entry points that reach files, programs, sockets or the environment (full-name match)
_DENY_SYM = re.compile(r"^_*(?:open|open64|openat|openat64|open_2|openat_2|fopen|fopen64|freopen|freopen64|fdopen|"
                       r"creat|creat64|syscall|name_to_handle_at|open_by_handle_at|opendir|fdopendir|scandir|"
                       r"dlopen|dlmopen|dlsym|dlvsym|system|popen|exec\w*|fork|vfork|clone|clone3|posix_spawnp?|"
                       r"socket|socketpair|connect|sendto|sendmsg|bind|getenv|secure_getenv|environ|"
                       r"PyRun_\w*|Py_RunMain|PyImport_\w*|PySys_\w*|_Py_\w*)$")


def _sym_name(args) -> str:
    for a in args or ():
        if isinstance(a, (str, bytes)):
            return os.fsdecode(a)
    return ""


def _hardened_worker(code, csv_map, carried, result_q, wd) -> None:
    import posix
    import sys
    keep = {k: os.environ[k] for k in _ENV_KEEP if k in os.environ}
    os.environ.clear()
    os.environ.update(keep)
    try:
        posix.environ.clear()
    except Exception:
        pass

    def _guard2(event, args):
        if event in ("ctypes.dlsym", "ctypes.dlsym/handle") and _DENY_SYM.match(_sym_name(args)):
            raise PermissionError("this ctypes call is not allowed in analysis code")
        if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
            try:
                rp = os.path.realpath(os.fsdecode(args[0]))
            except Exception:
                rp = ""
            if _PROC_SECRET_RE.match(rp) or _PROC_SECRET_RE.match(os.fsdecode(args[0]) if not isinstance(args[0], os.PathLike) else ""):
                raise PermissionError(f"file access outside the analysis folder is not allowed: {args[0]}")
        if event in ("socket.sendto", "socket.sendmsg", "socket.bind"):
            raise PermissionError(f"{event} is not allowed in analysis code")
    sys.addaudithook(_guard2)
    from code_sandbox import _worker
    _worker(code, csv_map, carried, result_q, wd)


def run_code(code: str, csv_map: Dict[str, str], carried: Optional[Dict[str, Any]] = None,
             *, timeout: Optional[int] = None, max_chars: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
    """The release run_code (bench/code_sandbox.py) with only the worker target swapped. Kept line-for-line in sync
    with it; `test_sandbox_equivalence.py` checks the outputs."""
    import code_sandbox as cs
    import queue as _queue
    import shutil
    import tempfile
    timeout = cs.CODE_TIMEOUT_SECONDS if timeout is None else timeout
    max_chars = cs.MAX_OUTPUT_CHARS if max_chars is None else max_chars
    code = cs._FENCE_RE.sub("", code).replace("```", "").strip()
    carried = carried or {}
    wd = tempfile.mkdtemp(prefix="rpg_code_")
    if SANDBOX_PY:
        try:
            out, new_vars = _run_on_sandbox_py(code, csv_map, carried, wd, timeout)
        finally:
            shutil.rmtree(wd, ignore_errors=True)
        if out.startswith("[TIMEOUT") or out.startswith("[NO OUTPUT"):
            return (out, new_vars)
        return (_cut(out, max_chars), new_vars)
    ctx = mp.get_context("spawn")
    result_q = ctx.Queue()
    proc = ctx.Process(target=_hardened_worker, args=(code, csv_map, carried, result_q, wd))
    proc.start()
    try:
        try:
            out, new_vars = result_q.get(timeout=timeout)
        except _queue.Empty:
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5)
                return (f"[TIMEOUT: execution exceeded {timeout}s]", {})
            return ("[NO OUTPUT: worker died without returning a result]", {})
        except Exception:
            proc.join(timeout=5)
            return ("[NO OUTPUT: worker died without returning a result]", {})
        proc.join(timeout=5)
        if proc.is_alive():
            proc.terminate()
    finally:
        shutil.rmtree(wd, ignore_errors=True)
    return (_cut(out, max_chars), new_vars)


def _cut(out: str, max_chars: int) -> str:
    if len(out) > max_chars:
        i = out.find("\n[PYTHON ERROR]\n")
        err = out[i:] if i >= 0 else ""
        body = out[:i] if i >= 0 else out
        out = (body[:max_chars] + f"\n[OUTPUT CUT: your code printed {len(body)} characters; only the first "
               f"{max_chars} are shown. Print less (e.g. summaries or .head()) to see everything.]" + err)
    return out


# --- separate interpreter (RLV10_SANDBOX_PY): see sandbox_runner.py -------------------------------------------------
SANDBOX_PY = os.environ.get("RLV10_SANDBOX_PY", "")
_HERE = os.path.dirname(os.path.abspath(__file__))


class Pickled:
    """A carried variable kept as the sandbox interpreter's pickle; the parent never unpickles agent objects."""
    __slots__ = ("b",)

    def __init__(self, b: bytes):
        self.b = b

    def __repr__(self):
        return f"<carried variable, {len(self.b)} bytes>"


def _run_on_sandbox_py(code, csv_map, carried, wd, timeout):
    import pickle
    import subprocess
    import code_sandbox as cs
    bench = os.path.dirname(os.path.abspath(cs.__file__))
    enc = {}
    for k, v in (carried or {}).items():
        if isinstance(v, Pickled):
            enc[k] = v.b
        else:                                   # a value that did not come from this sandbox (not expected)
            try:
                enc[k] = pickle.dumps(v)
            except Exception:
                pass
    payload = pickle.dumps({"code": code, "csv_map": dict(csv_map), "carried": enc, "wd": wd})
    env = {k: os.environ[k] for k in _ENV_KEEP if k in os.environ and k != "PYTHONPATH"}
    proc = subprocess.Popen([SANDBOX_PY, "-s", "-P", os.path.join(_HERE, "sandbox_runner.py"), bench, _HERE],
                            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, env=env, cwd=wd)
    try:
        proc.communicate(payload, timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
        return (f"[TIMEOUT: execution exceeded {timeout}s]", {})
    res = os.path.join(wd, ".rlv10_result.pkl")
    try:
        with open(res, "rb") as f:
            out, new = pickle.load(f)
    except Exception:
        return ("[NO OUTPUT: worker died without returning a result]", {})
    return (out, {k: Pickled(b) for k, b in new.items()})


_ORIGINAL = None


def install() -> None:
    """Route run_world's code turns through the hardened sandbox (idempotent)."""
    global _ORIGINAL
    import code_sandbox as cs
    if _ORIGINAL is None:
        _ORIGINAL = cs.run_code
    cs.run_code = run_code


def uninstall() -> None:
    import code_sandbox as cs
    if _ORIGINAL is not None:
        cs.run_code = _ORIGINAL
