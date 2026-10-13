#!/usr/bin/env python3
"""Runs the agent's analysis code in a separate process over its experiment CSVs."""
from __future__ import annotations

import multiprocessing as mp
import re
from typing import Any, Dict, List, Optional, Tuple

CODE_TIMEOUT_SECONDS = 30
MAX_OUTPUT_CHARS = 6000
_MAX_VAR_PICKLE_BYTES = 200_000
_MAX_NDARRAY_ELEMS = 200_000
_FENCE_RE = re.compile(r"^```(?:python)?\s*|\s*```$", re.MULTILINE)


def _worker(code: str, csv_map: Dict[str, str], carried: Dict[str, Any], result_q) -> None:
    import os
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import contextlib
    import io
    import pickle
    import traceback

    import tempfile
    try:
        _wd = os.path.dirname(next(iter(csv_map.values()))) if csv_map else tempfile.mkdtemp(prefix="rpg_code_")
        os.chdir(_wd)
    except Exception:
        pass
    import numpy as _np
    import pandas as _pd
    try:
        import scipy.stats as _stats
    except Exception:
        _stats = None

    ns: Dict[str, Any] = {"pd": _pd, "np": _np, "stats": _stats}
    _loaded_dfs = []
    for var_name, path in csv_map.items():
        ns[var_name] = path
        df_name = var_name[:-4] + "_df" if var_name.endswith("_csv") else var_name + "_df"
        try:
            ns[df_name] = _pd.read_csv(path)
            _loaded_dfs.append(df_name)
        except Exception:
            pass
    ns.update(carried)

    injected = set(ns.keys())
    buf = io.StringIO()
    error = None
    try:
        with contextlib.redirect_stdout(buf):
            exec(code, ns)
    except Exception:
        error = traceback.format_exc()

    out = buf.getvalue()
    if error:
        out += f"\n[PYTHON ERROR]\n{error}"

    new_vars: Dict[str, Any] = {}
    for k, v in ns.items():
        if k in injected or k.startswith("__"):
            continue
        if isinstance(v, (_pd.DataFrame, _pd.Series)):
            continue
        if isinstance(v, _np.ndarray) and v.size > _MAX_NDARRAY_ELEMS:
            continue
        try:
            if len(pickle.dumps(v)) <= _MAX_VAR_PICKLE_BYTES:
                new_vars[k] = v
        except Exception:
            continue
    result_q.put((out, new_vars))


def run_code(code: str, csv_map: Dict[str, str], carried: Optional[Dict[str, Any]] = None,
             *, timeout: int = CODE_TIMEOUT_SECONDS,
             max_chars: int = MAX_OUTPUT_CHARS) -> Tuple[str, Dict[str, Any]]:
    code = _FENCE_RE.sub("", code).replace("```", "").strip()
    carried = carried or {}
    ctx = mp.get_context("spawn")
    result_q = ctx.Queue()
    proc = ctx.Process(target=_worker, args=(code, csv_map, carried, result_q))
    proc.start()
    import queue as _queue
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
    if len(out) > max_chars:
        out = out[:max_chars] + f"\n[...output truncated at {max_chars} chars]"
    return (out, new_vars)
