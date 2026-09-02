"""Small, dependency-free artifact helpers."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable


def json_default(value: Any):
    """Serialize numpy scalar/array values without importing numpy eagerly."""
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        default=json_default,
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            tmp_name = handle.name
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if tmp_name and os.path.exists(tmp_name):
            os.unlink(tmp_name)


def atomic_write_json(path: Path, value: Any, *, indent: int = 2) -> None:
    atomic_write_text(
        path,
        json.dumps(value, indent=indent, ensure_ascii=False, default=json_default) + "\n",
    )


def atomic_write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    text = "".join(
        json.dumps(record, ensure_ascii=False, default=json_default) + "\n"
        for record in records
    )
    atomic_write_text(path, text)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: JSONL record is not an object")
            records.append(value)
    return records


def completed_rollout_records(path: Path) -> list[dict[str, Any]] | None:
    """Read one complete JSONL artifact once, returning all of its records."""
    try:
        records = read_jsonl(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if not records or records[-1].get("record_type") != "terminal":
        return None
    return records if records[-1].get("complete") is True else None


def completed_final_record(path: Path) -> dict[str, Any] | None:
    records = completed_rollout_records(path)
    return records[-1] if records is not None else None


def require_finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} is not numeric: {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} is not finite: {value!r}")
    return number
