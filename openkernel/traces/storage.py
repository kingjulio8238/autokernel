"""Trace storage: save/load OptimizationTrace as Parquet files.

Layout follows the design in docs/data-and-integrations.md:
    traces/raw/YYYY-MM/session_{session_id}.parquet
"""

from __future__ import annotations

import json
from dataclasses import asdict, fields
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from openkernel.traces.types import IterationTrace, OptimizationTrace


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

_SESSION_META_FIELDS = [
    "session_id",
    "timestamp",
    "problem_id",
    "problem_source",
    "hardware",
    "backend",
    "model_id",
    "openkernel_version",
    "final_speedup",
    "final_correct",
    "total_iterations",
    "total_tokens",
    "total_cost_usd",
    "total_time_seconds",
]

# Fields that are lists and stored as JSON strings in Parquet
_LIST_META_FIELDS = [
    "strategies_tried",
    "strategies_succeeded",
    "skills_retrieved",
    "skills_created",
]


def _flatten_trace(trace: OptimizationTrace) -> list[dict]:
    """Flatten an OptimizationTrace into one row per iteration.

    Session-level metadata is duplicated on every row so each row is
    self-contained (standard practice for columnar training data).
    """
    meta = {}
    for key in _SESSION_META_FIELDS:
        meta[key] = getattr(trace, key)
    for key in _LIST_META_FIELDS:
        meta[key] = json.dumps(getattr(trace, key))

    if not trace.iterations:
        # Still store a single row with session metadata even if no iterations
        row = dict(meta)
        empty_it = asdict(IterationTrace(iteration=0, intent=""))
        empty_it["profile_data"] = json.dumps(empty_it.get("profile_data", {}))
        # Mark iteration as None so unflatten skips it
        empty_it["iteration"] = None  # type: ignore[assignment]
        row.update(empty_it)
        return [row]

    rows: list[dict] = []
    for it in trace.iterations:
        row = dict(meta)
        it_dict = asdict(it)
        # Serialize profile_data dict as JSON string
        it_dict["profile_data"] = json.dumps(it_dict.get("profile_data", {}))
        row.update(it_dict)
        rows.append(row)
    return rows


def _unflatten_rows(rows: list[dict]) -> OptimizationTrace:
    """Reconstruct an OptimizationTrace from flattened Parquet rows."""
    if not rows:
        return OptimizationTrace()

    first = rows[0]

    trace = OptimizationTrace()
    for key in _SESSION_META_FIELDS:
        setattr(trace, key, first.get(key, getattr(trace, key)))
    for key in _LIST_META_FIELDS:
        raw = first.get(key, "[]")
        setattr(trace, key, json.loads(raw) if isinstance(raw, str) else raw)

    iteration_field_names = {f.name for f in fields(IterationTrace)}
    for row in rows:
        it_data = {k: v for k, v in row.items() if k in iteration_field_names}
        # Deserialize profile_data
        pd_raw = it_data.get("profile_data", "{}")
        it_data["profile_data"] = json.loads(pd_raw) if isinstance(pd_raw, str) else pd_raw
        # Skip sentinel rows that have no real iteration data
        if it_data.get("iteration") is None:
            continue
        trace.iterations.append(IterationTrace(**it_data))

    return trace


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def save_trace(
    trace: OptimizationTrace,
    output_dir: str | Path = "traces/raw",
) -> Path:
    """Save an OptimizationTrace as a Parquet file.

    Creates monthly subdirectory (YYYY-MM/) and writes
    session_{session_id}.parquet.

    Returns the path to the written file.
    """
    output_dir = Path(output_dir)
    ts = trace.timestamp or datetime.now(timezone.utc).isoformat()
    month_str = ts[:7]  # "YYYY-MM"
    subdir = output_dir / month_str
    subdir.mkdir(parents=True, exist_ok=True)

    filepath = subdir / f"session_{trace.session_id}.parquet"

    rows = _flatten_trace(trace)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, filepath, compression="snappy")

    return filepath


def load_trace(path: str | Path) -> OptimizationTrace:
    """Read a Parquet file back into an OptimizationTrace."""
    table = pq.read_table(Path(path))
    rows = table.to_pylist()
    return _unflatten_rows(rows)


def list_traces(traces_dir: str | Path = "traces/raw") -> list[Path]:
    """List all trace Parquet files under traces_dir, sorted by name."""
    traces_dir = Path(traces_dir)
    if not traces_dir.exists():
        return []
    return sorted(traces_dir.rglob("session_*.parquet"))
