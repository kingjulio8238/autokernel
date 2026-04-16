"""Data layer for the kernel code dashboard.

Reads JSON session cache files and converts to pandas DataFrames
for use by the Plotly Dash panels.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_SESSIONS_DIR = _PROJECT_ROOT / "cache" / "sessions"


def load_session(session_id: str) -> pd.DataFrame:
    """Load a session JSON file and return iterations as a DataFrame.

    Args:
        session_id: The session ID (filename stem) in cache/sessions/.

    Returns:
        DataFrame with one row per iteration, including flattened profile columns.
    """
    session_path = _SESSIONS_DIR / f"{session_id}.json"
    if not session_path.exists():
        raise FileNotFoundError(f"Session not found: {session_path}")

    data = json.loads(session_path.read_text())
    iterations = data.get("iterations", [])

    if not iterations:
        return pd.DataFrame()

    rows = []
    for it in iterations:
        profile = it.get("profile", {})
        row = {
            "iteration": it.get("iteration"),
            "speedup": it.get("speedup", 0.0),
            "status": it.get("status", ""),
            "decision": it.get("decision", ""),
            "runtime_us": it.get("runtime_us", 0.0),
            "ref_runtime_us": it.get("ref_runtime_us", 0.0),
            "intent": it.get("intent", ""),
            "error": it.get("error"),
            "bandwidth_util": profile.get("bandwidth_util", 0.0),
            "compute_util": profile.get("compute_util", 0.0),
            "cache_efficiency": profile.get("cache_efficiency", 0.0),
            "occupancy": profile.get("occupancy", 0.0),
            "bottleneck_type": profile.get("bottleneck_type", "unknown"),
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Add derived columns
    df["cumulative_best"] = df["speedup"].cummax()
    df["cost_estimate"] = df["iteration"] * 0.02  # ~$0.02 per iteration

    return df


def load_session_metadata(session_id: str) -> dict:
    """Load session-level metadata (hardware, backend, model, etc.)."""
    session_path = _SESSIONS_DIR / f"{session_id}.json"
    if not session_path.exists():
        raise FileNotFoundError(f"Session not found: {session_path}")

    data = json.loads(session_path.read_text())
    return {
        "session_id": data.get("session_id", session_id),
        "problem": data.get("problem", ""),
        "hardware": data.get("hardware", ""),
        "backend": data.get("backend", ""),
        "model": data.get("model", ""),
        "ref_runtime_us": data.get("ref_runtime_us", 0.0),
        "best_speedup": data.get("best_speedup", 0.0),
        "num_iterations": data.get("num_iterations", 0),
    }


def list_sessions() -> list[str]:
    """List available session IDs."""
    if not _SESSIONS_DIR.exists():
        return []
    return [p.stem for p in _SESSIONS_DIR.glob("*.json")]
