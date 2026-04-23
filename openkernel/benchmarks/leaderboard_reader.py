"""Read-only API over leaderboard records on disk.

Records live under ``results/leaderboard/`` as one JSON object per file.
This module provides loading, filtering, and latest-per-problem
deduplication. It is intentionally decoupled from the writer (1.2a) — it
only reads, never mutates.

Schema compatibility follows the reader contract in
``results/leaderboard/SCHEMA.md``.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

READER_MAJOR = 1

_DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "results" / "leaderboard"


def _parse_version(version: str) -> tuple[int, int]:
    major_str, minor_str = version.split(".", 1)
    return int(major_str), int(minor_str)


def _iter_record_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*.json"):
        if path.name.startswith("_"):
            continue
        if "kernels" in path.parts[len(root.parts):-1]:
            continue
        yield path


def load_all(root: Path | str | None = None) -> list[dict]:
    """Load all leaderboard records from ``root``.

    Skips underscore-prefixed files (schema artifacts) and any file
    nested under a ``kernels/`` directory (those are kernel sources,
    not records). Applies the ``schema_version`` reader contract:
    missing → treat as ``"0.0"`` with a warning; matching major →
    accept; unknown major → warn and skip; lower major with no
    migration path → warn and skip.
    """
    base = Path(root) if root is not None else _DEFAULT_ROOT
    if not base.exists():
        return []

    records: list[dict] = []
    for path in _iter_record_files(base):
        try:
            with path.open("r", encoding="utf-8") as f:
                record = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("leaderboard_reader: could not read %s: %s", path, exc)
            continue

        version = record.get("schema_version")
        if version is None:
            logger.warning(
                "leaderboard_reader: %s missing schema_version, treating as '0.0'",
                path,
            )
            version = "0.0"
            record.setdefault("schema_version", "0.0")

        try:
            major, _minor = _parse_version(version)
        except (ValueError, AttributeError):
            logger.warning(
                "leaderboard_reader: %s has malformed schema_version %r, skipping",
                path,
                version,
            )
            continue

        if major > READER_MAJOR:
            logger.warning(
                "leaderboard_reader: %s has unknown major schema_version %s, skipping",
                path,
                version,
            )
            continue
        if major < READER_MAJOR:
            logger.warning(
                "leaderboard_reader: %s has older major schema_version %s and no migration path; skipping",
                path,
                version,
            )
            continue

        records.append(record)

    return records


def filter_records(
    records: list[dict],
    tier: str | None = None,
    hardware: str | None = None,
    date_range: tuple[str, str] | None = None,
    correct_only: bool = True,
) -> list[dict]:
    """Filter leaderboard records by tier, hardware, date range, correctness.

    ``date_range`` is an inclusive ``(start, end)`` pair of ISO-8601
    calendar dates (``YYYY-MM-DD``); string comparison is correct for
    that format. ``correct_only`` defaults to ``True`` because
    leaderboards should exclude broken kernels.
    """
    out: list[dict] = []
    for r in records:
        if tier is not None and r.get("tier") != tier:
            continue
        if hardware is not None and r.get("hardware") != hardware:
            continue
        if date_range is not None:
            start, end = date_range
            date = r.get("date", "")
            if not (start <= date <= end):
                continue
        if correct_only:
            if "correct" not in r:
                logger.warning(
                    "record missing required 'correct' field; dropping (%s)",
                    r.get("problem_id", "<no id>"),
                )
                continue
            if r["correct"] is not True:
                continue
        out.append(r)
    return out


def prior_best_sol(
    problem_id: str,
    hardware: str,
    root: Path | str | None = None,
) -> float:
    """Return the best ``sol_score`` ever recorded for this problem+hardware.

    Considers only correct runs. Returns ``0.0`` when no prior record
    exists (callers treat that as "no prior best, any SOL above the
    floor is a new best"). A missing or non-numeric ``sol_score`` on a
    record contributes ``0.0`` rather than raising.
    """
    records = load_all(root)
    best = 0.0
    for r in records:
        if r.get("problem_id") != problem_id:
            continue
        if r.get("hardware") != hardware:
            continue
        if r.get("correct") is not True:
            continue
        sol = r.get("sol_score", 0.0)
        try:
            sol_f = float(sol) if sol is not None else 0.0
        except (TypeError, ValueError):
            sol_f = 0.0
        if sol_f > best:
            best = sol_f
    return best


def latest_per_problem(records: list[dict]) -> list[dict]:
    """Return the newest record for each ``(problem_id, hardware, config_hash)``.

    Timestamps are parsed as RFC 3339 (``Z``-suffixed). Records missing
    a timestamp are treated as the epoch so any real record wins.
    """
    epoch = datetime.fromisoformat("1970-01-01T00:00:00+00:00")
    best: dict[tuple[str, str, str], tuple[datetime, dict]] = {}
    for r in records:
        key = (r.get("problem_id", ""), r.get("hardware", ""), r.get("config_hash", ""))
        ts_raw = r.get("timestamp")
        if isinstance(ts_raw, str):
            try:
                ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
            except ValueError:
                logger.warning(
                    "record has unparseable timestamp %r; ordering as epoch (%s)",
                    ts_raw, r.get("problem_id", "<no id>"),
                )
                ts = epoch
        else:
            logger.warning(
                "record missing required 'timestamp' field; ordering as epoch (%s)",
                r.get("problem_id", "<no id>"),
            )
            ts = epoch

        current = best.get(key)
        if current is None or ts > current[0]:
            best[key] = (ts, r)

    return [entry[1] for entry in best.values()]
