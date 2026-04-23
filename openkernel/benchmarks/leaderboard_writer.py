"""Atomic, schema-valid leaderboard record writer.

Persists one JSON record per winning kernel under
``results/leaderboard/YYYY-MM-DD/{problem_id}_{hardware}_{config_hash}.json``
with a companion source file at
``results/leaderboard/kernels/{kernel_hash}.py``.

Writes are atomic (temp + ``os.rename``) so a crash mid-write never leaves
a partial record behind. Concurrent writers targeting the same uniqueness
tuple ``(problem_id, hardware, date, config_hash)`` are safe: each uses
its own pid-tagged temp file and the last rename wins — no corruption.

The 22-field schema (``schema_version == "1.0"``) is defined in
``results/leaderboard/SCHEMA.md``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openkernel.benchmarks.problem_spec import ProblemSpec

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0"

_VALID_TIERS = {"L1", "L2", "GPU_MODE"}
_VALID_HARDWARE = {"L40S", "H100", "A100-80GB", "B200"}
_VALID_BOTTLENECKS = {"compute-bound", "memory-bound", "balanced", "unknown"}
_VALID_KERNEL_EXT = {".py", ".cu", ".triton"}

# Placeholder source emitted by kernel_code/batch_optimizer.py for failure
# records (correct=False, speedup=0.0). Valid ONLY when correct=False.
_FAILURE_PLACEHOLDER = "# No correct kernel produced for this attempt.\n"

# Op tokens scanned in the first 2000 chars of kernel_source. If one of
# these tokens appears in the kernel's leading docstring/comments but is
# absent from spec.reference_source (and from spec.name/spec.id), that is
# strong evidence the kernel was generated for a DIFFERENT problem — the
# env-var-leak class of bug. Regex patterns use word boundaries where
# meaningful to cut false positives from substrings like "assortment".
_OP_TOKENS: tuple[tuple[str, str], ...] = (
    ("histogram", r"\bhistogram\b"),
    ("cumsum", r"\bcumsum\b"),
    ("prefix sum", r"\bprefix[\s-]?sum\b"),
    ("prefixsum", r"\bprefixsum\b"),
    ("matmul", r"\bmatmul\b"),
    ("softmax", r"\bsoftmax\b"),
    ("layernorm", r"\blayer[\s-]?norm\b"),
    ("conv2d", r"\bconv2d\b"),
    ("vectoradd", r"\bvector[\s-]?add\b"),
    ("vectorsum", r"\bvector[\s-]?sum\b"),
    ("grayscale", r"\bgrayscale\b"),
)

_REQUIRED_RESULT_KEYS: dict[str, type | tuple[type, ...]] = {
    "kernel_source": str,
    "hardware": str,
    "model": str,
    "speedup": (int, float),
    "sol_score": (int, float),
    "compute_util": (int, float),
    "bandwidth_util": (int, float),
    "bottleneck_type": str,
    "correct": bool,
    "cost_usd": (int, float),
    "elapsed_s": int,
    "stop_reason": str,
    "rounds": int,
    "iterations": int,
}


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _kernel_hash(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:16].lower()


def _config_hash(config: dict) -> str:
    digest = hashlib.sha256(_canonical_json(config).encode("utf-8")).hexdigest()[:12]
    return f"cfg_{digest}"


def _check_kernel_identity(
    spec: ProblemSpec,
    result: dict,
    date_dir: Path,
    khash: str,
) -> None:
    """Integrity defense against cross-problem kernel leakage.

    Raises ValueError when the kernel being written looks like it was
    generated for a different problem than ``spec``. Three guards:

    1. Placeholder source + ``correct=True`` — the failure-record
       placeholder is only valid for failure records.
    2. Op-token docstring mismatch — an op name in the kernel's leading
       docstring/comments that is absent from ``spec.reference_source``
       AND from ``spec.name``/``spec.id``. Heuristic; disable with
       ``OPENKERNEL_SKIP_KERNEL_IDENTITY_CHECK=1`` if noisy.
    3. Kernel hash already written against a DIFFERENT ``problem_id``
       on the same date — catches leaks that slip past the heuristic.
    """
    kernel_source = result["kernel_source"]
    correct = bool(result["correct"])

    if correct and kernel_source == _FAILURE_PLACEHOLDER:
        raise ValueError(
            "kernel_source is the failure placeholder but correct=True; "
            "placeholder is only valid for correct=False failure records"
        )

    if os.environ.get("OPENKERNEL_SKIP_KERNEL_IDENTITY_CHECK") != "1":
        head = kernel_source[:2000].lower()
        ref_lower = (spec.reference_source or "").lower()
        id_name = f"{spec.id} {spec.name}".lower()
        for token_name, pattern in _OP_TOKENS:
            if not re.search(pattern, head):
                continue
            if re.search(pattern, ref_lower):
                continue
            if re.search(pattern, id_name):
                continue
            logger.warning(
                "leaderboard_writer: kernel-identity mismatch for %s — "
                "kernel docstring mentions %r which is absent from "
                "reference_source and problem id/name; refusing to write. "
                "Set OPENKERNEL_SKIP_KERNEL_IDENTITY_CHECK=1 to bypass.",
                spec.id, token_name,
            )
            raise ValueError(
                f"kernel-identity mismatch: kernel_source for problem "
                f"{spec.id!r} mentions {token_name!r} in its docstring, "
                f"but that token is absent from reference_source and "
                f"from the problem id/name — likely cross-problem leak"
            )

    # Guard #3 (cross-problem hash reuse) does not apply to the failure
    # placeholder. The placeholder is intentionally deterministic so every
    # failure record across every problem collapses to one companion file
    # (``kernels/27c7dcf3ca247272.py``); collision is contract, not leak.
    # Skipping this guard here is safe because guard #1 above already
    # blocks the only dangerous placeholder case (correct=True).
    if kernel_source == _FAILURE_PLACEHOLDER:
        return

    if date_dir.exists():
        for sibling in date_dir.iterdir():
            if sibling.suffix != ".json" or ".tmp." in sibling.name:
                continue
            try:
                with open(sibling, "r") as f:
                    prior = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            if prior.get("kernel_hash") != khash:
                continue
            prior_pid = prior.get("problem_id")
            if prior_pid and prior_pid != spec.id:
                logger.warning(
                    "leaderboard_writer: kernel hash %s already written "
                    "for problem %r on %s; refusing to reuse for %r.",
                    khash, prior_pid, date_dir.name, spec.id,
                )
                raise ValueError(
                    f"kernel-identity mismatch: kernel_hash {khash!r} "
                    f"was previously written against problem "
                    f"{prior_pid!r} on {date_dir.name} — refusing to "
                    f"reuse for {spec.id!r}"
                )


def _atomic_write(target: Path, data: bytes) -> None:
    """Write ``data`` to ``target`` atomically via temp + rename.

    POSIX ``rename`` is atomic within a filesystem, so readers either see
    the pre-rename state or the fully-written file — never a partial one.
    """
    tmp = target.with_name(f"{target.name}.tmp.{os.getpid()}")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.rename(tmp, target)


def _validate(spec: ProblemSpec, result: dict) -> None:
    if not isinstance(spec, ProblemSpec):
        raise ValueError(f"spec must be a ProblemSpec, got {type(spec).__name__}")

    for key, expected_type in _REQUIRED_RESULT_KEYS.items():
        if key not in result:
            raise ValueError(f"missing required field: {key!r}")
        value = result[key]
        # bool is a subclass of int — reject bools where a number is expected
        # and reject ints where strictly bool is expected.
        if expected_type is bool and not isinstance(value, bool):
            raise ValueError(f"field {key!r} must be bool, got {type(value).__name__}")
        if expected_type is not bool and isinstance(value, bool) and expected_type != bool:
            raise ValueError(f"field {key!r} has wrong type bool")
        if not isinstance(value, expected_type):
            raise ValueError(
                f"field {key!r} must be {expected_type}, got {type(value).__name__}"
            )

    if not result["kernel_source"]:
        raise ValueError("kernel_source cannot be empty")
    if spec.tier not in _VALID_TIERS:
        raise ValueError(f"invalid tier {spec.tier!r}")
    if result["hardware"] not in _VALID_HARDWARE:
        raise ValueError(
            f"invalid hardware {result['hardware']!r}, must be one of {sorted(_VALID_HARDWARE)}"
        )
    if result["bottleneck_type"] not in _VALID_BOTTLENECKS:
        raise ValueError(
            f"invalid bottleneck_type {result['bottleneck_type']!r}, "
            f"must be one of {sorted(_VALID_BOTTLENECKS)}"
        )
    # speedup >= 0: 0 is the sentinel for "no correct kernel produced",
    # only valid when correct=False. Correct kernels must have speedup > 0.
    if result["speedup"] < 0:
        raise ValueError(f"speedup must be >= 0, got {result['speedup']}")
    if result["correct"] and result["speedup"] <= 0:
        raise ValueError(
            f"speedup must be > 0 when correct=True, got {result['speedup']}"
        )
    if not 0.0 <= result["sol_score"] <= 1.0:
        raise ValueError(f"sol_score must be in [0, 1], got {result['sol_score']}")
    if not 0.0 <= result["compute_util"] <= 100.0:
        raise ValueError(f"compute_util must be in [0, 100], got {result['compute_util']}")
    if not 0.0 <= result["bandwidth_util"] <= 100.0:
        raise ValueError(
            f"bandwidth_util must be in [0, 100], got {result['bandwidth_util']}"
        )
    if result["cost_usd"] < 0:
        raise ValueError(f"cost_usd must be >= 0, got {result['cost_usd']}")
    if result["elapsed_s"] < 0:
        raise ValueError(f"elapsed_s must be >= 0, got {result['elapsed_s']}")
    if result["iterations"] < 0:
        raise ValueError(f"iterations must be >= 0, got {result['iterations']}")
    if result["correct"] and result["rounds"] < 1:
        raise ValueError(f"rounds must be >= 1 when correct=True, got {result['rounds']}")
    if result["rounds"] < 0:
        raise ValueError(f"rounds must be >= 0, got {result['rounds']}")
    if not result["stop_reason"]:
        raise ValueError("stop_reason cannot be empty")
    if len(result["stop_reason"]) > 256:
        raise ValueError("stop_reason must be <= 256 chars")


def write_record(
    spec: ProblemSpec,
    result: dict,
    root: Path | str = Path("results/leaderboard"),
) -> Path:
    """Persist a leaderboard record + companion kernel source atomically.

    Args:
        spec: Problem descriptor — provides ``problem_id``, ``problem_name``,
            ``tier``.
        result: Run outputs. Required keys are listed in the module docstring
            and enforced by schema validation before any bytes hit disk. The
            optional ``config`` dict is hashed to produce ``config_hash``;
            when absent an empty dict is used.
        root: Leaderboard root directory. Override only in tests — production
            code should leave the default so reader and writer agree on
            layout.

    Returns:
        Path to the written record JSON.

    Raises:
        ValueError: if any required field is missing, of the wrong type, or
            outside its documented constraint range. No file is written when
            validation fails.

    Notes:
        Uniqueness key is ``(problem_id, hardware, date, config_hash)``. Two
        writers racing on the same tuple both rename onto the same target
        path — POSIX ``rename`` is atomic, so the last rename wins and no
        partial state is ever visible to readers.
    """
    root = Path(root)

    _validate(spec, result)

    now_utc = datetime.now(timezone.utc)
    date = result.get("date") or now_utc.strftime("%Y-%m-%d")
    timestamp = result.get("timestamp") or now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    config = result.get("config", {}) or {}

    kernel_source = result["kernel_source"]
    khash = _kernel_hash(kernel_source)
    chash = _config_hash(config)

    record = {
        "schema_version": SCHEMA_VERSION,
        "problem_id": spec.id,
        "problem_name": spec.name,
        "tier": spec.tier,
        "hardware": result["hardware"],
        "date": date,
        "timestamp": timestamp,
        "kernel_hash": khash,
        "kernel_source_path": f"kernels/{khash}.py",
        "model": result["model"],
        "speedup": float(result["speedup"]),
        "sol_score": float(result["sol_score"]),
        "compute_util": float(result["compute_util"]),
        "bandwidth_util": float(result["bandwidth_util"]),
        "bottleneck_type": result["bottleneck_type"],
        "correct": bool(result["correct"]),
        "cost_usd": float(result["cost_usd"]),
        "elapsed_s": int(result["elapsed_s"]),
        "stop_reason": result["stop_reason"],
        "config_hash": chash,
        "rounds": int(result["rounds"]),
        "iterations": int(result["iterations"]),
    }

    date_dir = root / date
    kernels_dir = root / "kernels"

    _check_kernel_identity(spec, result, date_dir, khash)

    date_dir.mkdir(parents=True, exist_ok=True)
    kernels_dir.mkdir(parents=True, exist_ok=True)

    kernel_path = kernels_dir / f"{khash}.py"
    # Skip re-writing an identical kernel source — same hash means same bytes.
    if not kernel_path.exists():
        _atomic_write(kernel_path, kernel_source.encode("utf-8"))

    record_name = f"{spec.id}_{result['hardware']}_{chash}.json"
    if record_name.startswith("_"):
        raise ValueError(
            f"refusing to write record with reserved '_' prefix: {record_name!r}"
        )
    record_path = date_dir / record_name
    payload = json.dumps(record, indent=2, sort_keys=True).encode("utf-8")
    _atomic_write(record_path, payload)

    return record_path


__all__ = ["write_record", "SCHEMA_VERSION"]
