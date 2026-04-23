"""Abstract backend interface.

Each backend (Triton, CUDA) knows how to:
- Assemble a generator prompt from context variables
- Validate generated kernel code (syntactic sanity checks)
- Declare its file extension
"""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path

logger = logging.getLogger(__name__)


class BackendBase(ABC):
    """Abstract base class for kernel generation backends."""

    @abstractmethod
    def get_generator_prompt(
        self,
        reference: str,
        hardware: str,
        intent: str,
        critic_feedback: str | None = None,
        skills: str | None = None,
        problem_context: str | None = None,
        strategy_hints: list[str] | None = None,
        archspec: dict | None = None,
        op_template: str | None = None,
        profile: dict | None = None,
    ) -> str:
        """Build the full generator prompt for the LLM.

        Parameters
        ----------
        reference : str
            The PyTorch reference implementation source code.
        hardware : str
            Target GPU description (e.g. "NVIDIA H100 80GB").
        intent : str
            The optimization intent from the world model.
        critic_feedback : str, optional
            Formatted critic diagnosis from a previous iteration.
        skills : str, optional
            Relevant skills retrieved from the skill library.
        problem_context : str, optional
            Formatted classifier output (tier, op type, bottleneck, etc.).
        strategy_hints : list[str], optional
            Actionable hints derived from the classifier.
        archspec : dict, optional
            Structured hardware archspec (device, sm_count, memory_gb,
            bandwidth_gbps, peak_tflops_fp16/fp32).
        op_template : str, optional
            Canonical op-type skeleton body (markdown) selected by the
            classifier's ``op_type``. Empty/None falls back to a neutral
            placeholder in the rendered prompt.

        Returns
        -------
        str
            The assembled prompt ready to send to the LLM.
        """

    @abstractmethod
    def validate_kernel(self, code: str) -> tuple[bool, str]:
        """Run lightweight validation on generated kernel code.

        Returns
        -------
        tuple[bool, str]
            ``(is_valid, message)`` — if invalid, *message* describes the issue.
        """

    @abstractmethod
    def get_file_extension(self) -> str:
        """Return the file extension for kernels of this backend (e.g. '.py')."""


# ---------------------------------------------------------------------------
# Shared formatting helpers (used by Triton and CUDA backends)
# ---------------------------------------------------------------------------


def safe_format(template: str, **kwargs: str) -> str:
    """Format a template string, leaving unknown ``{placeholders}`` intact.

    This avoids ``KeyError`` when prompt templates contain documentation
    placeholders (e.g. refinement-section ``{speedup}`` / ``{bottleneck_type}``)
    that are not meant to be filled by the backend.
    """

    def _replacer(match: re.Match) -> str:
        key = match.group(1)
        if key in kwargs:
            return kwargs[key]
        return match.group(0)

    return re.sub(r"\{(\w+)\}", _replacer, template)


def format_hints(hints: list[str] | None) -> str:
    """Format strategy hints as a compact bullet list (or empty string)."""
    if not hints:
        return ""
    return "\n".join(f"- {h}" for h in hints)


def profile_placeholders(profile: dict | None) -> dict[str, str]:
    """Extract the 4 metric placeholders from a profile dict.

    Accepts both fractional (0-1) and percentage (0-100) ``*_utilization`` /
    ``cache_efficiency`` values; values <= 1.0 are treated as fractions and
    scaled to percent. Missing keys default to 0.0 / ``"unknown"`` so callers
    never crash on a partial profile dict.
    """
    p = profile or {}

    def _pct(key: str) -> str:
        v = p.get(key, 0.0)
        try:
            f = float(v)
        except (TypeError, ValueError):
            f = 0.0
        if f <= 1.0:
            f *= 100.0
        return f"{f:.1f}"

    return {
        "bandwidth_utilization": _pct("bandwidth_utilization"),
        "compute_utilization": _pct("compute_utilization"),
        "cache_efficiency": _pct("cache_efficiency"),
        "bottleneck_type": str(p.get("bottleneck_type") or "unknown"),
    }


def format_archspec(archspec: dict | None) -> str:
    """Format an archspec dict as compact JSON suitable for prompt injection."""
    if not archspec:
        return ""
    try:
        return json.dumps(archspec, separators=(", ", ": "), sort_keys=True)
    except (TypeError, ValueError):
        return ""


# ---------------------------------------------------------------------------
# Hardware archspec resolver
# ---------------------------------------------------------------------------

_ARCHSPEC_FALLBACK: dict[str, dict] = {
    # Minimal structured specs. Kept in sync with data/skills/hw_*.json prose.
    "l40s": {
        "device": "NVIDIA L40S",
        "arch": "Ada Lovelace",
        "compute_cap": "sm_89",
        "sm_count": 142,
        "smem_per_sm_kb": 100,
        "memory_gb": 48,
        "bandwidth_gbps": 864,
        "peak_tflops_fp16": 183,
    },
    "h100": {
        "device": "NVIDIA H100",
        "arch": "Hopper",
        "compute_cap": "sm_90",
        "sm_count": 132,
        "smem_per_sm_kb": 192,
        "memory_gb": 80,
        "bandwidth_gbps": 3350,
        "peak_tflops_fp16": 989,
    },
    "a100": {
        "device": "NVIDIA A100",
        "arch": "Ampere",
        "compute_cap": "sm_80",
        "sm_count": 108,
        "smem_per_sm_kb": 164,
        "memory_gb": 80,
        "bandwidth_gbps": 2039,
        "peak_tflops_fp16": 312,
    },
    "b200": {
        "device": "NVIDIA B200",
        "arch": "Blackwell",
        "compute_cap": "sm_100",
        "sm_count": 148,
        "smem_per_sm_kb": 228,
        "memory_gb": 192,
        "bandwidth_gbps": 8000,
        "peak_tflops_fp16": 2250,
        "peak_tflops_fp8": 4500,
    },
}


_SKILLS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "skills"
_TEMPLATE_DIR = Path(__file__).resolve().parents[2] / "data" / "prompts" / "op_templates"


def load_op_template(op_type: str | None) -> str:
    """Load the op-type template markdown for the given OpType enum value.

    Returns the template body as a string, or empty string when the
    op_type is None, unknown, or the template file doesn't exist. Never
    raises — missing templates are soft-fallback.
    """
    if not op_type:
        return ""
    slug = str(op_type).strip().lower()
    if not slug:
        return ""
    path = _TEMPLATE_DIR / f"{slug}.md"
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("load_op_template: failed to read %s — %s", path, exc)
        return ""

# Single-line numeric patterns we try to extract from the prose "approach" field.
_APPROACH_PATTERNS: list[tuple[str, str, type]] = [
    # (key, regex, caster)
    ("sm_count", r"(\d+)\s*SMs?\b", int),
    ("smem_per_sm_kb", r"(\d+)\s*KB\s*(?:of\s*)?(?:shared memory|SMEM)\s*per\s*SM", int),
    ("bandwidth_gbps", r"(\d+(?:\.\d+)?)\s*(?:GB|TB)/s", float),
    ("peak_tflops_fp16", r"(\d+(?:\.\d+)?)\s*TFLOPS\s*(?:FP16|BF16)", float),
    ("peak_tflops_fp8", r"(\d+(?:\.\d+)?)\s*TFLOPS\s*FP8", float),
    ("peak_tflops_fp32", r"(\d+(?:\.\d+)?)\s*TFLOPS\s*FP32", float),
]


def _hardware_archspec(hardware: str) -> dict:
    """Return a structured archspec dict for the given hardware label.

    Tries, in order:
    1. ``data/skills/hw_{hardware_lower}_optimization.json`` — extract numeric
       specs from the free-text ``approach`` field.
    2. Hard-coded fallback table (L40S / H100 / A100 / B200).
    3. Empty dict with just ``device`` populated.

    Any failure is logged and degrades to the next option — this helper
    never raises.
    """
    key = (hardware or "").strip().lower()
    # Strip common vendor prefixes / suffixes so "NVIDIA H100 80GB" -> "h100"
    for token in ("l40s", "h100", "a100", "b200"):
        if token in key:
            key = token
            break

    # 1. Try JSON
    spec: dict = {}
    json_path = _SKILLS_DIR / f"hw_{key}_optimization.json"
    if json_path.exists():
        try:
            with open(json_path) as f:
                data = json.load(f)
            trigger = data.get("trigger", "")
            for piece in trigger.split(","):
                piece = piece.strip()
                if piece.startswith("sm_") or piece.startswith("compute_"):
                    spec["compute_cap"] = piece
                elif piece in ("Ada Lovelace", "Ampere", "Hopper", "Blackwell"):
                    spec["arch"] = piece
            approach = data.get("approach", "")
            for field, pattern, caster in _APPROACH_PATTERNS:
                m = re.search(pattern, approach, re.IGNORECASE)
                if m:
                    try:
                        value = caster(m.group(1))
                        # Normalize TB/s -> GB/s for bandwidth
                        if field == "bandwidth_gbps" and "tb/s" in m.group(0).lower():
                            value *= 1000
                        spec[field] = value
                    except (ValueError, IndexError):
                        continue
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("archspec: failed to parse %s — %s", json_path, exc)

    # 2. Merge fallback (fills in anything the JSON did not provide)
    fallback = _ARCHSPEC_FALLBACK.get(key, {})
    for fkey, fval in fallback.items():
        spec.setdefault(fkey, fval)

    # 3. Always at least identify the device
    spec.setdefault("device", hardware or "unknown")

    return spec
