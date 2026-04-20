"""Parse optimization goals from natural language.

Extracts structured parameters from KE's natural language input:
  "optimize @my_kernel.py for H100, need 2x speedup, budget $10"
  → GoalSpec(file="my_kernel.py", hardware="H100", target=2.0, budget=10.0)

Fills missing values from settings, validates all params.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


# Valid GPU types we support
VALID_GPUS = {"L40S", "H100", "A100-80GB", "A100-40GB"}
VALID_BACKENDS = {"triton", "cuda"}


@dataclass
class ParsedGoal:
    """Extracted goal parameters from natural language."""

    file: str = ""
    hardware: str = ""
    backend: str = ""
    target_speedup: float = 0.0
    budget_usd: float = 0.0
    time_limit_seconds: int = 0
    model: str = ""

    # What was explicitly stated vs inferred
    explicit: set = field(default_factory=set)


def parse_goal(text: str) -> ParsedGoal:
    """Extract optimization parameters from natural language."""
    goal = ParsedGoal()

    # --- File: look for @file.py or quoted paths ---
    file_match = re.search(r"@([\w./_-]+\.py)", text)
    if file_match:
        goal.file = file_match.group(1)
        goal.explicit.add("file")

    # Also check for "--- path ---" injected content
    injected = re.search(r"--- ([\w./_-]+\.py) ---", text)
    if injected and not goal.file:
        goal.file = injected.group(1)
        goal.explicit.add("file")

    # --- Hardware: look for GPU names ---
    for gpu in VALID_GPUS:
        if gpu.lower() in text.lower():
            goal.hardware = gpu
            goal.explicit.add("hardware")
            break
    # Common aliases
    if "h100" in text.lower():
        goal.hardware = "H100"
        goal.explicit.add("hardware")
    elif "a100" in text.lower():
        goal.hardware = "A100-80GB"
        goal.explicit.add("hardware")
    elif "l40" in text.lower():
        goal.hardware = "L40S"
        goal.explicit.add("hardware")

    # --- Backend: only match explicit "backend cuda" or "use triton", not device="cuda" ---
    if re.search(r"\b(?:backend|using|use|with)\s+cuda\b", text.lower()):
        goal.backend = "cuda"
        goal.explicit.add("backend")
    elif re.search(r"\b(?:backend|using|use|with)\s+triton\b", text.lower()):
        goal.backend = "triton"
        goal.explicit.add("backend")

    # --- Target speedup: "2x", "2.0x", "2x speedup", "need 2x" ---
    speed_match = re.search(r"(\d+\.?\d*)\s*x\s*(?:speedup|faster|improvement)?", text.lower())
    if speed_match:
        goal.target_speedup = float(speed_match.group(1))
        goal.explicit.add("target")

    # --- Budget: "$10", "$5.00", "budget $10", "budget 10" ---
    budget_match = re.search(r"\$\s*(\d+\.?\d*)", text)
    if budget_match:
        goal.budget_usd = float(budget_match.group(1))
        goal.explicit.add("budget")
    else:
        budget_match2 = re.search(r"budget\s+(\d+\.?\d*)", text.lower())
        if budget_match2:
            goal.budget_usd = float(budget_match2.group(1))
            goal.explicit.add("budget")

    # --- Time limit: "30m", "1h", "30 minutes" ---
    time_match = re.search(r"(\d+)\s*(?:m(?:in(?:ute)?s?)?)\b", text.lower())
    if time_match:
        goal.time_limit_seconds = int(time_match.group(1)) * 60
        goal.explicit.add("time")
    time_match_h = re.search(r"(\d+)\s*(?:h(?:ours?)?)\b", text.lower())
    if time_match_h:
        goal.time_limit_seconds = int(time_match_h.group(1)) * 3600
        goal.explicit.add("time")

    # --- Model ---
    for model_name in ["gpt-4o", "gpt-4o-mini", "claude", "o3-mini"]:
        if model_name in text.lower():
            goal.model = model_name
            goal.explicit.add("model")
            break

    return goal


def validate_goal(goal: ParsedGoal, project_root: Path) -> list[str]:
    """Validate parsed goal. Returns list of errors (empty = valid)."""
    errors = []

    if goal.file:
        path = project_root / goal.file
        if not path.is_file():
            errors.append(f"File not found: {goal.file}")

    if goal.hardware and goal.hardware not in VALID_GPUS:
        errors.append(f"Unknown GPU: {goal.hardware}. Valid: {', '.join(VALID_GPUS)}")

    if goal.backend and goal.backend not in VALID_BACKENDS:
        errors.append(f"Unknown backend: {goal.backend}. Valid: triton, cuda")

    if goal.target_speedup < 0:
        errors.append(f"Invalid target: {goal.target_speedup}x")

    if goal.budget_usd < 0:
        errors.append(f"Invalid budget: ${goal.budget_usd}")

    return errors
