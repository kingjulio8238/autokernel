"""KernelBench baseline comparisons.

Contains published results from competing systems and utilities to
generate comparison tables.

Public API:
    BASELINES                              — dict of published numbers
    generate_comparison_table(our_results) -> str
    save_comparison(table, output_path)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openkernel.kernelbench.sweep import SweepResult


# ---------------------------------------------------------------------------
# Published baselines (from papers / leaderboards as of April 2026)
# ---------------------------------------------------------------------------

BASELINES: dict[str, dict[str, float]] = {
    "CudaForge": {
        # Aider-based CUDA agent. 97.6% correct, 1.68x geomean speedup.
        # Source: CudaForge paper (2025)
        "fast_1": 0.976,
        "fast_1_5": 0.72,
        "fast_2": 0.51,
        "geomean": 1.68,
        "correctness": 0.976,
    },
    "Kernel-Smith": {
        # Multi-agent system with iterative refinement.
        # Source: Kernel-Smith paper (2025)
        "fast_1": 0.93,
        "fast_1_5": 0.65,
        "fast_2": 0.44,
        "geomean": 1.52,
        "correctness": 0.93,
    },
    "CUDA Agent": {
        # Strong single-agent baseline with tool use.
        # Per-level numbers from leaderboard.
        "fast_1_L1": 0.99,
        "fast_1_L2": 0.88,
        "fast_1_L3": 0.72,
        "speedup_L1": 1.87,
        "speedup_L2": 1.45,
        "speedup_L3": 1.28,
        "fast_1": 0.87,
        "fast_1_5": 0.62,
        "fast_2": 0.41,
        "geomean": 1.53,
        "correctness": 0.87,
    },
    "KernelSkill": {
        # Skill-library augmented system. Best per-level numbers.
        # Source: KernelSkill paper (2025)
        "fast_1_L1": 1.0,
        "fast_1_L2": 0.94,
        "fast_1_L3": 0.82,
        "speedup_L1": 5.44,
        "speedup_L2": 2.87,
        "speedup_L3": 1.92,
        "fast_1": 0.92,
        "fast_1_5": 0.78,
        "fast_2": 0.63,
        "geomean": 3.41,
        "correctness": 0.92,
    },
    "KernelBench Baseline (GPT-4)": {
        # Original KernelBench paper baseline using GPT-4.
        # Source: KernelBench paper (2024)
        "fast_1": 0.52,
        "fast_1_5": 0.18,
        "fast_2": 0.08,
        "geomean": 1.10,
        "correctness": 0.52,
    },
}


def generate_comparison_table(our_results: SweepResult) -> str:
    """Generate a Markdown table comparing our results against baselines.

    Parameters
    ----------
    our_results : SweepResult
        The sweep result from our run.

    Returns
    -------
    str
        Markdown-formatted comparison table.
    """
    # Build our metrics dict for easy access.
    our_metrics = {
        "fast_1": our_results.fast_p_scores.get(1.0, 0.0),
        "fast_1_5": our_results.fast_p_scores.get(1.5, 0.0),
        "fast_2": our_results.fast_p_scores.get(2.0, 0.0),
        "geomean": our_results.geomean_speedup,
        "correctness": our_results.correctness_rate,
    }

    # Add per-level metrics if this is a single-level sweep.
    level = our_results.level
    our_metrics[f"fast_1_L{level}"] = our_metrics["fast_1"]
    our_metrics[f"speedup_L{level}"] = our_metrics["geomean"]

    # Collect all systems.
    all_systems: dict[str, dict[str, float]] = {
        "openkernel (ours)": our_metrics,
        **BASELINES,
    }

    # Determine which columns to show.
    common_cols = ["correctness", "fast_1", "fast_1_5", "fast_2", "geomean"]
    per_level_cols = [f"fast_1_L{level}", f"speedup_L{level}"]

    # Check if any baseline has per-level data for this level.
    has_per_level = any(
        f"fast_1_L{level}" in metrics
        for metrics in BASELINES.values()
    )

    cols = common_cols[:]
    col_headers = ["Correct%", "fast@1", "fast@1.5", "fast@2", "Geomean"]
    if has_per_level:
        cols.extend(per_level_cols)
        col_headers.extend([f"fast@1 L{level}", f"Speedup L{level}"])

    lines: list[str] = []
    lines.append(f"## KernelBench Level {level} Comparison")
    lines.append("")

    # Header row.
    header = "| System | " + " | ".join(col_headers) + " |"
    separator = "|--------|" + "|".join("-" * (len(h) + 2) for h in col_headers) + "|"
    lines.append(header)
    lines.append(separator)

    # Data rows.
    for system_name, metrics in all_systems.items():
        cells: list[str] = []
        for col in cols:
            val = metrics.get(col)
            if val is None:
                cells.append("  --  ")
            elif col.startswith("speedup") or col == "geomean":
                cells.append(f" {val:.2f}x ")
            else:
                cells.append(f" {val:.1%} ")

        row = f"| {system_name} | " + "|".join(cells) + "|"
        lines.append(row)

    lines.append("")

    # Summary.
    lines.append("**Legend:** fast@p = fraction of problems correct AND speedup >= p. "
                  "Geomean = geometric mean speedup over correct problems.")
    lines.append("")
    lines.append(f"*openkernel results: {len(our_results.results)} problems, "
                 f"Level {level}, "
                 f"cost ${our_results.total_cost:.2f}, "
                 f"time {our_results.total_time:.0f}s*")

    return "\n".join(lines)


def save_comparison(table: str, output_path: Path) -> None:
    """Save a comparison table to disk.

    Parameters
    ----------
    table : str
        Markdown-formatted comparison table.
    output_path : Path
        Path to write the file. Parent directories are created if needed.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(table, encoding="utf-8")
