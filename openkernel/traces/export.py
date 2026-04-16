"""Export traces into training-ready formats for kernelgen-1.

Produces DataFrames / Parquet files under traces/processed/.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from openkernel.traces.storage import list_traces, load_trace


def export_training_pairs(
    traces_dir: str | Path = "traces/raw",
    min_speedup: float = 1.0,
) -> pd.DataFrame:
    """Extract (prompt, response, reward) training pairs from successful traces.

    Filters for final_correct=True and final_speedup >= min_speedup.
    From each qualifying trace, extracts every iteration that was kept
    (decision == "keep") as a training pair.

    Returns DataFrame with columns:
        prompt, response, reward, problem_id, backend, model_id
    """
    paths = list_traces(traces_dir)
    rows: list[dict] = []

    for path in paths:
        trace = load_trace(path)

        if not trace.final_correct:
            continue
        if trace.final_speedup < min_speedup:
            continue

        for it in trace.iterations:
            if it.decision != "keep":
                continue
            rows.append(
                {
                    "prompt": it.generator_prompt,
                    "response": it.kernel_code,
                    "reward": it.speedup,
                    "problem_id": trace.problem_id,
                    "backend": trace.backend,
                    "model_id": trace.model_id,
                }
            )

    return pd.DataFrame(rows, columns=["prompt", "response", "reward", "problem_id", "backend", "model_id"])


def export_strategy_rewards(traces_dir: str | Path = "traces/raw") -> pd.DataFrame:
    """Extract which strategies worked and their resulting speedups.

    Returns DataFrame with columns:
        problem_id, strategy, final_speedup
    """
    paths = list_traces(traces_dir)
    rows: list[dict] = []

    for path in paths:
        trace = load_trace(path)
        for strategy in trace.strategies_tried:
            rows.append(
                {
                    "problem_id": trace.problem_id,
                    "strategy": strategy,
                    "final_speedup": trace.final_speedup,
                }
            )

    return pd.DataFrame(rows, columns=["problem_id", "strategy", "final_speedup"])


def save_processed(df: pd.DataFrame, output_path: str | Path) -> Path:
    """Save a processed DataFrame to Parquet under traces/processed/.

    Creates parent directories if needed. Returns the output path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False, engine="pyarrow")
    return output_path
