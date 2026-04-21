"""Post-run analysis for kernel optimization.

Reads run logs + dev logs, analyzes LLM behavior, identifies issues
(wrong classification, bad strategies, missed optimizations), and
produces actionable recommendations for harness improvement.

Usage::

    from kernel_code.run_analysis import analyze_run

    report = analyze_run(
        run_log_path=".kernel-code/runs/2026-04-21_10-05-15_smart.log",
        dev_log_path=".kernel-code/dev_logs/run_2026-04-21_10-05-15.jsonl",
    )
    console.print(report)
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from rich.console import Group
from rich.table import Table
from rich.text import Text

_CLAY = "#d77757"
_SUCCESS = "#4eba65"
_DIM = "#999999"
_CYAN = "#22d3ee"
_GOLD = "#f5a850"
_RED = "#ff6b80"


def analyze_run(
    run_log_path: str | Path,
    dev_log_path: str | Path | None = None,
) -> Group:
    """Analyze a completed optimization run and produce a diagnostic report.

    Args:
        run_log_path: Path to the .log file from .kernel-code/runs/
        dev_log_path: Optional path to the .jsonl dev log. Auto-detected if not provided.

    Returns:
        Rich Group ready for console.print()
    """
    run_log_path = Path(run_log_path)
    if not run_log_path.exists():
        return Group(Text(f"  [red]Log file not found: {run_log_path}[/red]"))

    # Parse run log
    run_data = _parse_run_log(run_log_path)

    # Auto-detect dev log if not provided
    if dev_log_path is None:
        # Extract timestamp from run log name: 2026-04-21_10-05-15_smart.log → 2026-04-21_10-05-15
        stem = run_log_path.stem
        # Remove suffix like _smart, _native
        ts_match = re.match(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})", stem)
        if ts_match:
            ts = ts_match.group(1)
            candidate = Path(".kernel-code/dev_logs") / f"run_{ts}.jsonl"
            if candidate.exists():
                dev_log_path = candidate

    # Parse dev log
    dev_entries = []
    if dev_log_path and Path(dev_log_path).exists():
        dev_entries = _parse_dev_log(Path(dev_log_path))

    # Run all analysis checks
    issues = []
    recommendations = []

    _check_kernel_quality(run_data, issues, recommendations)
    _check_classification(run_data, issues, recommendations)
    _check_strategies(run_data, dev_entries, issues, recommendations)
    _check_reflection_quality(dev_entries, issues, recommendations)
    _check_model_behavior(dev_entries, issues, recommendations)
    _check_cost_efficiency(run_data, issues, recommendations)

    # Build report
    return _build_report(run_data, dev_entries, issues, recommendations)


def _parse_run_log(path: Path) -> dict:
    """Parse a run log file into structured data."""
    text = path.read_text()
    data: dict = {"raw": text, "path": str(path)}

    # Extract JSON summary
    json_match = re.search(r"JSON SUMMARY\n(\{.*?\})", text, re.DOTALL)
    if json_match:
        try:
            data["summary"] = json.loads(json_match.group(1))
        except json.JSONDecodeError:
            data["summary"] = {}

    # Extract best kernel
    kernel_match = re.search(r"BEST KERNEL\n(.*?)(?:={10,}|$)", text, re.DOTALL)
    if kernel_match:
        data["best_kernel"] = kernel_match.group(1).strip()

    # Extract round timestamps
    rounds = re.findall(r"\[\s*([\d.]+)s\] ── Round (\d+): (.+?) ──", text)
    data["rounds"] = [
        {"elapsed_s": float(t), "round": int(r), "strategy": s.strip()}
        for t, r, s in rounds
    ]

    # Extract result
    speedup_match = re.search(r"Best speedup:\s+([\d.]+)x", text)
    data["best_speedup"] = float(speedup_match.group(1)) if speedup_match else 0.0

    cost_match = re.search(r"Cost:\s+\$([\d.]+)", text)
    data["cost"] = float(cost_match.group(1)) if cost_match else 0.0

    stop_match = re.search(r"Stop reason:\s+(.+)", text)
    data["stop_reason"] = stop_match.group(1).strip() if stop_match else ""

    return data


def _parse_dev_log(path: Path) -> list[dict]:
    """Parse a JSONL dev log into a list of entries."""
    entries = []
    for line in path.read_text().strip().split("\n"):
        if line.strip():
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def _check_kernel_quality(run_data: dict, issues: list, recommendations: list):
    """Analyze the best kernel for quality issues."""
    kernel = run_data.get("best_kernel", "")
    if not kernel:
        issues.append(("CRITICAL", "No best kernel found in log"))
        return

    # Check for autotune
    if "@triton.autotune" not in kernel and "triton.autotune" not in kernel:
        issues.append(("HIGH", "Kernel has no @triton.autotune — uses hardcoded block size"))
        recommendations.append("Add @triton.autotune with multiple BLOCK_SIZE configs (256, 512, 1024, 2048)")

    # Check for hardcoded dtypes
    if "torch.float16" in kernel and "dtype" not in kernel.split("torch.float16")[0][-50:]:
        issues.append(("MEDIUM", "Kernel hardcodes torch.float16 instead of using input tensor dtype"))

    # Check for excessive validation
    validation_count = kernel.count("raise ValueError") + kernel.count("raise RuntimeError")
    if validation_count >= 3:
        issues.append(("MEDIUM", f"Kernel has {validation_count} validation checks — adds overhead to hot path"))
        recommendations.append("Move validation to test code, not kernel function")

    # Check for __main__ block
    if "__main__" in kernel:
        issues.append(("LOW", "Kernel includes if __name__ == '__main__' test block (dead code)"))

    # Check for excessive comments
    comment_lines = sum(1 for line in kernel.split("\n") if line.strip().startswith("#") or line.strip().startswith('"""'))
    total_lines = len(kernel.split("\n"))
    if total_lines > 0 and comment_lines / total_lines > 0.4:
        issues.append(("LOW", f"Kernel is {int(comment_lines/total_lines*100)}% comments — model over-documenting instead of optimizing"))

    # Check kernel size
    if total_lines > 100:
        issues.append(("LOW", f"Kernel is {total_lines} lines — model generating verbose wrapper code"))

    # Check for vectorized loads
    if "eviction_policy" not in kernel and "num_stages" not in kernel:
        recommendations.append("Consider vectorized loads with eviction_policy='evict_last' for memory-bound kernels")


def _check_classification(run_data: dict, issues: list, recommendations: list):
    """Check if problem classification was correct."""
    kernel = run_data.get("best_kernel", "")
    summary = run_data.get("summary", {})

    # Detect actual operation type from kernel
    if kernel:
        has_matmul = any(p in kernel.lower() for p in ["tl.dot", "matmul", "gemm"])
        has_reduction = any(p in kernel.lower() for p in ["tl.sum", "tl.max", "tl.min"])
        has_elementwise = "tl.load" in kernel and "tl.store" in kernel and not has_matmul and not has_reduction

        if has_elementwise:
            # Check if strategies were appropriate for elementwise
            rounds = run_data.get("rounds", [])
            bad_strategies = [r for r in rounds if any(
                s in r["strategy"].lower() for s in
                ["shared memory tiling", "tmem", "tensor core", "pipeline fusion", "parallelism axis"]
            )]
            if bad_strategies:
                issues.append(("HIGH",
                    f"{len(bad_strategies)} rounds used complex strategies "
                    f"(tiling, TMEM, pipeline fusion) on a simple elementwise kernel"
                ))
                recommendations.append(
                    "Elementwise kernels are memory-bound — strategies should focus on "
                    "vectorized loads, coalesced access, and grid-stride loops, not compute optimizations"
                )


def _check_strategies(run_data: dict, dev_entries: list, issues: list, recommendations: list):
    """Analyze strategy effectiveness."""
    rounds = run_data.get("rounds", [])
    if len(rounds) < 2:
        return

    # Check for strategy repetition (same strategy, different words)
    strategies = [r["strategy"].lower() for r in rounds]
    unique_concepts = set()
    for s in strategies:
        # Extract core concept
        for concept in ["shared memory", "tiling", "parallelism", "pipeline", "hardware", "buffer", "fusion", "autotune", "block size"]:
            if concept in s:
                unique_concepts.add(concept)

    if len(unique_concepts) < len(rounds) * 0.5 and len(rounds) >= 3:
        issues.append(("MEDIUM",
            f"Strategy diversity is low: {len(unique_concepts)} unique concepts across {len(rounds)} rounds — "
            f"reflection may be generating semantically similar pivots"
        ))

    # Check for strategies inappropriate to problem type
    speedup = run_data.get("best_speedup", 0.0)
    if speedup < 1.0 and len(rounds) >= 3:
        issues.append(("HIGH",
            f"After {len(rounds)} rounds, best speedup is {speedup:.2f}x (slower than baseline) — "
            f"problem may be near-optimal for PyTorch or strategies are misaligned"
        ))
        recommendations.append("Consider early stopping when speedup stays below 1.0x for 3+ rounds")


def _check_reflection_quality(dev_entries: list, issues: list, recommendations: list):
    """Analyze LLM reflection quality."""
    reflections = [e for e in dev_entries if e.get("type") == "reflection"]
    if not reflections:
        issues.append(("MEDIUM", "No reflection logs found — was OPENKERNEL_DEV_LOG=1 set?"))
        return

    # Check if reflection actually changed strategy
    responses = [r.get("response", "") for r in reflections]
    continues = sum(1 for r in responses if r.upper().startswith("CONTINUE"))
    pivots = sum(1 for r in responses if r.upper().startswith("PIVOT"))
    stops = sum(1 for r in responses if r.upper().startswith("STOP"))

    if continues > 0 and pivots == 0 and len(reflections) >= 3:
        issues.append(("HIGH",
            f"Reflection always chose CONTINUE ({continues} times, never PIVOT) — "
            f"may be too conservative about strategy changes"
        ))

    # Check if reflection is hallucinating advanced techniques for simple problems
    for ref in reflections:
        resp = ref.get("response", "").lower()
        best = ref.get("best_speedup", 0)
        if best < 1.0 and any(t in resp for t in ["tmem", "tensor core", "tcgen05", "pipeline stage"]):
            issues.append(("HIGH",
                f"Reflection suggested advanced techniques (TMEM, tensor cores) "
                f"for a kernel that can't even beat baseline ({best:.2f}x)"
            ))
            recommendations.append(
                "Reflection prompt should consider problem tier — "
                "L1 elementwise kernels don't benefit from L3/L4 strategies"
            )
            break


def _check_model_behavior(dev_entries: list, issues: list, recommendations: list):
    """Analyze worker LLM behavior from dev logs."""
    workers = [e for e in dev_entries if e.get("type") == "worker"]
    if not workers:
        return

    # Check success rate
    total = len(workers)
    successes = sum(1 for w in workers if w.get("success"))
    if total > 0:
        rate = successes / total * 100
        if rate < 50:
            issues.append(("HIGH",
                f"Worker success rate is {rate:.0f}% ({successes}/{total}) — "
                f"most LLM-generated kernels fail verification"
            ))

    # Check prompt size
    prompt_sizes = [len(w.get("prompt", "")) for w in workers if w.get("prompt")]
    if prompt_sizes:
        avg_prompt = sum(prompt_sizes) / len(prompt_sizes)
        if avg_prompt > 20000:
            issues.append(("MEDIUM", f"Average prompt size is {int(avg_prompt)} chars — may be hitting context limits"))
        if avg_prompt < 500:
            issues.append(("MEDIUM", f"Average prompt size is {int(avg_prompt)} chars — may be missing context"))

    # Check response quality
    for w in workers:
        resp = w.get("response", "")
        if resp and "```" not in resp:
            issues.append(("LOW", f"Worker round {w.get('round', '?')} response has no code block"))
            break


def _check_cost_efficiency(run_data: dict, issues: list, recommendations: list):
    """Analyze cost efficiency."""
    cost = run_data.get("cost", 0.0)
    speedup = run_data.get("best_speedup", 0.0)
    rounds = len(run_data.get("rounds", []))

    if cost > 0 and speedup > 0 and rounds > 0:
        cost_per_round = cost / rounds
        if speedup < 1.0:
            issues.append(("MEDIUM",
                f"Spent ${cost:.2f} ({rounds} rounds) but kernel is slower than baseline ({speedup:.2f}x)"
            ))
            recommendations.append("Add early stopping: if best_speedup < 1.0x after round 2, stop and report")


def _build_report(
    run_data: dict, dev_entries: list, issues: list, recommendations: list
) -> Group:
    """Build the final Rich report."""
    parts: list = []

    # Header
    parts.append(Text())
    header = Text()
    header.append("  ── Run Analysis ", style=f"bold {_CYAN}")
    header.append("─" * 40, style=_DIM)
    parts.append(header)

    # Summary
    summary = run_data.get("summary", {})
    speedup = run_data.get("best_speedup", 0.0)
    cost = run_data.get("cost", 0.0)
    rounds = run_data.get("rounds", [])
    stop = run_data.get("stop_reason", "")

    info = Text()
    info.append(f"  Speedup: ", style=_DIM)
    sp_color = _SUCCESS if speedup >= 1.0 else _RED
    info.append(f"{speedup:.2f}x", style=f"bold {sp_color}")
    info.append(f"  |  Rounds: {len(rounds)}", style=_DIM)
    info.append(f"  |  Cost: ${cost:.2f}", style=_DIM)
    info.append(f"  |  Model: {summary.get('config', {}).get('model', '?')}", style=_DIM)
    parts.append(info)

    if stop:
        parts.append(Text(f"  Stop: {stop}", style=_DIM))

    # Dev log stats
    workers = [e for e in dev_entries if e.get("type") == "worker"]
    reflections = [e for e in dev_entries if e.get("type") == "reflection"]
    if workers or reflections:
        dev_info = Text()
        dev_info.append(f"  Dev log: ", style=_DIM)
        dev_info.append(f"{len(workers)} worker entries, {len(reflections)} reflections", style=_DIM)
        parts.append(dev_info)

    parts.append(Text())

    # Round-by-round summary
    if rounds:
        round_table = Table(
            title="Round Summary",
            show_header=True,
            header_style=f"bold {_CYAN}",
            border_style=_DIM,
        )
        round_table.add_column("Round", justify="right", width=6)
        round_table.add_column("Time", justify="right", width=8)
        round_table.add_column("Strategy", width=50)

        for r in rounds:
            strategy = r["strategy"]
            if len(strategy) > 48:
                strategy = strategy[:47] + "\u2026"
            round_table.add_row(
                str(r["round"]),
                f"{r['elapsed_s']:.0f}s",
                strategy,
            )
        parts.append(round_table)
        parts.append(Text())

    # Issues table
    if issues:
        issue_table = Table(
            title=f"Issues Found ({len(issues)})",
            show_header=True,
            header_style=f"bold {_GOLD}",
            border_style=_DIM,
        )
        issue_table.add_column("Severity", width=10)
        issue_table.add_column("Issue", ratio=1)

        for severity, desc in sorted(issues, key=lambda x: {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}.get(x[0], 4)):
            color = {
                "CRITICAL": _RED,
                "HIGH": _CLAY,
                "MEDIUM": _GOLD,
                "LOW": _DIM,
            }.get(severity, _DIM)
            issue_table.add_row(
                Text(severity, style=f"bold {color}"),
                desc,
            )
        parts.append(issue_table)
    else:
        parts.append(Text("  No issues found.", style=_SUCCESS))

    parts.append(Text())

    # Recommendations
    if recommendations:
        rec_header = Text()
        rec_header.append(f"  Recommendations ({len(recommendations)})", style=f"bold {_SUCCESS}")
        parts.append(rec_header)
        for i, rec in enumerate(recommendations, 1):
            parts.append(Text(f"  {i}. {rec}", style="white"))

    parts.append(Text())
    return Group(*parts)
