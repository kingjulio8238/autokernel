"""Run logger — captures optimization runs to timestamped log files.

Every /optimize and /autopilot run is logged to .kernel-code/runs/ with:
- Full iteration history (speedups, statuses, intents, errors)
- Configuration used (model, hardware, backend, settings)
- Timing and cost data
- Stop reason
- Best kernel code (if any)

Logs are plain text + JSON for easy review and iteration.

Usage::

    from kernel_code.run_log import RunLogger

    logger = RunLogger()
    logger.start_run(command="/optimize", config={...})
    logger.log_iteration(1, speedup=0.85, status="discard", intent="tiled matmul")
    logger.log_event("Stopping: converged")
    logger.end_run(best_speedup=1.85, best_kernel="...")
    print(logger.log_path)  # .kernel-code/runs/2026-04-19_08-25-16_optimize.log
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_RUNS_DIR = _PROJECT_ROOT / ".kernel-code" / "runs"


class RunLogger:
    """Captures a full optimization run to a log file."""

    def __init__(self) -> None:
        _RUNS_DIR.mkdir(parents=True, exist_ok=True)
        self._log_path: Path | None = None
        self._start_time: float = 0.0
        self._entries: list[str] = []
        self._iterations: list[dict] = []
        self._config: dict = {}

    @property
    def log_path(self) -> Path | None:
        return self._log_path

    def start_run(self, command: str, config: dict) -> Path:
        """Start logging a new run. Returns the log file path."""
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        cmd_name = command.lstrip("/").split()[0] if command else "run"
        self._log_path = _RUNS_DIR / f"{ts}_{cmd_name}.log"
        self._start_time = time.time()
        self._config = {**config, "command": command}
        self._iterations = []
        self._entries = []

        self._write_header(command, config)
        return self._log_path

    def log_iteration(
        self,
        num: int,
        speedup: float,
        status: str,
        intent: str,
        error: str = "",
        profile: dict | None = None,
    ) -> None:
        """Log an iteration result."""
        profile_entry: dict = {}
        if profile:
            _PROFILE_KEYS = (
                "sol_score",
                "compute_util",
                "bandwidth_util",
                "bottleneck_type",
                "total_flops",
                "total_bytes",
                "operational_intensity",
                "cache_efficiency",
                "occupancy",
            )
            for key in _PROFILE_KEYS:
                if key in profile:
                    profile_entry[key] = profile[key]

        entry = {
            "iteration": num,
            "speedup": speedup,
            "status": status,
            "intent": intent,
            "error": error[:200] if error else "",
            "elapsed": round(time.time() - self._start_time, 1),
            "profile": profile_entry,
        }
        self._iterations.append(entry)

        status_sym = {"keep": "✓", "discard": "✗", "error": "!", "compile_error": "!"}.get(status, "?")
        sol_score = profile_entry.get("sol_score") if profile_entry else None
        if sol_score is not None:
            line = (
                f"  [{entry['elapsed']:>6.1f}s] #{num:>2} SOL {float(sol_score):.2f} "
                f"({speedup:.2f}x) {status_sym} {status:<14} {intent[:50]}"
            )
        else:
            line = f"  [{entry['elapsed']:>6.1f}s] #{num:>2} {speedup:>6.2f}x {status_sym} {status:<14} {intent[:50]}"
        if error:
            line += f"\n           ERROR: {error[:100]}"
        self._log(line)

    def log_event(self, message: str) -> None:
        """Log a freeform event (stop reason, pivot, etc.)."""
        elapsed = round(time.time() - self._start_time, 1)
        self._log(f"  [{elapsed:>6.1f}s] {message}")

    def log_round(self, round_num: int, strategy: str) -> None:
        """Log the start of an autopilot round."""
        elapsed = round(time.time() - self._start_time, 1)
        self._log(f"\n  [{elapsed:>6.1f}s] ── Round {round_num}: {strategy} ──")

    def end_run(
        self,
        best_speedup: float = 0.0,
        best_kernel: str = "",
        stop_reason: str = "",
        total_cost: float = 0.0,
        best_sol: float = 0.0,
    ) -> None:
        """Finalize the run log with results.

        Phase 2 dual-display: when ``best_sol`` is non-zero (or recoverable
        from any iteration's profile), SOL is printed as the primary headline
        in the RESULT block and speedup drops to a secondary line. Speedup
        falls back to the sole headline (with an explicit "SOL unknown" tag)
        when no iteration recorded a SOL score. The JSON schema is additive —
        ``best_speedup`` stays for back-compat; ``best_sol`` is new.
        """
        elapsed = time.time() - self._start_time

        if not best_sol:
            for it in self._iterations:
                prof = it.get("profile") or {}
                s = float(prof.get("sol_score", 0.0) or 0.0) if isinstance(prof, dict) else 0.0
                if s > best_sol:
                    best_sol = s

        self._log(f"\n{'='*60}")
        self._log(f"  RESULT")
        if best_sol > 0:
            self._log(f"  Best SOL:      {best_sol:.2f}   ({int(best_sol * 100)}% of hardware peak)")
            self._log(f"  Best speedup:  {best_speedup:.2f}x")
        else:
            self._log(f"  Best speedup:  {best_speedup:.2f}x   (SOL unknown)")
        kept = sum(1 for it in self._iterations if it["status"] == "keep")
        total = len(self._iterations)
        self._log(f"  Iterations:    {kept}/{total} kept")
        self._log(f"  LLM Cost:      ${total_cost:.2f}")
        self._log(f"  Elapsed:       {int(elapsed)}s")
        if stop_reason:
            self._log(f"  Stop reason:   {stop_reason}")

        # Write JSON summary
        self._log(f"\n{'='*60}")
        self._log("  JSON SUMMARY")
        summary = {
            "command": self._config.get("command", ""),
            "timestamp": datetime.now().isoformat(),
            "config": self._config,
            "iterations": self._iterations,
            "best_speedup": best_speedup,
            "best_sol": best_sol,
            "total_cost": total_cost,
            "cost_note": "LLM tokens only; GPU compute billed separately via Modal",
            "elapsed_seconds": round(elapsed, 1),
            "stop_reason": stop_reason,
        }
        self._log(json.dumps(summary, indent=2))

        # Write best kernel if any
        if best_kernel:
            self._log(f"\n{'='*60}")
            self._log("  BEST KERNEL")
            self._log(best_kernel)

        # Flush to disk
        self._flush()

    def _write_header(self, command: str, config: dict) -> None:
        """Write the log file header."""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._log(f"{'='*60}")
        self._log(f"  openkernel run log")
        self._log(f"  {ts}")
        self._log(f"  Command: {command}")
        self._log(f"{'='*60}")
        self._log(f"  Model:     {config.get('model', '?')}")
        self._log(f"  Hardware:  {config.get('hardware', '?')}")
        self._log(f"  Backend:   {config.get('backend', '?')}")
        self._log(f"  Reference: {config.get('reference') or config.get('file', '?')}")
        if config.get("target"):
            self._log(f"  Target:    {config['target']}x")
        if config.get("budget"):
            self._log(f"  Budget:    ${config['budget']:.2f}")
        self._log(f"{'='*60}\n")

    def _log(self, line: str) -> None:
        """Append a line to the log."""
        self._entries.append(line)

    def _flush(self) -> None:
        """Write all entries to disk."""
        if self._log_path:
            self._log_path.write_text("\n".join(self._entries) + "\n")
