"""Bridge between kernel-code shell and Meta's KernelAgent.

Replaces our native inner loop with KernelAgent's proven pipeline:
- 4 parallel workers at varied temperatures
- 10-round error feedback loop
- Battle-tested Jinja2 prompt templates
- Structured error feedback

Adds our Modal remote eval and progress callbacks on top.

Usage::

    bridge = KernelAgentBridge(
        reference_source=open("reference.py").read(),
        model_name="claude-sonnet-4-20250514",
        hardware="L40S",
        live_display=display,
    )
    result = bridge.run()
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kernel_code.live_display import LiveOptimizationDisplay
    from kernel_code.run_log import RunLogger

logger = logging.getLogger(__name__)


def _modal_eval(kernel_code: str, reference_code: str) -> dict:
    """Evaluate a kernel via Modal remote GPU."""
    import modal

    eval_fn = modal.Function.from_name("openkernel-eval", "eval_kernel_on_gpu")
    result = eval_fn.remote(
        kernel_source=kernel_code,
        reference_source=reference_code,
        eval_mode="fast",
    )
    return result


class KernelAgentBridge:
    """Runs KernelAgent with Modal eval and our UX hooks."""

    def __init__(
        self,
        reference_source: str,
        model_name: str = "gpt-4o",
        num_workers: int = 4,
        max_rounds: int = 8,
        hardware: str = "L40S",
        live_display: "LiveOptimizationDisplay | None" = None,
        run_logger: "RunLogger | None" = None,
        use_modal: bool = True,
    ) -> None:
        self._reference = reference_source
        self._model_name = model_name
        self._num_workers = num_workers
        self._max_rounds = max_rounds
        self._hardware = hardware
        self._live_display = live_display
        self._run_logger = run_logger
        self._use_modal = use_modal

        self._best_speedup: float = 0.0
        self._best_kernel: str = ""
        self._iteration_count: int = 0

    def run(self) -> dict[str, Any]:
        """Run KernelAgent optimization.

        Returns dict with: success, kernel_code, speedup, rounds, worker_id
        """
        from kernel_agent.agent import TritonKernelAgent

        start_time = time.time()

        # Suppress ALL KernelAgent and LLM client logging.
        # Must suppress before AND after agent creation since loggers are
        # created lazily. Also suppress at root level to catch child loggers.
        _suppress_names = [
            "kernel_agent", "TritonKernelAgent", "httpx", "openai",
            "anthropic", "litellm",
            "kernel_agent.ka_utils", "kernel_agent.ka_utils.providers",
            "kernel_agent.ka_utils.providers.openai_base",
            "kernel_agent.manager", "kernel_agent.worker",
            "kernel_agent.agent", "kernel_agent.prompt_manager",
        ]
        for name in _suppress_names:
            lg = logging.getLogger(name)
            lg.setLevel(logging.CRITICAL)
            lg.propagate = False
            lg.handlers = []  # remove any existing handlers

        # Ensure API keys are in env — KernelAgent's providers read from os.environ
        from kernel_code.settings import load_settings, inject_api_keys
        settings = load_settings()
        inject_api_keys(settings)

        # Build problem description from reference
        problem_desc = (
            f"Optimize the following PyTorch code into a fast Triton kernel "
            f"for {self._hardware} GPU.\n\n"
            f"Reference implementation:\n```python\n{self._reference}\n```\n\n"
            f"The kernel must be correct (torch.allclose with rtol=1e-2, atol=1e-2)."
        )

        # Create agent
        agent = TritonKernelAgent(
            num_workers=self._num_workers,
            max_rounds=self._max_rounds,
            model_name=self._model_name,
        )

        # Configure workers with Modal eval and callbacks
        self._configure_agent(agent)

        # Report phase
        if self._live_display:
            self._live_display.update_phase(
                f"KernelAgent: {self._num_workers} workers × {self._max_rounds} rounds "
                f"({self._model_name})"
            )

        # Run KernelAgent in a background thread so the Rich.Live display
        # can keep refreshing (KernelAgent blocks on multiprocessing.join)
        import concurrent.futures

        result = None
        error = None

        def _run_agent():
            nonlocal result, error
            try:
                result = agent.generate_kernel(
                    problem_description=problem_desc,
                )
            except Exception as exc:
                error = exc

        # Track agent log dir for progress polling — only find sessions
        # created AFTER this run started
        agent_log_dir = Path(agent.log_dir) if hasattr(agent, 'log_dir') else None
        _found_session_dir = [None]
        _run_start = time.time()

        def _find_workers_dir():
            """Discover the current session's workers directory (created after run start)."""
            if _found_session_dir[0]:
                wdir = _found_session_dir[0] / "workers"
                if wdir.exists():
                    return wdir
            if agent_log_dir and agent_log_dir.exists():
                # Only look at sessions created after this run started
                for sdir in sorted(agent_log_dir.glob("session_*"), key=lambda p: p.name, reverse=True):
                    if sdir.stat().st_mtime >= _run_start - 2:
                        _found_session_dir[0] = sdir
                        wdir = sdir / "workers"
                        if wdir.exists():
                            return wdir
                        break
            return None

        def _get_worker_action(wdir: Path) -> str:
            """Parse the last log line to get the worker's current action."""
            log_file = None
            for f in wdir.iterdir():
                if f.suffix == ".log":
                    log_file = f
                    break
            if not log_file or not log_file.exists():
                return ""
            try:
                # Read last 500 bytes to find the last log line
                size = log_file.stat().st_size
                with open(log_file, "r") as fh:
                    if size > 500:
                        fh.seek(size - 500)
                    lines = fh.readlines()
                # Find last INFO line with a meaningful action
                for line in reversed(lines):
                    if "INFO" not in line:
                        continue
                    # Extract the message after the log prefix
                    parts = line.split(" - INFO - ", 1)
                    if len(parts) == 2:
                        msg = parts[1].strip()
                        # Shorten common patterns
                        if "Refining kernel" in msg:
                            return "refining"
                        if "Remote eval PASS" in msg:
                            return "eval passed"
                        if "Remote eval FAIL" in msg:
                            return "eval failed"
                        if "Round" in msg:
                            return "evaluating"
                        if "Writing" in msg or "wrote" in msg.lower():
                            return "writing"
                        if "Test" in msg and "passed" in msg.lower():
                            return "test passed"
                        if "Test" in msg and "failed" in msg.lower():
                            return "test failed"
                        return msg[:30]
                return ""
            except Exception:
                return ""

        def _poll_workers():
            """Poll worker directories for round progress and current action."""
            workers_dir = _find_workers_dir()
            if not workers_dir:
                return
            worker_states = []
            for i in range(self._num_workers):
                wdir = workers_dir / f"worker_{i}"
                if wdir.exists():
                    round_files = list(wdir.glob("round_*.json"))
                    rounds = len(round_files)
                    status = "working"
                    action = _get_worker_action(wdir)
                    if rounds > 0:
                        try:
                            latest = max(round_files, key=lambda p: p.name)
                            data = json.loads(latest.read_text())
                            if data.get("success"):
                                status = "passed"
                                action = "correct kernel found"
                        except Exception:
                            pass
                    worker_states.append({
                        "id": i, "round": rounds,
                        "max_rounds": self._max_rounds, "status": status,
                        "action": action,
                    })
                else:
                    worker_states.append({
                        "id": i, "round": 0,
                        "max_rounds": self._max_rounds, "status": "waiting",
                        "action": "",
                    })
            if self._live_display:
                self._live_display.update_workers(worker_states)

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_run_agent)
            while not future.done():
                time.sleep(0.5)
                _poll_workers()
                if self._live_display:
                    self._live_display._refresh()
            future.result()

        if error:
            logger.error("KernelAgent failed: %s", error)
            return {
                "success": False,
                "kernel_code": "",
                "speedup": 0.0,
                "error": str(error),
                "elapsed": time.time() - start_time,
            }

        if result is None:
            return {
                "success": False,
                "kernel_code": "",
                "speedup": 0.0,
                "error": "KernelAgent returned no result",
                "elapsed": time.time() - start_time,
            }

        elapsed = time.time() - start_time

        # If successful, eval on Modal for speedup measurement
        kernel_code = result.get("kernel_code", "")
        speedup = 0.0

        if result.get("success") and kernel_code and self._use_modal:
            try:
                # Wrap in ModelNew if needed
                from kernel_agent.model_wrapper import wrap_in_model_new
                wrapped = wrap_in_model_new(kernel_code, self._reference)

                eval_result = _modal_eval(wrapped, self._reference)
                if eval_result.get("correct"):
                    speedup = eval_result.get("speedup", 0.0)

                if self._live_display:
                    self._live_display.update_iteration(
                        num=self._iteration_count + 1,
                        speedup=speedup,
                        status="keep" if speedup > 1.0 else "discard",
                        intent=f"KernelAgent ({self._model_name})",
                    )
                    self._iteration_count += 1
            except Exception as exc:
                logger.warning("Modal eval of final kernel failed: %s", exc)

        return {
            "success": result.get("success", False),
            "kernel_code": kernel_code,
            "speedup": speedup,
            "worker_id": result.get("worker_id"),
            "rounds": result.get("rounds", 0),
            "elapsed": elapsed,
        }

    def _configure_agent(self, agent: Any) -> None:
        """Configure agent's workers with Modal eval via env vars.

        Workers are spawned via multiprocessing.Process — we can't pass
        Python callbacks or function objects. Instead, we set env vars
        that workers read at startup:
        - OPENKERNEL_USE_MODAL=1 — tells workers to use Modal for eval
        - OPENKERNEL_REFERENCE_CODE — the reference source code
        """
        if self._use_modal:
            os.environ["OPENKERNEL_USE_MODAL"] = "1"
            os.environ["OPENKERNEL_REFERENCE_CODE"] = self._reference
