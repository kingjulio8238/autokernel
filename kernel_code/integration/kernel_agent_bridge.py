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
    from kernel_code.optimization_log import OptimizationLog
    from kernel_code.run_log import RunLogger

from kernel_code.optimization_gate import is_removable_kernel
from kernel_code.problem_classifier import classify_problem

logger = logging.getLogger(__name__)


def _detect_dtype_simple(code: str) -> str:
    """Quick dtype detection from code."""
    if "float16" in code or "fp16" in code:
        return "float16"
    if "bfloat16" in code or "bf16" in code:
        return "bfloat16"
    return "float32"


def _strip_dead_code(code: str) -> str:
    """Remove dead code the LLM often adds: __main__ blocks, inline tests.

    Keeps kernel + wrapper function only. Safety net for the prompt instruction.
    """
    if not code:
        return code

    # Remove `if __name__ == "__main__":` block and everything after
    lines = code.split("\n")
    cleaned = []
    skip_until_dedent = False
    main_indent = 0

    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("if __name__"):
            skip_until_dedent = True
            main_indent = len(line) - len(stripped)
            continue
        if skip_until_dedent:
            # Still inside __main__ block if indented more than its header
            if line.strip() == "" or (len(line) - len(line.lstrip())) > main_indent:
                continue
            # Dedented back — stop skipping
            skip_until_dedent = False
        cleaned.append(line)

    # Remove trailing blank lines
    while cleaned and not cleaned[-1].strip():
        cleaned.pop()

    return "\n".join(cleaned) + "\n"


from kernel_code.gpu_functions import GPU_FUNCTION_MAP as _GPU_FUNCTION_MAP


def _modal_eval(
    kernel_code: str,
    reference_code: str,
    problem_format: str = "auto",
    gpu_type: str = "L40S",
) -> dict:
    """Evaluate a kernel via Modal remote GPU.

    Routes to the correct GPU-specific Modal function based on gpu_type.
    KernelBench leaderboard standardizes on L40S — other GPU results
    are not directly comparable.
    """
    import modal

    fn_name = _GPU_FUNCTION_MAP.get(gpu_type, "eval_kernel_on_gpu")
    eval_fn = modal.Function.from_name("openkernel-eval", fn_name)
    result = eval_fn.remote(
        kernel_source=kernel_code,
        reference_source=reference_code,
        eval_mode="fast",
        problem_format=problem_format,
        gpu_type=gpu_type,
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
        problem_format: str = "auto",  # "auto", "kernelbench", "gpumode"
        optimization_log: "OptimizationLog | None" = None,
    ) -> None:
        self._reference = reference_source
        self._model_name = model_name
        self._num_workers = num_workers
        self._max_rounds = max_rounds
        self._hardware = hardware
        self._live_display = live_display
        self._run_logger = run_logger
        self._use_modal = use_modal
        self._problem_format = problem_format
        self._optimization_log = optimization_log
        self._per_worker_latest: list[dict] = []

        # Start from global best if provided (for multi-round autopilot)
        global_best = float(os.environ.get("OPENKERNEL_BEST_SPEEDUP", "0.0"))
        self._best_speedup: float = global_best
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
            "anthropic", "litellm", "LiteLLM",
            "kernel_agent.ka_utils", "kernel_agent.ka_utils.providers",
            "kernel_agent.ka_utils.providers.openai_base",
            "kernel_agent.manager", "kernel_agent.worker",
            "kernel_agent.agent", "kernel_agent.prompt_manager",
            "kernel_code.ke_profile",
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

        # Use Problem interface for format-aware test code
        from kernel_code.problem import (
            load_problem, build_test_code, detect_format,
            make_self_contained, Problem,
        )

        # Detect problem format
        fmt = self._problem_format
        if fmt == "auto":
            fmt = detect_format(self._reference)
        self._resolved_format = fmt

        # Build a Problem instance — try loading from file for full context
        ref_path = Path(os.environ.get("OPENKERNEL_REFERENCE_PATH", "reference.py"))
        if ref_path.is_file():
            try:
                problem = load_problem(ref_path)
            except Exception:
                problem = Problem(
                    reference_code=self._reference, format=fmt,
                    dtype=_detect_dtype_simple(self._reference),
                )
        else:
            problem = Problem(
                reference_code=self._reference, format=fmt,
                dtype=_detect_dtype_simple(self._reference),
            )

        # Make reference self-contained (inline task.py, utils.py for GPU Mode)
        self_contained_ref = make_self_contained(problem)
        self._self_contained_ref = self_contained_ref

        # --- Classify problem type (used for strategy selection + skill matching) ---
        classification = classify_problem(self._reference)

        # --- Few-shot: find a working solution for a similar problem ---
        few_shot = self._find_few_shot_example(ref_path, classification=classification)

        # --- Best kernel carry-forward from previous round ---
        prev_kernel = os.environ.get("OPENKERNEL_BEST_KERNEL", "")
        prev_speedup = os.environ.get("OPENKERNEL_BEST_SPEEDUP", "")

        # Build problem description with all context
        problem_desc = (
            f"Optimize the following PyTorch code into a fast Triton kernel "
            f"for {self._hardware} GPU.\n\n"
            f"Reference implementation:\n```python\n{self_contained_ref}\n```\n\n"
            f"CRITICAL REQUIREMENTS:\n"
            f"- Use the EXACT same dtypes as the reference implementation\n"
            f"- The kernel must be correct: torch.allclose(ref, kernel, rtol=1e-2, atol=1e-2)\n"
            f"- Output dtype must match reference output dtype\n"
            f"- Do NOT hardcode bfloat16 or float16 — use the input tensor's dtype\n"
            f"\n"
            f"PERFORMANCE REQUIREMENTS:\n"
            f"- MUST use @triton.autotune with 4+ configs varying BLOCK_SIZE "
            f"(256, 512, 1024, 2048) and num_warps (2, 4, 8)\n"
            f"- For memory-bound kernels (elementwise, reductions, norms), "
            f"prioritize vectorized loads and coalesced access\n"
            f"- Do NOT include shape/dtype/device validation in kernel_function — "
            f"inputs are pre-validated by the test harness\n"
            f"- Do NOT include if __name__ == '__main__' blocks or test code — "
            f"only the kernel and its Python wrapper\n"
            f"- Keep comments minimal — one-line description per function max, "
            f"no multi-paragraph docstrings"
        )

        # --- Deterministic gate: skip trivial kernels ---
        if is_removable_kernel(self._reference):
            logger.warning("Removable kernel detected — skipping optimization")
            if self._live_display:
                self._live_display.print_permanent(
                    "  [yellow]Skipped: trivial kernel (identity/memcpy/view-only)[/yellow]"
                )
            return {
                "success": False,
                "kernel_code": "",
                "speedup": 0.0,
                "error": "Removable kernel — optimization skipped",
                "elapsed": 0.0,
            }

        # Inject classification context (classification computed earlier for few-shot)
        problem_desc += f"\n\n{classification.to_context_string()}"

        # Add few-shot example if available
        if few_shot:
            problem_desc += (
                f"\n\n## WORKING EXAMPLE (similar problem)\n"
                f"Here is a correct, working Triton kernel for a similar problem. "
                f"Use it as a reference for Triton syntax and patterns:\n"
                f"```python\n{few_shot}\n```"
            )

        # Add best kernel from previous round for refinement
        if prev_kernel and prev_speedup:
            problem_desc += (
                f"\n\n## PREVIOUS BEST KERNEL ({prev_speedup}x speedup)\n"
                f"This kernel works but is not fast enough. Improve it:\n"
                f"```python\n{prev_kernel}\n```\n"
                f"Focus on: vectorized loads, autotune configs, memory coalescing."
            )

        # Add optimization trajectory context if available
        if self._optimization_log and self._optimization_log.rounds:
            problem_desc += (
                f"\n\n## OPTIMIZATION TRAJECTORY\n"
                f"Previous rounds of optimization have been attempted. "
                f"Study what worked and what didn't, then try a DIFFERENT approach:\n"
                f"{self._optimization_log.to_context_string(max_rounds=5)}"
            )

        # Build format-appropriate test code (uses self-contained ref)
        problem.reference_code = self_contained_ref
        test_code = build_test_code(problem)

        # Create agent
        agent = TritonKernelAgent(
            num_workers=self._num_workers,
            max_rounds=self._max_rounds,
            model_name=self._model_name,
        )

        # Configure workers with Modal eval and callbacks
        self._configure_agent(agent)

        # Pass classification context to live display
        if self._live_display and hasattr(self._live_display, '_problem_tier'):
            self._live_display._problem_tier = classification.tier.value
            self._live_display._problem_type = classification.op_type.value
            self._live_display._is_memory_bound = classification.is_memory_bound_likely

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
                    test_code=test_code,
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
                        # Truncate at word boundary
                        if len(msg) > 25:
                            return msg[:25].rsplit(" ", 1)[0] + "\u2026"
                        return msg
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
                    speedup = 0.0
                    if rounds > 0:
                        try:
                            import re as _re
                            # Scan all rounds for best speedup (for Plot A),
                            # but use the LATEST round's success for worker status
                            best_speedup = 0.0
                            latest_success = False
                            for rf in sorted(round_files, key=lambda p: p.name):
                                try:
                                    data = json.loads(rf.read_text())
                                    sp = float(data.get("speedup", 0.0))
                                    if sp == 0.0:
                                        stdout = data.get("stdout", "")
                                        sp_match = _re.search(r"Speedup:\s*([\d.]+)x", stdout)
                                        if sp_match:
                                            sp = float(sp_match.group(1))
                                    if sp > best_speedup:
                                        best_speedup = sp
                                    # Track latest round's success (sorted order
                                    # means last iteration = most recent file)
                                    latest_success = bool(data.get("success"))
                                except Exception:
                                    continue
                            speedup = best_speedup
                            if latest_success:
                                status = "passed"
                                action = f"correct ({speedup:.2f}x)" if speedup > 0 else "correct kernel found"
                        except Exception:
                            pass
                    # Local ID for worker bar display, global ID for plot tracking
                    round_offset = int(os.environ.get("OPENKERNEL_ROUND_OFFSET", "0"))
                    worker_states.append({
                        "id": i,  # local ID for Worker 1, Worker 2 display
                        "global_id": round_offset + i,  # global ID for Plot A
                        "round": rounds,
                        "max_rounds": self._max_rounds, "status": status,
                        "action": action, "speedup": speedup,
                    })
                else:
                    round_offset = int(os.environ.get("OPENKERNEL_ROUND_OFFSET", "0"))
                    worker_states.append({
                        "id": i,
                        "global_id": round_offset + i,
                        "round": 0,
                        "max_rounds": self._max_rounds, "status": "waiting",
                        "action": "",
                    })
            if self._live_display:
                self._live_display.update_workers(worker_states)
            # Capture for round-summary plot. Profile populated later after
            # the final Modal eval — we only have per-worker speedup here from
            # KernelAgent's round files (no profile in those).
            self._per_worker_latest = [
                {"id": w["id"], "speedup": w.get("speedup", 0.0), "profile": {}}
                for w in worker_states
            ]

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

        # Strip dead code the model often adds (defensive cleanup)
        kernel_code = _strip_dead_code(kernel_code)

        speedup = 0.0

        profile = {}
        ref_runtime_us = 0.0
        kernel_runtime_us = 0.0

        if result.get("success") and kernel_code and self._use_modal:
            try:
                # Only wrap in ModelNew for KernelBench format
                # GPU Mode format uses kernel_function() directly
                if self._resolved_format == "kernelbench":
                    from kernel_agent.model_wrapper import wrap_in_model_new
                    eval_code = wrap_in_model_new(kernel_code, self._reference)
                else:
                    eval_code = kernel_code

                eval_result = _modal_eval(
                    eval_code, self._self_contained_ref,
                    problem_format=self._resolved_format,
                    gpu_type=self._hardware,
                )
                if eval_result.get("correct"):
                    speedup = eval_result.get("speedup", 0.0)
                    ref_runtime_us = eval_result.get("ref_runtime_us", 0.0)
                    kernel_runtime_us = eval_result.get("runtime_us", 0.0)
                    profile = eval_result.get("profile", {})

                if self._live_display:
                    # Only "keep" if it beats the current best
                    if speedup > self._best_speedup:
                        status = "keep"
                        self._best_speedup = speedup
                    else:
                        status = "discard"

                    # Show the optimization strategy, not "KernelAgent (gpt-4o)"
                    strategy = os.environ.get("OPENKERNEL_CURRENT_STRATEGY", "")
                    intent = strategy[:50] if strategy else f"round {self._iteration_count + 1}"

                    # Compute SOL score for live display
                    _sol = 0.0
                    if profile and ref_runtime_us > 0 and kernel_runtime_us > 0:
                        try:
                            from kernel_code.sol_metrics import compute_sol_score
                            _sol = compute_sol_score(
                                kernel_runtime_us=kernel_runtime_us,
                                ref_runtime_us=ref_runtime_us,
                                total_flops=profile.get("total_flops", 0),
                                total_bytes=profile.get("total_bytes", 0),
                                gpu_type=self._hardware,
                            )
                        except Exception:
                            pass

                    self._live_display.update_iteration(
                        num=self._iteration_count + 1,
                        speedup=speedup,
                        status=status,
                        intent=intent,
                        sol_score=_sol,
                    )
                    # Pass profile metrics to live display
                    if hasattr(self._live_display, 'update_profile') and profile:
                        self._live_display.update_profile(
                            bw_pct=profile.get("bandwidth_utilization", 0.0) * 100,
                            compute_pct=profile.get("compute_utilization", 0.0) * 100,
                        )
                    self._iteration_count += 1
            except Exception as exc:
                logger.warning("Modal eval of final kernel failed: %s", exc)

        # Attach the final Modal profile to the winning worker's per-worker
        # entry (identified by worker_id or highest speedup fallback). Only
        # one profile exists per round — KernelAgent's per-round JSON files
        # don't include profiling data, so non-winning workers get {}.
        winning_id = result.get("worker_id")
        if profile and self._per_worker_latest:
            target = None
            if winning_id is not None:
                for w in self._per_worker_latest:
                    if w.get("id") == winning_id:
                        target = w
                        break
            if target is None:
                target = max(
                    self._per_worker_latest,
                    key=lambda w: w.get("speedup", 0.0) or 0.0,
                )
            target["profile"] = profile

        return {
            "success": result.get("success", False),
            "kernel_code": kernel_code,
            "speedup": speedup,
            "ref_runtime_us": ref_runtime_us,
            "kernel_runtime_us": kernel_runtime_us,
            "profile": profile,
            "worker_id": result.get("worker_id"),
            "rounds": result.get("rounds", self._max_rounds),
            "elapsed": elapsed,
            "per_worker": self._per_worker_latest,
        }

    def _find_few_shot_example(self, ref_path: Path, classification: Any = None) -> str:
        """Find a relevant Triton code example for the problem.

        Four-level search:
        1. Exact solution (GPU Mode benchmarks with solutions/correct/)
        2. KernelBook match from HuggingFace (25K PyTorch→Triton pairs)
        3. Skill template match (10 skills with code_template by problem type)
        4. Generic Triton pattern (vectoradd as universal example)
        """
        # Level 1: Exact solution from benchmark
        for search_dir in [ref_path.parent, ref_path.parent.parent]:
            solutions_dir = search_dir / "solutions" / "correct"
            if solutions_dir.is_dir():
                for sol in sorted(solutions_dir.iterdir()):
                    if sol.suffix == ".py" and "triton" in sol.name.lower():
                        return sol.read_text()

        # Level 2: KernelBook match from HuggingFace
        try:
            from kernel_code.kernelbook import find_similar_example
            kb_match = find_similar_example(self._reference)
            if kb_match:
                return kb_match
        except Exception:
            pass  # KernelBook unavailable — continue to skill match

        # Level 3: Match skills library by problem keywords + classification boost
        ref_code = self._reference.lower()
        skills_dir = Path(__file__).resolve().parent.parent.parent / "data" / "skills"
        if skills_dir.is_dir():
            import json as _json
            # Use passed classification or compute fresh
            try:
                if classification is None:
                    classification = classify_problem(self._reference)
                op_type = classification.op_type.value
                tier = classification.tier.value
                is_mem = classification.is_memory_bound_likely
            except Exception:
                op_type = ""
                tier = ""
                is_mem = None

            best_skill = None
            best_score = 0
            for skill_file in skills_dir.glob("*.json"):
                try:
                    skill = _json.loads(skill_file.read_text())
                    keywords = skill.get("tags", [])
                    trigger_kw = skill.get("auto_trigger", {}).get("problem_keywords", [])
                    all_kw = keywords + trigger_kw
                    # Prefer triton skills
                    backend = skill.get("backend", "")
                    is_triton = backend == "triton" or "triton" in skill.get("id", "")
                    score = sum(1 for kw in all_kw if kw in ref_code)
                    if is_triton:
                        score += 5

                    # Classification boost: +3 for matching op_type in tags
                    if op_type:
                        tags_lower = [t.lower() for t in keywords]
                        if op_type in tags_lower:
                            score += 3
                    # Hardware match boost: +2 for matching hardware in tags
                    hw_lower = self._hardware.lower()
                    if any(hw_lower in t.lower() for t in keywords):
                        score += 2
                    # Memory/compute bound match: +2
                    if is_mem is not None:
                        if is_mem and "memory_bound" in [t.lower() for t in keywords]:
                            score += 2
                        elif not is_mem and "compute_bound" in [t.lower() for t in keywords]:
                            score += 2

                    if score > best_score and skill.get("code_template"):
                        best_score = score
                        best_skill = skill
                except Exception:
                    continue
            if best_skill and best_skill.get("code_template"):
                return best_skill["code_template"]

        return ""

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
            # Send self-contained reference so Modal has all dependencies
            os.environ["OPENKERNEL_REFERENCE_CODE"] = self._self_contained_ref
            os.environ["OPENKERNEL_PROBLEM_FORMAT"] = self._resolved_format
            os.environ["OPENKERNEL_GPU_TYPE"] = self._hardware
