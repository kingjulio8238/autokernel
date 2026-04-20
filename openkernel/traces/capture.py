"""Trace capture for optimization sessions (feeds kernelgen-1 training pipeline).

Usage:
    tc = TraceCapture()
    tc.start_session("L1#1", "H100", "triton", "claude-sonnet-4")
    tc.record_iteration(...)
    tc.end_session(final_speedup=2.3, final_correct=True)
    trace = tc.get_trace()
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone

from openkernel.config import OpenKernelConfig
from openkernel.eval.types import CriticDiagnosis, EvalResult
from openkernel.traces.types import IterationTrace, OptimizationTrace


class TraceCapture:
    """Captures a full optimization session as an OptimizationTrace."""

    def __init__(
        self,
        session_id: str | None = None,
        config: OpenKernelConfig | None = None,
    ) -> None:
        self._session_id = session_id or uuid.uuid4().hex[:12]
        self._config = config or OpenKernelConfig()
        self._trace = OptimizationTrace(session_id=self._session_id)
        self._start_time: float | None = None
        self._started = False
        self._ended = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_session(
        self,
        problem_id: str,
        hardware: str,
        backend: str,
        model_id: str,
        problem_source: str = "kernelbench",
    ) -> None:
        """Initialize the trace with session metadata."""
        if self._started:
            raise RuntimeError("Session already started")

        self._start_time = time.monotonic()
        self._started = True

        self._trace.problem_id = problem_id
        self._trace.hardware = hardware
        self._trace.backend = backend
        self._trace.model_id = model_id
        self._trace.problem_source = problem_source
        self._trace.timestamp = datetime.now(timezone.utc).isoformat()

    def record_iteration(
        self,
        iteration: int,
        intent: str,
        generator_prompt: str,
        generator_response: str,
        kernel_code: str,
        eval_result: EvalResult,
        critic_diagnosis: CriticDiagnosis | None,
        decision: str,
        tokens_used: int,
        latency_seconds: float,
        *,
        critic_prompt: str | None = None,
        critic_response: str | None = None,
    ) -> None:
        """Record a single optimization iteration."""
        if not self._started:
            raise RuntimeError("Must call start_session() first")
        if self._ended:
            raise RuntimeError("Session already ended")

        it = IterationTrace(
            iteration=iteration,
            intent=intent,
            generator_prompt=generator_prompt,
            generator_response=generator_response,
            critic_prompt=critic_prompt,
            critic_response=critic_response,
            kernel_code=kernel_code,
            eval_status=eval_result.status.value,
            speedup=eval_result.speedup,
            runtime_us=eval_result.runtime_us,
            ref_runtime_us=eval_result.ref_runtime_us,
            profile_data=eval_result.profile.raw_metrics if eval_result.profile else {},
            bottleneck_type=(
                critic_diagnosis.bottleneck_type.value if critic_diagnosis else None
            ),
            critic_diagnosis=(
                critic_diagnosis.specific_issue if critic_diagnosis else None
            ),
            decision=decision,
            tokens_used=tokens_used,
            latency_seconds=latency_seconds,
        )

        self._trace.iterations.append(it)
        self._trace.total_tokens += tokens_used

        # Track strategies via intent
        if intent and intent not in self._trace.strategies_tried:
            self._trace.strategies_tried.append(intent)
        if decision == "keep" and intent and intent not in self._trace.strategies_succeeded:
            self._trace.strategies_succeeded.append(intent)

    def end_session(self, final_speedup: float, final_correct: bool) -> None:
        """Finalize the trace with session results."""
        if not self._started:
            raise RuntimeError("Must call start_session() first")
        if self._ended:
            raise RuntimeError("Session already ended")

        self._ended = True
        self._trace.final_speedup = final_speedup
        self._trace.final_correct = final_correct
        self._trace.total_iterations = len(self._trace.iterations)
        self._trace.total_time_seconds = time.monotonic() - self._start_time  # type: ignore[operator]

    def get_trace(self) -> OptimizationTrace:
        """Return the current OptimizationTrace (can be called at any point)."""
        return self._trace
