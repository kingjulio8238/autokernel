"""Bridge between the kernel-code TUI and the openkernel trace capture system.

Wraps :class:`openkernel.traces.capture.TraceCapture` so the CLI can record
optimization sessions for the kernelgen-1 training pipeline.

Usage::

    tb = TraceBridge(session_id="abc", config=OpenKernelConfig())
    tb.start(problem_id="L1#23", hardware="H100", backend="triton", model_id="claude-sonnet-4")
    tb.record_iteration({...})
    tb.finish(final_speedup=2.3, final_correct=True)
"""

from __future__ import annotations

import logging
from pathlib import Path

from openkernel.config import OpenKernelConfig
from openkernel.eval.types import EvalResult, EvalStatus, ProfileData
from openkernel.traces.capture import TraceCapture

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_TRACES_DIR = _PROJECT_ROOT / "traces"


class TraceBridge:
    """Thin wrapper around :class:`TraceCapture` for use by the kernel-code CLI.

    Accepts plain dicts (matching the TUI iteration schema) and translates
    them into the typed ``record_iteration`` calls that ``TraceCapture``
    expects.
    """

    def __init__(
        self,
        session_id: str,
        config: OpenKernelConfig | None = None,
    ) -> None:
        self._config = config or OpenKernelConfig()
        self._session_id = session_id
        self._capture = TraceCapture(session_id=session_id, config=self._config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(
        self,
        problem_id: str,
        hardware: str,
        backend: str,
        model_id: str,
    ) -> None:
        """Initialize trace capture for a session."""
        self._capture.start_session(
            problem_id=problem_id,
            hardware=hardware,
            backend=backend,
            model_id=model_id,
        )
        logger.info("Trace capture started: session=%s", self._session_id)

    def record_iteration(self, iteration_data: dict) -> None:
        """Record a single iteration from a plain dict.

        The dict keys match the TUI JSON schema produced by mock_data.py
        and openkernel_bridge.py.  Missing keys are filled with safe
        defaults.
        """
        profile = iteration_data.get("profile", {})
        profile_data = ProfileData(
            bandwidth_utilization=profile.get("bandwidth_util", 0.0),
            compute_utilization=profile.get("compute_util", 0.0),
            cache_efficiency=profile.get("cache_efficiency", 0.0),
            occupancy=profile.get("occupancy", 0.0),
        )

        status_str = iteration_data.get("status", "error")
        status_map = {
            "keep": EvalStatus.CORRECT,
            "correct": EvalStatus.CORRECT,
            "discard": EvalStatus.INCORRECT,
            "incorrect": EvalStatus.INCORRECT,
            "compile_error": EvalStatus.COMPILE_ERROR,
            "error": EvalStatus.ERROR,
        }
        eval_status = status_map.get(status_str, EvalStatus.ERROR)

        eval_result = EvalResult(
            status=eval_status,
            correct=eval_status == EvalStatus.CORRECT,
            speedup=iteration_data.get("speedup", 0.0),
            runtime_us=iteration_data.get("runtime_us", 0.0),
            ref_runtime_us=iteration_data.get("ref_runtime_us", 0.0),
            profile=profile_data,
            error=iteration_data.get("error"),
        )

        self._capture.record_iteration(
            iteration=iteration_data.get("iteration", 0),
            intent=iteration_data.get("intent", ""),
            generator_prompt="",  # not available from TUI-level data
            generator_response="",
            kernel_code=iteration_data.get("kernel_code_snippet", ""),
            eval_result=eval_result,
            critic_diagnosis=None,  # not available at this level
            decision=iteration_data.get("decision", ""),
            tokens_used=0,
            latency_seconds=0.0,
        )

    def finish(self, final_speedup: float, final_correct: bool) -> None:
        """End the trace capture session and save the trace.

        The trace is saved as JSON to the traces directory.  Parquet
        export can be added when the training pipeline requires it.
        """
        self._capture.end_session(
            final_speedup=final_speedup,
            final_correct=final_correct,
        )

        trace = self._capture.get_trace()

        # Save as JSON (Parquet export is a future enhancement)
        _TRACES_DIR.mkdir(parents=True, exist_ok=True)
        out_path = _TRACES_DIR / f"{self._session_id}.json"
        try:
            import dataclasses
            import json

            def _serialize(obj):
                if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
                    return dataclasses.asdict(obj)
                raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

            out_path.write_text(json.dumps(
                dataclasses.asdict(trace),
                indent=2,
                default=_serialize,
            ))
            logger.info("Trace saved: %s", out_path)
        except Exception:
            logger.warning("Failed to save trace to %s", out_path, exc_info=True)

    @property
    def trace(self):
        """Access the underlying OptimizationTrace (for inspection/testing)."""
        return self._capture.get_trace()
