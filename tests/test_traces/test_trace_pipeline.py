"""End-to-end test: capture -> save -> load -> export."""

from __future__ import annotations

import tempfile
from pathlib import Path

from openkernel.eval.types import (
    BottleneckType,
    CriticDiagnosis,
    EvalResult,
    EvalStatus,
    ProfileData,
)
from openkernel.traces import (
    TraceCapture,
    export_strategy_rewards,
    export_training_pairs,
    list_traces,
    load_trace,
    save_processed,
    save_trace,
)


def _make_eval(status: EvalStatus, speedup: float) -> EvalResult:
    return EvalResult(
        status=status,
        correct=status == EvalStatus.CORRECT,
        speedup=speedup,
        runtime_us=100.0 / max(speedup, 0.01),
        ref_runtime_us=100.0,
        profile=ProfileData(raw_metrics={"occupancy": 0.8}),
    )


def _make_critic(issue: str) -> CriticDiagnosis:
    return CriticDiagnosis(
        bottleneck_type=BottleneckType.MEMORY_BOUND,
        specific_issue=issue,
        recommendation="Use tiling",
        confidence=0.9,
    )


def test_capture_record_end():
    """TraceCapture lifecycle: start -> 3 iterations -> end."""
    tc = TraceCapture(session_id="test-001")
    tc.start_session("L1#1", "H100", "triton", "test-model")

    # Iteration 1: compile error
    tc.record_iteration(
        iteration=1,
        intent="naive_tiling",
        generator_prompt="optimize this kernel",
        generator_response="```triton\n...\n```",
        kernel_code="@triton.jit\ndef k(): pass",
        eval_result=_make_eval(EvalStatus.COMPILE_ERROR, 0.0),
        critic_diagnosis=None,
        decision="retry",
        tokens_used=500,
        latency_seconds=1.2,
    )

    # Iteration 2: correct but slow
    tc.record_iteration(
        iteration=2,
        intent="naive_tiling",
        generator_prompt="optimize this kernel (retry)",
        generator_response="```triton\n...\n```",
        kernel_code="@triton.jit\ndef k(): ...",
        eval_result=_make_eval(EvalStatus.CORRECT, 0.8),
        critic_diagnosis=_make_critic("strided access"),
        decision="discard",
        tokens_used=600,
        latency_seconds=1.5,
    )

    # Iteration 3: success
    tc.record_iteration(
        iteration=3,
        intent="coalesced_tiling",
        generator_prompt="use coalesced access",
        generator_response="```triton\n...\n```",
        kernel_code="@triton.jit\ndef k_fast(): ...",
        eval_result=_make_eval(EvalStatus.CORRECT, 2.3),
        critic_diagnosis=_make_critic("looks good"),
        decision="keep",
        tokens_used=700,
        latency_seconds=2.0,
    )

    tc.end_session(final_speedup=2.3, final_correct=True)
    trace = tc.get_trace()

    assert trace.session_id == "test-001"
    assert trace.total_iterations == 3
    assert trace.final_speedup == 2.3
    assert trace.final_correct is True
    assert trace.total_tokens == 500 + 600 + 700
    assert len(trace.iterations) == 3
    assert trace.strategies_tried == ["naive_tiling", "coalesced_tiling"]
    assert trace.strategies_succeeded == ["coalesced_tiling"]


def test_save_load_roundtrip():
    """Save a trace to Parquet and load it back with identical data."""
    tc = TraceCapture(session_id="roundtrip-001")
    tc.start_session("L2#5", "A100", "cuda", "gpt-4")

    tc.record_iteration(
        iteration=1,
        intent="vectorize",
        generator_prompt="prompt1",
        generator_response="response1",
        kernel_code="__global__ void k() {}",
        eval_result=_make_eval(EvalStatus.CORRECT, 1.5),
        critic_diagnosis=_make_critic("bandwidth limited"),
        decision="keep",
        tokens_used=400,
        latency_seconds=1.0,
    )

    tc.record_iteration(
        iteration=2,
        intent="shared_memory",
        generator_prompt="prompt2",
        generator_response="response2",
        kernel_code="__global__ void k2() {}",
        eval_result=_make_eval(EvalStatus.CORRECT, 3.1),
        critic_diagnosis=None,
        decision="keep",
        tokens_used=500,
        latency_seconds=1.1,
    )

    tc.end_session(final_speedup=3.1, final_correct=True)
    trace = tc.get_trace()

    with tempfile.TemporaryDirectory() as tmpdir:
        raw_dir = Path(tmpdir) / "raw"
        path = save_trace(trace, output_dir=raw_dir)

        assert path.exists()
        assert "session_roundtrip-001.parquet" in path.name

        loaded = load_trace(path)

        assert loaded.session_id == trace.session_id
        assert loaded.problem_id == trace.problem_id
        assert loaded.hardware == trace.hardware
        assert loaded.backend == trace.backend
        assert loaded.model_id == trace.model_id
        assert loaded.final_speedup == trace.final_speedup
        assert loaded.final_correct == trace.final_correct
        assert loaded.total_iterations == trace.total_iterations
        assert loaded.total_tokens == trace.total_tokens
        assert loaded.strategies_tried == trace.strategies_tried
        assert loaded.strategies_succeeded == trace.strategies_succeeded

        assert len(loaded.iterations) == 2
        assert loaded.iterations[0].intent == "vectorize"
        assert loaded.iterations[0].kernel_code == "__global__ void k() {}"
        assert loaded.iterations[0].eval_status == "correct"
        assert loaded.iterations[0].speedup == 1.5
        assert loaded.iterations[0].profile_data == {"occupancy": 0.8}
        assert loaded.iterations[1].intent == "shared_memory"


def test_list_traces():
    """list_traces finds all Parquet files."""
    tc = TraceCapture(session_id="list-001")
    tc.start_session("L1#1", "H100", "triton", "test")
    tc.record_iteration(
        iteration=1, intent="x", generator_prompt="p", generator_response="r",
        kernel_code="k", eval_result=_make_eval(EvalStatus.CORRECT, 1.0),
        critic_diagnosis=None, decision="keep", tokens_used=10, latency_seconds=0.1,
    )
    tc.end_session(final_speedup=1.0, final_correct=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        raw_dir = Path(tmpdir) / "raw"
        save_trace(tc.get_trace(), output_dir=raw_dir)

        found = list_traces(raw_dir)
        assert len(found) == 1
        assert found[0].name == "session_list-001.parquet"


def test_export_training_pairs():
    """export_training_pairs filters and formats correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_dir = Path(tmpdir) / "raw"

        # Good trace: correct, speedup >= 1.0, has a "keep" iteration
        tc1 = TraceCapture(session_id="good-001")
        tc1.start_session("L1#1", "H100", "triton", "model-a")
        tc1.record_iteration(
            iteration=1, intent="tile", generator_prompt="p1", generator_response="r1",
            kernel_code="good_kernel_code", eval_result=_make_eval(EvalStatus.CORRECT, 2.0),
            critic_diagnosis=None, decision="keep", tokens_used=100, latency_seconds=0.5,
        )
        tc1.end_session(final_speedup=2.0, final_correct=True)
        save_trace(tc1.get_trace(), output_dir=raw_dir)

        # Bad trace: incorrect
        tc2 = TraceCapture(session_id="bad-001")
        tc2.start_session("L1#2", "H100", "triton", "model-a")
        tc2.record_iteration(
            iteration=1, intent="tile", generator_prompt="p2", generator_response="r2",
            kernel_code="bad_kernel_code", eval_result=_make_eval(EvalStatus.INCORRECT, 0.0),
            critic_diagnosis=None, decision="discard", tokens_used=100, latency_seconds=0.5,
        )
        tc2.end_session(final_speedup=0.0, final_correct=False)
        save_trace(tc2.get_trace(), output_dir=raw_dir)

        df = export_training_pairs(raw_dir, min_speedup=1.0)

        assert len(df) == 1
        assert df.iloc[0]["prompt"] == "p1"
        assert df.iloc[0]["response"] == "good_kernel_code"
        assert df.iloc[0]["reward"] == 2.0
        assert df.iloc[0]["problem_id"] == "L1#1"
        assert df.iloc[0]["backend"] == "triton"
        assert df.iloc[0]["model_id"] == "model-a"


def test_export_strategy_rewards():
    """export_strategy_rewards extracts strategy-level data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_dir = Path(tmpdir) / "raw"

        tc = TraceCapture(session_id="strat-001")
        tc.start_session("L1#3", "H100", "triton", "model-b")
        tc.record_iteration(
            iteration=1, intent="tiling", generator_prompt="p", generator_response="r",
            kernel_code="k", eval_result=_make_eval(EvalStatus.CORRECT, 1.5),
            critic_diagnosis=None, decision="keep", tokens_used=100, latency_seconds=0.5,
        )
        tc.record_iteration(
            iteration=2, intent="vectorize", generator_prompt="p2", generator_response="r2",
            kernel_code="k2", eval_result=_make_eval(EvalStatus.CORRECT, 2.5),
            critic_diagnosis=None, decision="keep", tokens_used=100, latency_seconds=0.5,
        )
        tc.end_session(final_speedup=2.5, final_correct=True)
        save_trace(tc.get_trace(), output_dir=raw_dir)

        df = export_strategy_rewards(raw_dir)

        assert len(df) == 2
        assert set(df["strategy"]) == {"tiling", "vectorize"}
        assert all(df["final_speedup"] == 2.5)


def test_save_processed():
    """save_processed writes a DataFrame to Parquet."""
    import pandas as pd

    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "processed" / "test.parquet"
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        result = save_processed(df, out)

        assert result.exists()
        loaded = pd.read_parquet(result)
        assert list(loaded.columns) == ["a", "b"]
        assert len(loaded) == 2
