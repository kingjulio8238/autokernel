"""End-to-end system test for the openkernel 3-level hybrid engine.

Exercises the full optimization pipeline using mock implementations from
tests.mocks so that no real GPU or LLM API keys are required.

Run directly:
    python tests/test_e2e/test_full_loop.py
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the repo root is on sys.path so "openkernel" is importable when
# running the script directly (python tests/test_e2e/test_full_loop.py).
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Imports from the project
# ---------------------------------------------------------------------------
from openkernel.config import OpenKernelConfig
from openkernel.engine.orchestrator import (
    Orchestrator,
    OptimizationResult,
)
from tests.mocks import MockInnerLoop, MockLLMCaller
from openkernel.engine.strategy_evolution import StrategyEvolution
from openkernel.engine.world_model import IntentTree
from openkernel.eval.types import (
    BottleneckType,
    CriticDiagnosis,
    EvalResult,
    EvalStatus,
)
from openkernel.memory.skill_library import SkillLibrary
from openkernel.traces.capture import TraceCapture
from openkernel.traces.types import OptimizationTrace

# ---------------------------------------------------------------------------
# Fake reference source (simple torch.matmul — same shape as reference.py)
# ---------------------------------------------------------------------------
REFERENCE_SOURCE = '''\
import torch
import torch.nn as nn

class Model(nn.Module):
    """Simple model that performs a single square matrix multiplication (C = A * B)"""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return torch.matmul(A, B)

M = 4096
K = 4096
N = 4096

def get_inputs():
    A = torch.randn(M, K)
    B = torch.randn(K, N)
    return [A, B]

def get_init_inputs():
    return []
'''

# ---------------------------------------------------------------------------
# Test runner helpers
# ---------------------------------------------------------------------------
_passed = 0
_failed = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    """Record and print a pass/fail check."""
    global _passed, _failed
    if condition:
        _passed += 1
        status = "PASS"
    else:
        _failed += 1
        status = "FAIL"
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{status}] {label}{suffix}")


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


# ===================================================================
# Test 1: Full orchestrator optimization loop (MockInnerLoop + MockLLMCaller)
# ===================================================================
def test_full_optimization_loop() -> OptimizationResult | None:
    section("Test 1: Full Orchestrator Optimization Loop")

    config = OpenKernelConfig(max_iterations=10)

    # Build the orchestrator with mocks — no API keys needed.
    mock_inner = MockInnerLoop(seed=42, success_rate=0.6)
    mock_llm = MockLLMCaller(seed=42)
    strategy_evo = StrategyEvolution(
        strategies_dir=_REPO_ROOT / "data" / "strategies"
    )
    strategy_evo.load()

    orch_config = {
        "max_iterations": config.max_iterations,
        "stagnation_threshold": config.stagnation_threshold,
        "max_retries_per_intent": config.max_retries_per_intent,
    }

    orchestrator = Orchestrator(
        inner_loop=mock_inner,
        config=orch_config,
        llm=mock_llm,
        strategy_evolution=strategy_evo,
    )

    start = time.monotonic()
    result = orchestrator.optimize(
        reference_code=REFERENCE_SOURCE,
        backend="triton",
        hardware="H100",
    )
    elapsed = time.monotonic() - start

    # Verify it returned an OptimizationResult
    check(
        "optimize() returns OptimizationResult",
        isinstance(result, OptimizationResult),
        f"got {type(result).__name__}",
    )

    # Verify final_speedup > 0
    check(
        "final_speedup > 0",
        result.final_speedup > 0,
        f"final_speedup={result.final_speedup:.3f}",
    )

    # Verify tree was built (tree_history has snapshots with nodes)
    check(
        "tree_history is non-empty",
        len(result.tree_history) > 0,
        f"snapshots={len(result.tree_history)}",
    )

    # Check that the tree has more than just the root node
    last_snapshot = result.tree_history[-1]
    node_count = len(last_snapshot.get("nodes", {}))
    check(
        "IntentTree has multiple nodes",
        node_count > 1,
        f"nodes={node_count}",
    )

    # Multiple iterations ran
    check(
        "Multiple iterations ran",
        result.iterations_total > 1,
        f"iterations_total={result.iterations_total}",
    )

    # Multiple intents explored
    check(
        "Intents explored > 0",
        result.intents_explored > 0,
        f"explored={result.intents_explored}",
    )

    # Print summary
    print(f"\n  Summary:")
    print(f"    Final speedup:        {result.final_speedup:.3f}x")
    print(f"    Total iterations:     {result.iterations_total}")
    print(f"    Intents explored:     {result.intents_explored}")
    print(f"    Intents succeeded:    {result.intents_succeeded}")
    print(f"    Intents failed:       {result.intents_failed}")
    print(f"    Stagnation triggered: {result.stagnation_triggered}")
    print(f"    Tree snapshots:       {len(result.tree_history)}")
    print(f"    Nodes in final tree:  {node_count}")
    print(f"    Wall time:            {elapsed:.3f}s")

    return result


# ===================================================================
# Test 2: Trace capture
# ===================================================================
def test_trace_capture() -> None:
    section("Test 2: Trace Capture")

    tc = TraceCapture(session_id="test-e2e-001")
    tc.start_session(
        problem_id="L1#1",
        hardware="H100",
        backend="triton",
        model_id="mock-llm",
    )

    # Record a few mock iterations
    for i in range(1, 4):
        eval_result = EvalResult(
            status=EvalStatus.CORRECT,
            correct=True,
            speedup=1.0 + 0.3 * i,
            runtime_us=100.0 / (1.0 + 0.3 * i),
            ref_runtime_us=100.0,
        )
        diagnosis = CriticDiagnosis(
            bottleneck_type=BottleneckType.MEMORY_BOUND,
            roofline_position=0.5,
            specific_issue="strided memory access",
            recommendation="use vectorized loads",
            estimated_headroom=1.5,
            confidence=0.8,
        )
        tc.record_iteration(
            iteration=i,
            intent=f"Optimize iteration {i}",
            generator_prompt=f"Generate kernel v{i}",
            generator_response=f"# kernel v{i}\nimport triton",
            kernel_code=f"# kernel v{i}\nimport triton",
            eval_result=eval_result,
            critic_diagnosis=diagnosis,
            decision="keep" if i > 1 else "discard",
            tokens_used=1000 * i,
            latency_seconds=2.0 * i,
        )

    tc.end_session(final_speedup=1.9, final_correct=True)
    trace = tc.get_trace()

    check(
        "get_trace() returns OptimizationTrace",
        isinstance(trace, OptimizationTrace),
        f"got {type(trace).__name__}",
    )
    check(
        "session_id matches",
        trace.session_id == "test-e2e-001",
        f"session_id={trace.session_id}",
    )
    check(
        "3 iterations recorded",
        trace.total_iterations == 3,
        f"total_iterations={trace.total_iterations}",
    )
    check(
        "final_speedup == 1.9",
        trace.final_speedup == 1.9,
        f"final_speedup={trace.final_speedup}",
    )
    check(
        "final_correct is True",
        trace.final_correct is True,
    )
    check(
        "total_tokens accumulated",
        trace.total_tokens == 6000,
        f"total_tokens={trace.total_tokens}",
    )
    check(
        "strategies_tried is non-empty",
        len(trace.strategies_tried) > 0,
        f"strategies_tried={trace.strategies_tried}",
    )
    check(
        "strategies_succeeded is non-empty",
        len(trace.strategies_succeeded) > 0,
        f"strategies_succeeded={trace.strategies_succeeded}",
    )
    check(
        "total_time_seconds > 0",
        trace.total_time_seconds > 0,
        f"total_time_seconds={trace.total_time_seconds:.3f}",
    )


# ===================================================================
# Test 3: Skill library
# ===================================================================
def test_skill_library() -> None:
    section("Test 3: Skill Library")

    lib = SkillLibrary(skills_dir=_REPO_ROOT / "data" / "skills")
    lib.load()

    all_skills = lib.all_skills
    check(
        "5 pre-seeded skills loaded",
        len(all_skills) == 5,
        f"loaded={len(all_skills)}",
    )

    # Search for "softmax" skills
    softmax_results = lib.search_skills("softmax reduction memory-bound", backend="triton")
    check(
        "search_skills('softmax') returns results",
        len(softmax_results) > 0,
        f"found={len(softmax_results)}",
    )

    # Verify returned skills have expected structure
    if softmax_results:
        skill = softmax_results[0]
        check(
            "First skill has non-empty name",
            bool(skill.name),
            f"name={skill.name!r}",
        )
        check(
            "First skill has non-empty approach",
            bool(skill.approach),
            f"approach={skill.approach[:60]!r}...",
        )

    # Search for "gemm" — should find both triton and cuda GEMM skills
    gemm_results = lib.search_skills("gemm compute-bound tiling")
    check(
        "search_skills('gemm') returns results",
        len(gemm_results) > 0,
        f"found={len(gemm_results)}",
    )

    # Print skill names for visibility
    print(f"\n  All loaded skills:")
    for s in all_skills:
        print(f"    - {s.name} (backend={s.backend}, id={s.id})")


# ===================================================================
# Test 4: Strategy evolution
# ===================================================================
def test_strategy_evolution() -> None:
    section("Test 4: Strategy Evolution")

    evo = StrategyEvolution(strategies_dir=_REPO_ROOT / "data" / "strategies")
    evo.load()

    all_strats = evo.all_strategies
    check(
        "3 pre-seeded strategies loaded",
        len(all_strats) == 3,
        f"loaded={len(all_strats)}",
    )

    # Retrieve strategies for a "memory bound" problem
    memory_bound = evo.retrieve_strategies(
        problem_description="softmax memory bound reduction kernel",
        backend="triton",
        top_k=3,
    )
    check(
        "retrieve_strategies('memory bound') returns results",
        len(memory_bound) > 0,
        f"found={len(memory_bound)}",
    )

    if memory_bound:
        top = memory_bound[0]
        check(
            "Top strategy has 'memory' in description",
            "memory" in top.description.lower(),
            f"description={top.description[:60]!r}...",
        )

    # Verify Pareto frontier (all untested strategies are on the frontier)
    frontier = evo.frontier
    check(
        "Pareto frontier includes all strategies (none have scores yet)",
        len(frontier) == len(all_strats),
        f"frontier={len(frontier)}, total={len(all_strats)}",
    )

    # Test update_strategy — record a mock result
    if all_strats:
        sid = all_strats[0].id
        evo.update_strategy(sid, {
            "speedup": 2.1,
            "correct": True,
            "iterations": 3,
            "problem": "test problem",
            "backend": "triton",
        })
        updated = evo._strategies[sid]
        check(
            "update_strategy records success",
            len(updated.success_history) == 1,
            f"successes={len(updated.success_history)}",
        )
        check(
            "Pareto scores recomputed after update",
            updated.pareto_scores.get("avg_speedup", 0) > 0,
            f"avg_speedup={updated.pareto_scores.get('avg_speedup')}",
        )

    # Print loaded strategies
    print(f"\n  All loaded strategies:")
    for s in all_strats:
        print(f"    - {s.description[:70]}... (id={s.id})")


# ===================================================================
# Test 5: IntentTree standalone
# ===================================================================
def test_intent_tree_standalone() -> None:
    section("Test 5: IntentTree Standalone")

    tree = IntentTree(root_description="Optimize GEMM for H100")

    # Add child intents
    child1 = tree.add_node(tree.root.id, "Vectorize loads with float4", priority=0.8)
    child2 = tree.add_node(tree.root.id, "Apply shared memory tiling", priority=0.6)
    child3 = tree.add_node(child1.id, "Combine vectorization with warp shuffle", priority=0.7)

    check(
        "Tree has 4 nodes (root + 3 children)",
        len(tree.all_nodes) == 4,
        f"nodes={len(tree.all_nodes)}",
    )

    # Priority selection
    top = tree.get_highest_priority_pending()
    check(
        "Highest priority pending is root (priority=1.0)",
        top is not None and top.id == tree.root.id,
        f"top={top.description[:40] if top else 'None'}",
    )

    # Serialize / deserialize round-trip
    data = tree.serialize()
    tree2 = IntentTree.deserialize(data)
    check(
        "Serialize/deserialize round-trip preserves node count",
        len(tree2.all_nodes) == len(tree.all_nodes),
        f"original={len(tree.all_nodes)}, restored={len(tree2.all_nodes)}",
    )

    # Stagnation detection
    check(
        "No stagnation initially",
        not tree.stagnation_detected(threshold=3),
    )

    # Summary JSON for LLM context
    summary = tree.to_summary_json()
    check(
        "to_summary_json() has expected keys",
        "root_id" in summary and "nodes" in summary and "total_nodes" in summary,
    )


# ===================================================================
# Test 6: create_engine imports work (smoke test)
# ===================================================================
def test_create_engine_smoke() -> None:
    section("Test 6: create_engine() Smoke Test")

    try:
        from openkernel.engine.factory import create_engine
        from openkernel.llm.provider import LLMProvider
        from tests.mocks import MockEvalFn

        config = OpenKernelConfig(max_iterations=2)
        # Pass a mock eval_fn so the factory doesn't try to set up Modal
        mock_eval = MockEvalFn(seed=42)
        engine = create_engine(config=config, eval_fn=mock_eval)
        check(
            "create_engine() returns Orchestrator",
            isinstance(engine, Orchestrator),
            f"got {type(engine).__name__}",
        )
        # The engine should now use the real LLMProvider for tree operations
        check(
            "Engine uses real LLMProvider (not a mock)",
            isinstance(engine._llm, LLMProvider),
            f"got {type(engine._llm).__name__}",
        )
    except Exception as exc:
        check(
            "create_engine() imports and constructs without error",
            False,
            f"{type(exc).__name__}: {exc}",
        )


# ===================================================================
# Main
# ===================================================================
def main() -> None:
    global _passed, _failed
    print("\n" + "#" * 60)
    print("#  openkernel End-to-End System Test")
    print("#  MockInnerLoop + MockLLMCaller (no GPU, no API keys)")
    print("#" * 60)

    tests = [
        ("Full Optimization Loop", test_full_optimization_loop),
        ("Trace Capture", test_trace_capture),
        ("Skill Library", test_skill_library),
        ("Strategy Evolution", test_strategy_evolution),
        ("IntentTree Standalone", test_intent_tree_standalone),
        ("create_engine Smoke", test_create_engine_smoke),
    ]

    for name, fn in tests:
        try:
            fn()
        except Exception:
            _failed += 1
            print(f"\n  [FAIL] {name} raised an unhandled exception:")
            traceback.print_exc()

    # Final summary
    total = _passed + _failed
    print(f"\n{'=' * 60}")
    print(f"  RESULTS: {_passed}/{total} checks passed, {_failed} failed")
    print(f"{'=' * 60}\n")

    if _failed > 0:
        sys.exit(1)
    else:
        print("  All checks passed. The 3-level hybrid engine is working.\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
