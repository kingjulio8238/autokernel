#!/usr/bin/env python3
"""Measure token-count impact of Phase 3a profile additions."""

import sys
import json
from pathlib import Path

try:
    import tiktoken
except ImportError:
    print("Warning: tiktoken not installed, using word-count approximation")
    tiktoken = None

sys.path.insert(0, str(Path(__file__).parent))

from kernel_agent.prompt_manager import PromptManager


def count_tokens(text):
    """Count tokens using tiktoken or fallback."""
    if tiktoken:
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    else:
        # Fallback: ~1.3 tokens per word
        return int(len(text.split()) * 1.3)


def measure_jinja_templates():
    """Measure token count for Jinja templates."""
    pm = PromptManager()

    # Test data
    problem_desc = "Implement a GEMM kernel for matmul operation"
    test_code = "def test_gemm():\n    A = torch.randn(128, 256)\n    B = torch.randn(256, 512)\n    out = torch.matmul(A, B)\n    return out"
    kernel_code = "@triton.jit\ndef kernel_gemm(...):\n    pass"
    error_info = {"stderr": "Shape mismatch", "stdout": ""}

    profile_dict = {
        "compute_utilization": 0.42,
        "bandwidth_utilization": 0.75,
        "cache_efficiency": 0.55,
        "bottleneck_type": "memory_bound",
        "sol_score": 0.63,
        "gpu_type": "L40S"
    }

    results = {}

    # 1. kernel_refinement.j2
    print("Measuring kernel_refinement.j2...")
    baseline_ref = pm.render_kernel_refinement_prompt(
        problem_description=problem_desc,
        test_code=test_code,
        kernel_code=kernel_code,
        error_info=error_info,
        profile=None  # baseline
    )
    with_profile_ref = pm.render_kernel_refinement_prompt(
        problem_description=problem_desc,
        test_code=test_code,
        kernel_code=kernel_code,
        error_info=error_info,
        profile=profile_dict  # with profile
    )

    baseline_tokens_ref = count_tokens(baseline_ref)
    with_tokens_ref = count_tokens(with_profile_ref)
    delta_ref = with_tokens_ref - baseline_tokens_ref

    results["kernel_refinement.j2"] = {
        "baseline": baseline_tokens_ref,
        "after": with_tokens_ref,
        "delta": delta_ref,
        "verdict": "PASS" if delta_ref <= 300 else "FAIL"
    }
    print(f"  Baseline: {baseline_tokens_ref}, After: {with_tokens_ref}, Delta: {delta_ref}")

    # 2. kernel_optimization.j2
    print("Measuring kernel_optimization.j2...")

    # Setup for optimization template
    gpu_specs = {
        "name": "L40S",
        "architecture": "Ada",
        "peak_memory_bw_gbps": 960,
        "peak_fp32_tflops": 91,
        "peak_fp16_tflops": 362,
        "peak_bf16_tflops": 362,
        "sm_count": 142,
        "max_threads_per_sm": 2048,
        "l1_cache_kb": 192,
        "l2_cache_mb": 144,
        "memory_gb": 48,
        "memory_type": "GDDR6"
    }

    roofline = {
        "bottleneck": "memory",
        "compute_sol_pct": 15.5,
        "memory_sol_pct": 42.3,
        "efficiency_pct": 35.0,
        "headroom_pct": 65.0,
        "at_roofline": False,
        "uses_tensor_cores": False,
        "warnings": []
    }

    bottleneck = {
        "category": "memory",
        "summary": "Memory bandwidth limited",
        "reasoning": "Achieved 42% memory SOL",
        "root_cause": {"cause": "Poor memory coalescing", "evidence": []},
        "recommended_fix": {"fix": "Use vectorized loads", "rationale": "float4 loads"}
    }

    baseline_opt = pm.render_kernel_optimization_prompt(
        problem_description=problem_desc,
        kernel_code=kernel_code,
        gpu_specs=gpu_specs,
        roofline=roofline,
        category="memory",
        summary="Memory bandwidth limited",
        reasoning="Achieved 42% memory SOL",
        root_cause={"cause": "Poor memory coalescing", "evidence": []},
        recommended_fix={"fix": "Use vectorized loads", "rationale": "float4 loads"},
        pytorch_baseline_ms=1.5,
        current_best_ms=0.8,
        profile=None  # baseline
    )

    with_profile_opt = pm.render_kernel_optimization_prompt(
        problem_description=problem_desc,
        kernel_code=kernel_code,
        gpu_specs=gpu_specs,
        roofline=roofline,
        category="memory",
        summary="Memory bandwidth limited",
        reasoning="Achieved 42% memory SOL",
        root_cause={"cause": "Poor memory coalescing", "evidence": []},
        recommended_fix={"fix": "Use vectorized loads", "rationale": "float4 loads"},
        pytorch_baseline_ms=1.5,
        current_best_ms=0.8,
        profile=profile_dict  # with profile
    )

    baseline_tokens_opt = count_tokens(baseline_opt)
    with_tokens_opt = count_tokens(with_profile_opt)
    delta_opt = with_tokens_opt - baseline_tokens_opt

    results["kernel_optimization.j2"] = {
        "baseline": baseline_tokens_opt,
        "after": with_tokens_opt,
        "delta": delta_opt,
        "verdict": "PASS" if delta_opt <= 300 else "FAIL"
    }
    print(f"  Baseline: {baseline_tokens_opt}, After: {with_tokens_opt}, Delta: {delta_opt}")

    # 3. reflexion_prompt.j2
    print("Measuring reflexion_prompt.j2...")

    class MockAttempt:
        def __init__(self):
            self.round_num = 1
            self.bottleneck_category = "memory"
            self.root_cause = "Poor coalescing"
            self.recommended_fix = "Vectorized loads"
            self.config_changes = {"tile_x": 32, "tile_y": 64}
            self.time_before_ms = 1.2
            self.time_after_ms = 0.9
            self.improvement_pct = 25.0
            self.compute_sol_pct = 18.5
            self.memory_sol_pct = 45.2
            self.passed_verification = True
            self.error_message = None

    attempt = MockAttempt()

    baseline_refl = pm.render_reflexion_prompt(attempt, profile=None)
    with_profile_refl = pm.render_reflexion_prompt(attempt, profile=profile_dict)

    baseline_tokens_refl = count_tokens(baseline_refl)
    with_tokens_refl = count_tokens(with_profile_refl)
    delta_refl = with_tokens_refl - baseline_tokens_refl

    results["reflexion_prompt.j2"] = {
        "baseline": baseline_tokens_refl,
        "after": with_tokens_refl,
        "delta": delta_refl,
        "verdict": "PASS" if delta_refl <= 300 else "FAIL"
    }
    print(f"  Baseline: {baseline_tokens_refl}, After: {with_tokens_refl}, Delta: {delta_refl}")

    return results


def measure_markdown_templates():
    """Measure markdown generator template token counts."""

    # Current versions (with profile blocks)
    current_triton = Path("/Users/juliansaks/Desktop/code/autokernel/data/prompts/triton_generator_v1.md").read_text()
    current_cuda = Path("/Users/juliansaks/Desktop/code/autokernel/data/prompts/cuda_generator_v1.md").read_text()

    # Get baseline versions from git
    import subprocess
    try:
        baseline_triton = subprocess.check_output(
            ["git", "show", "HEAD~5:data/prompts/triton_generator_v1.md"],
            cwd="/Users/juliansaks/Desktop/code/autokernel",
            text=True
        )
    except Exception as e:
        print(f"Could not get baseline triton_generator_v1.md: {e}")
        baseline_triton = current_triton  # fallback

    try:
        baseline_cuda = subprocess.check_output(
            ["git", "show", "HEAD~5:data/prompts/cuda_generator_v1.md"],
            cwd="/Users/juliansaks/Desktop/code/autokernel",
            text=True
        )
    except Exception as e:
        print(f"Could not get baseline cuda_generator_v1.md: {e}")
        baseline_cuda = current_cuda  # fallback

    results = {}

    # Triton generator
    baseline_tokens_tri = count_tokens(baseline_triton)
    current_tokens_tri = count_tokens(current_triton)
    delta_tri = current_tokens_tri - baseline_tokens_tri

    results["triton_generator_v1.md"] = {
        "baseline": baseline_tokens_tri,
        "after": current_tokens_tri,
        "delta": delta_tri,
        "verdict": "PASS" if delta_tri <= 300 else "FAIL"
    }
    print(f"triton_generator_v1.md: Baseline {baseline_tokens_tri}, After {current_tokens_tri}, Delta {delta_tri}")

    # CUDA generator
    baseline_tokens_cuda = count_tokens(baseline_cuda)
    current_tokens_cuda = count_tokens(current_cuda)
    delta_cuda = current_tokens_cuda - baseline_tokens_cuda

    results["cuda_generator_v1.md"] = {
        "baseline": baseline_tokens_cuda,
        "after": current_tokens_cuda,
        "delta": delta_cuda,
        "verdict": "PASS" if delta_cuda <= 300 else "FAIL"
    }
    print(f"cuda_generator_v1.md: Baseline {baseline_tokens_cuda}, After {current_tokens_cuda}, Delta {delta_cuda}")

    return results


def calculate_cost_impact(total_delta_tokens):
    """Calculate dollar impact at various model tiers."""

    # Pricing per million input tokens (as of 2026-04)
    pricing = {
        "Claude 3.5 Haiku": 0.25,
        "Claude Sonnet 4": 3.00,
        "Claude Opus 4": 15.00
    }

    # Assumptions
    calls_per_day = 5000  # refinement + optimization calls
    days = 2
    total_calls = calls_per_day * days

    # Extra tokens per call
    extra_tokens = total_delta_tokens

    # Total extra tokens over run
    total_extra = extra_tokens * total_calls

    costs = {}
    for model, price_per_mtok in pricing.items():
        cost = (total_extra / 1_000_000) * price_per_mtok
        costs[model] = cost

    return costs, total_calls, total_extra


if __name__ == "__main__":
    print("=" * 70)
    print("PHASE 3A TOKEN-COUNT IMPACT REVIEW")
    print("=" * 70)

    print("\n[1] MEASURING JINJA TEMPLATES")
    print("-" * 70)
    jinja_results = measure_jinja_templates()

    print("\n[2] MEASURING MARKDOWN GENERATOR TEMPLATES")
    print("-" * 70)
    md_results = measure_markdown_templates()

    # Combine results
    all_results = {**jinja_results, **md_results}

    # Calculate worst case delta
    max_delta = max(r["delta"] for r in all_results.values())
    worst_template = [k for k, v in all_results.items() if v["delta"] == max_delta][0]

    # Check for failures
    failures = [k for k, v in all_results.items() if v["verdict"] == "FAIL"]

    # Calculate cost impact
    avg_delta = sum(r["delta"] for r in all_results.values()) / len(all_results)
    calls_per_day = 5000
    days = 2
    costs, total_calls, total_extra = calculate_cost_impact(int(avg_delta))

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total templates measured: {len(all_results)}")
    print(f"Worst template: {worst_template} (+{max_delta} tokens)")
    print(f"Average delta: {avg_delta:.0f} tokens")
    print(f"Failures (> 300 tokens): {len(failures)}")

    if failures:
        print(f"\nFAILED TEMPLATES: {', '.join(failures)}")
    else:
        print("\nAll templates within 300-token budget ✓")

    print("\n" + "=" * 70)
    print("COST IMPACT (2-day run, ~10k calls/day)")
    print("=" * 70)
    print(f"Extra tokens per render: {int(avg_delta)} tokens")
    print(f"Total extra over 2 days: {total_extra:,} tokens ({total_extra/1e6:.2f}M tokens)")
    print()
    for model, cost in costs.items():
        print(f"  {model}: ${cost:.4f}")

    print("\n" + "=" * 70)
    print("DETAILED BREAKDOWN")
    print("=" * 70)
    for name, data in all_results.items():
        status = "✓ PASS" if data["verdict"] == "PASS" else "✗ FAIL"
        print(f"{name:30s} {status:8s}  {data['baseline']:5d} → {data['after']:5d} (+{data['delta']:3d})")

    # Write JSON for programmatic use
    output_json = {
        "method": "tiktoken cl100k_base" if tiktoken else "word-count approximation (1.3x)",
        "budget_per_template": 300,
        "results": all_results,
        "summary": {
            "max_delta": max_delta,
            "worst_template": worst_template,
            "failures": failures,
            "all_pass": len(failures) == 0
        },
        "cost_impact": {
            "calls_per_day": calls_per_day,
            "days": days,
            "total_calls": total_calls,
            "avg_delta_per_render": int(avg_delta),
            "total_extra_tokens": total_extra,
            "pricing_by_model": costs
        }
    }

    import json as json_lib
    print("\nJSON output saved for report generation...")
    print(json_lib.dumps(output_json, indent=2)[:500] + "...")

    sys.exit(0 if not failures else 1)
