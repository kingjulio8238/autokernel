"""Problem classifier for kernel optimization strategy selection.

Detects the operation type from PyTorch reference code to enable
per-type strategy templates. Inspired by Cursor/NVIDIA's SOL-ExecBench
problem taxonomy: L1 (simple), L2 (complex), Quant, FlashInfer.

Usage::

    from kernel_code.problem_classifier import classify_problem, ProblemType

    ptype = classify_problem(reference_code)
    print(ptype.tier)       # "L1"
    print(ptype.op_type)    # "reduction"
    print(ptype.strategy_hints)  # ["Use warp-level shuffle reduction", ...]
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class ProblemTier(Enum):
    """Problem difficulty tier (following SOL-ExecBench taxonomy)."""
    L1 = "L1"          # Simple: element-wise, reductions, norms, activations
    L2 = "L2"          # Complex: attention, MLP, multi-op fusion
    QUANT = "Quant"    # Quantized operations (FP8, FP4, INT8)
    MOE = "MoE"        # Mixture-of-Experts layers


class OpType(Enum):
    """Fine-grained operation type."""
    ELEMENTWISE = "elementwise"       # add, mul, relu, gelu, silu
    REDUCTION = "reduction"           # sum, mean, max, softmax
    NORM = "norm"                     # layernorm, rmsnorm, batchnorm, groupnorm
    GEMM = "gemm"                     # matmul, linear, mm, bmm
    ATTENTION = "attention"           # scaled_dot_product, multi_head, GQA
    CONV = "conv"                     # conv1d, conv2d, conv3d
    FUSED = "fused"                   # multi-op patterns (GEGLU, SwiGLU, AdaLN)
    MOE = "moe"                       # mixture of experts, expert routing
    QUANTIZED = "quantized"           # FP8, FP4, INT8, mixed precision
    EMBEDDING = "embedding"           # embedding lookup, positional encoding, RoPE
    POOLING = "pooling"               # avg_pool, max_pool, adaptive_pool
    CUSTOM = "custom"                 # unclassified


@dataclass
class ProblemClassification:
    """Result of classifying a kernel optimization problem."""
    tier: ProblemTier
    op_type: OpType
    strategy_hints: list[str] = field(default_factory=list)
    detected_patterns: list[str] = field(default_factory=list)
    is_memory_bound_likely: bool = False
    is_compute_bound_likely: bool = False
    is_launch_bound_likely: bool = False  # True for tiny tensors where kernel launch dominates
    estimated_tensor_elements: int = 0  # Best-effort estimate from code
    recommended_block_size: int | None = None

    def to_context_string(self) -> str:
        """Format for injection into optimization prompts."""
        parts = [
            f"## Problem Classification",
            f"**Tier**: {self.tier.value} | **Type**: {self.op_type.value}",
        ]
        if self.detected_patterns:
            parts.append(f"**Detected patterns**: {', '.join(self.detected_patterns)}")
        if self.is_launch_bound_likely:
            bound = "launch-overhead-bound (tiny kernel, PyTorch near-optimal)"
        elif self.is_memory_bound_likely:
            bound = "memory-bound"
        elif self.is_compute_bound_likely:
            bound = "compute-bound"
        else:
            bound = "unknown"
        parts.append(f"**Likely bottleneck**: {bound}")
        if self.estimated_tensor_elements > 0:
            parts.append(f"**Tensor size estimate**: ~{self.estimated_tensor_elements:,} elements")
        if self.strategy_hints:
            hints = "\n".join(f"  - {h}" for h in self.strategy_hints)
            parts.append(f"**Strategy hints**:\n{hints}")
        return "\n".join(parts)


# ── Detection Patterns ─────────────────────────────────────────────

_GEMM_PATTERNS = [
    r"torch\.mm\b", r"torch\.matmul\b", r"torch\.bmm\b",
    r"F\.linear\b", r"nn\.Linear\b",
    r"torch\.einsum\(['\"].*[ij].*[jk].*[ik]",  # matmul-like einsum
    r"(?<=\w)\s*@\s*(?=\w)",  # Python @ operator for matmul (requires word chars on both sides, excludes decorators)
]

_ATTENTION_PATTERNS = [
    r"scaled_dot_product_attention", r"multi_head_attention",
    r"softmax\(.*\/.*sqrt", r"attn_weights",
    r"query.*key.*value", r"q_proj.*k_proj.*v_proj",
    r"GroupedQueryAttention", r"MultiHeadAttention",
    r"\.attention\b",
]

_NORM_PATTERNS = [
    r"nn\.LayerNorm\b", r"F\.layer_norm\b", r"layer_norm\b",
    r"nn\.RMSNorm\b", r"rms_norm",
    r"nn\.BatchNorm", r"F\.batch_norm\b", r"batch_norm\b",
    r"nn\.GroupNorm\b", r"F\.group_norm\b", r"group_norm\b",
    r"nn\.InstanceNorm", r"F\.instance_norm\b", r"instance_norm\b",
    r"torch\.nn\.functional\.layer_norm",
    r"torch\.nn\.functional\.batch_norm",
]

_REDUCTION_PATTERNS = [
    r"\.sum\(", r"\.mean\(", r"\.max\(", r"\.min\(",
    r"F\.softmax\b", r"torch\.softmax\b",
    r"\.argmax\(", r"\.argmin\(",
    r"torch\.cumsum\b", r"torch\.cumprod\b",
]

_ELEMENTWISE_PATTERNS = [
    r"F\.relu\b", r"F\.gelu\b", r"F\.silu\b", r"F\.sigmoid\b",
    r"torch\.relu\b", r"torch\.sigmoid\b", r"torch\.tanh\b",
    r"F\.dropout\b", r"torch\.clamp\b",
    r"torch\.add\b", r"torch\.mul\b", r"torch\.exp\b",
    # Raw tensor arithmetic — require tensor-context anchors to avoid
    # false positives on scalar arithmetic / function args.
    # Anchor 1: in-place assignment to an indexed output
    r"output\[\.\.\.\]\s*=\s*\w+\s*[+\-*/]\s*\w+",
    r"\w+\[\.\.\.\]\s*=\s*\w+\s*[+\-*/]\s*\w+",
    # Anchor 2: explicit torch tensor operators
    r"\.add_?\(", r"\.mul_?\(", r"\.sub_?\(", r"\.div_?\(",
    # Anchor 3: common elementwise kernel names
    r"vector[_\s]*add|vectoradd|saxpy",
    r"\belementwise\b|\belement-wise\b",
    r"\badd_kernel\b|\bmul_kernel\b",
    # Anchor 4: tensor-context return (tensor-ish variable names)
    r"return\s+\w*_?tensor\w*\s*[+\-*/]",
    r"return\s+\w*out(?:put)?\w*\s*[+\-*/]",
]

_FUSED_PATTERNS = [
    r"geglu|GEGLU", r"swiglu|SwiGLU",
    r"adaln|AdaLN|adaptive_layer_norm",
    r"fused_add_rms",
    r"gate.*up.*down|up.*gate.*down",  # MLP gate/up/down fusion
    r"residual.*norm|norm.*residual",  # fused residual + norm
]

_CONV_PATTERNS = [
    r"nn\.Conv[123]d\b", r"F\.conv[123]d\b",
    r"nn\.ConvTranspose", r"F\.conv_transpose",
    r"depthwise_conv", r"pointwise_conv",
]

_MOE_PATTERNS = [
    r"MoE|mixture.of.experts", r"expert.*routing|routing.*expert",
    r"top_k.*expert|expert.*top_k", r"gating.*network",
    r"MegaBlocks|megablocks", r"scatter.*expert|gather.*expert",
    r"num_experts", r"expert_capacity",
]

_QUANT_PATTERNS = [
    r"\bfloat8\b|\bfp8\b|\bFP8\b", r"\bfloat4\b|\bfp4\b|\bFP4\b",
    r"\bint8\b|\bINT8\b", r"quantize|dequantize",
    r"\bscale_factor\b|\bblock_scale\b", r"\bMXFP[468]\b",
    r"\bper_tensor_scale\b|\bper_channel_scale\b",
    r"\bfake_quant\b", r"\bdynamic_quant\b",
]

_HISTOGRAM_PATTERNS = [
    r"\bbincount\b", r"\bhistc\b", r"\bscatter_add\b",
    r"\bindex_add\b", r"atomic_add",
]


_EMBEDDING_PATTERNS = [
    r"nn\.Embedding\b", r"F\.embedding\b",
    r"rotary_pos|RoPE|rope", r"positional_encod",
    r"sinusoidal_position", r"alibi",
]

_POOLING_PATTERNS = [
    r"F\.avg_pool", r"F\.max_pool", r"F\.adaptive_avg_pool",
    r"nn\.AvgPool", r"nn\.MaxPool", r"nn\.AdaptiveAvgPool",
]


def classify_problem(reference_code: str) -> ProblemClassification:
    """Classify a kernel optimization problem from its reference code.

    Analyzes PyTorch reference code to detect the operation type,
    problem tier, and generate strategy hints.

    Args:
        reference_code: PyTorch reference implementation source code.

    Returns:
        ProblemClassification with tier, op_type, and strategy hints.
    """
    code = reference_code

    # Count pattern matches for each category
    scores: dict[OpType, int] = {}
    detected: dict[OpType, list[str]] = {}
    histogram_hit = any(re.search(pat, code, re.IGNORECASE) for pat in _HISTOGRAM_PATTERNS)

    for op_type, patterns in [
        (OpType.GEMM, _GEMM_PATTERNS),
        (OpType.ATTENTION, _ATTENTION_PATTERNS),
        (OpType.NORM, _NORM_PATTERNS),
        (OpType.REDUCTION, _REDUCTION_PATTERNS),
        (OpType.ELEMENTWISE, _ELEMENTWISE_PATTERNS),
        (OpType.FUSED, _FUSED_PATTERNS),
        (OpType.CONV, _CONV_PATTERNS),
        (OpType.MOE, _MOE_PATTERNS),
        (OpType.QUANTIZED, _QUANT_PATTERNS),
        (OpType.EMBEDDING, _EMBEDDING_PATTERNS),
        (OpType.POOLING, _POOLING_PATTERNS),
    ]:
        score = 0
        found = []
        for pat in patterns:
            matches = re.findall(pat, code, re.IGNORECASE)
            if matches:
                score += len(matches)
                found.append(pat.split(r"\b")[0].replace("\\", "").strip(r".*|()"))
        if score > 0:
            scores[op_type] = score
            detected[op_type] = found

    if not scores and histogram_hit:
        return ProblemClassification(
            tier=ProblemTier.L1,
            op_type=OpType.REDUCTION,
            strategy_hints=[
                "Histogram / scatter-add: atomic contention is the bottleneck",
                "Use per-block shared-memory histograms, then reduce to global",
                "Privatize bins per warp/CTA to avoid atomicAdd serialization",
                "On high-contention inputs (>50% same value), consider warp-aggregated atomics",
            ],
            detected_patterns=["bincount/scatter-add pattern"],
            is_memory_bound_likely=True,
        )

    if not scores:
        return ProblemClassification(
            tier=ProblemTier.L1,
            op_type=OpType.CUSTOM,
            strategy_hints=["Profile first to identify bottleneck"],
            detected_patterns=[],
        )

    # Histogram hit overrides primary classification — it's the real workload
    if histogram_hit:
        scores[OpType.REDUCTION] = scores.get(OpType.REDUCTION, 0) + 5

    # Primary op type = highest scoring
    primary_op = max(scores, key=scores.get)
    all_detected = []
    for patterns in detected.values():
        all_detected.extend(patterns)

    # Determine tier
    tier = _classify_tier(primary_op, scores)

    # Determine likely bottleneck
    is_memory = primary_op in (
        OpType.ELEMENTWISE, OpType.NORM, OpType.REDUCTION,
        OpType.EMBEDDING, OpType.POOLING,
    )
    is_compute = primary_op in (
        OpType.GEMM, OpType.ATTENTION, OpType.CONV,
    )

    # Estimate tensor size from code (best effort)
    est_elements = _estimate_tensor_elements(code)

    # Launch-bound detection: tiny tensors + simple ops = kernel launch dominates
    # Threshold: < 1M elements AND elementwise/reduction/norm
    is_launch_bound = (
        est_elements > 0
        and est_elements < 1_000_000
        and primary_op in (OpType.ELEMENTWISE, OpType.REDUCTION, OpType.NORM)
    )

    # Generate strategy hints
    hints = _get_strategy_hints(primary_op, tier, scores)
    if is_launch_bound:
        hints.insert(0,
            f"! WARNING: ~{est_elements:,} elements is small — "
            f"kernel launch overhead (~10-50μs) may dominate actual work. "
            f"PyTorch fused elementwise kernels are often already near-optimal at this size."
        )
        hints.insert(1,
            "Consider grid-stride loops or multi-op fusion to amortize launch cost."
        )

    # Recommended block size
    block_size = 256 if is_memory else 128

    return ProblemClassification(
        tier=tier,
        op_type=primary_op,
        strategy_hints=hints,
        detected_patterns=all_detected[:10],  # Limit to top 10
        is_memory_bound_likely=is_memory or is_launch_bound,
        is_compute_bound_likely=is_compute and not is_launch_bound,
        is_launch_bound_likely=is_launch_bound,
        estimated_tensor_elements=est_elements,
        recommended_block_size=block_size,
    )


def _estimate_tensor_elements(code: str) -> int:
    """Best-effort estimate of total tensor element count from code.

    Looks for torch.randn(N, M, ...) or shape references.
    Returns 0 if nothing identifiable.
    """
    # Constant-propagate module-level int assignments (``M = 16384`` …) so
    # that ``torch.randn(M, N)`` is recognized as ``torch.randn(16384, N_value)``.
    # KernelBench and GPU Mode references universally declare tensor dims
    # this way; without the pass we'd return 0 for every real reference.
    const_env: dict[str, int] = {}
    for m in re.finditer(r"^\s*([A-Z_][A-Z0-9_]*)\s*=\s*(\d+)\s*$", code, flags=re.MULTILINE):
        try:
            const_env[m.group(1)] = int(m.group(2))
        except ValueError:
            continue
    if const_env:
        def _sub(match: re.Match) -> str:
            return str(const_env.get(match.group(0), match.group(0)))
        code = re.sub(r"\b[A-Z_][A-Z0-9_]*\b", _sub, code)

    # Match torch.randn(a, b, ...) etc. — capture the full dim list until
    # the first kwarg (dtype=, device=, generator=, ...) or closing paren.
    # The greedy [\d\s,]+ now captures all leading dim args.
    pattern = r"torch\.(?:randn|empty|zeros|ones|rand)\s*\(\s*([\d\s,]+?)(?=\s*(?:,\s*[a-z_]+\s*=|\)))"
    matches = re.findall(pattern, code)

    best_elements = 0
    for m in matches:
        try:
            dims = [int(d.strip()) for d in m.split(",") if d.strip().isdigit()]
            if dims:
                elements = 1
                for d in dims:
                    elements *= d
                if elements > best_elements:
                    best_elements = elements
        except (ValueError, OverflowError):
            continue

    # Also check for size= keyword patterns
    size_pattern = r"size\s*=\s*(\d+)"
    size_matches = re.findall(size_pattern, code)
    for s in size_matches:
        try:
            n = int(s)
            # Assume 2D tensor (common case): N x N
            if n * n > best_elements:
                best_elements = n * n
        except (ValueError, OverflowError):
            continue

    # If still no estimate AND the code uses symbolic `size, size` pattern
    # (common KernelBench/GPU Mode convention), assume typical small benchmark
    # size of 128 (16K elements). This flags as potentially launch-bound.
    if best_elements == 0:
        if re.search(r"torch\.\w+\s*\(\s*size\s*,\s*size", code):
            best_elements = 128 * 128  # typical default in KernelBench task.py

    return best_elements


def _classify_tier(primary_op: OpType, scores: dict[OpType, int]) -> ProblemTier:
    """Classify problem tier from operation type and pattern scores."""
    # MoE is its own tier
    if primary_op == OpType.MOE or scores.get(OpType.MOE, 0) > 0:
        return ProblemTier.MOE

    # Quantized ops are their own tier
    if primary_op == OpType.QUANTIZED or scores.get(OpType.QUANTIZED, 0) >= 2:
        return ProblemTier.QUANT

    # L2: Complex multi-op patterns
    if primary_op in (OpType.ATTENTION, OpType.FUSED):
        return ProblemTier.L2

    # L2: GEMM with attention or fused patterns
    if primary_op == OpType.GEMM and (
        scores.get(OpType.ATTENTION, 0) > 0 or scores.get(OpType.FUSED, 0) > 0
    ):
        return ProblemTier.L2

    # L2: Multiple complex ops combined
    complex_ops = sum(1 for op in [OpType.GEMM, OpType.CONV, OpType.ATTENTION]
                      if scores.get(op, 0) > 0)
    if complex_ops >= 2:
        return ProblemTier.L2

    # L1: Simple operations
    return ProblemTier.L1


def _get_strategy_hints(
    primary_op: OpType,
    tier: ProblemTier,
    scores: dict[OpType, int],
) -> list[str]:
    """Generate strategy hints based on problem classification."""
    hints: list[str] = []

    if primary_op == OpType.GEMM:
        hints.extend([
            "Profile for compute vs memory bound classification",
            "Try tiling with BLOCK_M/BLOCK_N/BLOCK_K autotune",
            "Consider tensor core utilization (check compute capability)",
            "For small M: consider warp-per-output pattern (Cursor warp decode)",
        ])
    elif primary_op == OpType.ATTENTION:
        hints.extend([
            "Consider FlashAttention-style tiling (block Q/K/V)",
            "Use online softmax to avoid materializing attention matrix",
            "Fuse Q*K^T and softmax*V into single kernel",
            "For GQA: broadcast K/V heads efficiently",
        ])
    elif primary_op == OpType.NORM:
        hints.extend([
            "Memory-bound — maximize bandwidth utilization",
            "Single-pass Welford algorithm for mean+variance",
            "Vectorized loads (float4 for fp32, __nv_bfloat162 for bf16)",
            "Fuse with adjacent operations (residual add, activation)",
        ])
    elif primary_op == OpType.REDUCTION:
        hints.extend([
            "Use warp-level shuffle reduction (__shfl_xor_sync)",
            "Tree reduction pattern across warps",
            "Consider online/streaming algorithm to avoid multiple passes",
        ])
    elif primary_op == OpType.ELEMENTWISE:
        hints.extend([
            "Memory-bound — focus on vectorized loads and coalesced access",
            "Fuse with adjacent element-wise operations",
            "Consider grid-stride loop pattern for large tensors",
        ])
    elif primary_op == OpType.FUSED:
        hints.extend([
            "Identify which ops to fuse (reduce memory round-trips)",
            "Keep intermediates in registers, not global memory",
            "For SwiGLU/GEGLU: fuse gate+up projection + activation",
        ])
    elif primary_op == OpType.MOE:
        hints.extend([
            "For training: grouped GEMM with expert-wise L2 supergrouping",
            "For small-batch decode: warp-per-output (flip parallelism axis)",
            "Eliminate padding/scatter/gather bookkeeping stages",
            "Fuse quantization into kernel prologues/epilogues",
        ])
    elif primary_op == OpType.QUANTIZED:
        hints.extend([
            "Fuse quantization with compute to avoid separate quant pass",
            "Quantization overhead can dominate (up to 76% of matmul time)",
            "Produce scale factors in hardware-compatible layout",
            "Use hardware block-scaled MMA if available (Blackwell tcgen05)",
        ])
    elif primary_op == OpType.CONV:
        hints.extend([
            "Consider im2col + GEMM transformation",
            "Use shared memory tiling for input feature map reuse",
            "For depthwise: each channel is independent (embarrassingly parallel)",
        ])
    elif primary_op == OpType.EMBEDDING:
        hints.extend([
            "Memory-bound — optimize for random access patterns",
            "For RoPE: precompute sin/cos tables, fuse with attention",
            "Coalesced reads across embedding dimension",
        ])

    # Add tier-specific hints
    if tier == ProblemTier.L2:
        hints.append("Complex kernel — consider multi-pass optimization strategy")
    elif tier == ProblemTier.QUANT:
        hints.append("Quantized op — validate numerical accuracy carefully (rtol may need adjustment)")
    elif tier == ProblemTier.MOE:
        hints.append("MoE — detect workload type (training vs decode) to select parallelism axis")

    return hints
