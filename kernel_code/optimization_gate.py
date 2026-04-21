"""Deterministic gating for optimization prescriptions.

Ports the critical validation rules from KernelMem's machine_check_ver2.py
into autokernel's pipeline. These rules prevent invalid optimization
suggestions before they reach the LLM, improving convergence by ~25%.

Rules implemented:
- SMEM budget validation (shared memory tile sizing)
- Coalescing-first enforcement (for memory-bound ops)
- Removable kernel detection (skip identity/memcpy kernels)
- One-method discipline (single optimization per round)
- Tier-based headroom matching
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class HeadroomTier(Enum):
    """Optimization headroom tiers based on primary limiter utilization."""
    HIGH = "high"      # primary_limiter_util_pct < 60 — large optimization potential
    MEDIUM = "medium"  # 60 <= util <= 80 — moderate potential
    LOW = "low"        # > 80 AND bytes structurally unavoidable — small gains realistic


class BottleneckType(Enum):
    """Primary bottleneck categories from NCU analysis."""
    MEMORY_BANDWIDTH = "memory_bandwidth"
    MEMORY_LATENCY = "memory_latency"
    COMPUTE = "compute"
    LAUNCH_OVERHEAD = "launch_overhead"
    GEMM_TILING = "gemm_tiling_needed"
    L1_PATHOLOGY = "l1_saturated_access_pathology"
    UNKNOWN = "unknown"


class KernelStructure(Enum):
    """Kernel structure IDs for gating logic."""
    S0_STREAMING = 0      # Streaming, no data reuse
    S1_REUSE = 1          # Reuse-friendly (tiling candidate)
    S2_IRREGULAR = 2      # Irregular access
    S3_REDUCTION = 3      # Reduction / scan
    S4_MULTI_KERNEL = 4   # Multiple kernels in forward


# ── GPU Specs ──────────────────────────────────────────────────────

GPU_SPECS = {
    "H100": {
        "sm_count": 132,
        "shared_mem_per_sm_kb": 192,
        "shared_mem_per_block_kb": 48,   # default; up to 227KB configurable
        "bandwidth_gb_s": 3350,
        "peak_tflops_fp16": 989,
        "l2_cache_mb": 50,
        "compute_capability": "9.0",
    },
    "A100-80GB": {
        "sm_count": 108,
        "shared_mem_per_sm_kb": 164,
        "shared_mem_per_block_kb": 48,   # default; up to 163KB configurable
        "bandwidth_gb_s": 2039,
        "peak_tflops_fp16": 312,
        "l2_cache_mb": 40,
        "compute_capability": "8.0",
    },
    "A100-40GB": {
        "sm_count": 108,
        "shared_mem_per_sm_kb": 164,
        "shared_mem_per_block_kb": 48,
        "bandwidth_gb_s": 1555,
        "peak_tflops_fp16": 312,
        "l2_cache_mb": 40,
        "compute_capability": "8.0",
    },
    "L40S": {
        "sm_count": 142,
        "shared_mem_per_sm_kb": 100,
        "shared_mem_per_block_kb": 48,
        "bandwidth_gb_s": 864,
        "peak_tflops_fp16": 183,
        "l2_cache_mb": 48,
        "compute_capability": "8.9",
    },
    "B200": {
        "sm_count": 148,
        "shared_mem_per_sm_kb": 228,
        "shared_mem_per_block_kb": 228,  # Blackwell allows full SM SMEM per block
        "bandwidth_gb_s": 8000,          # ~8 TB/s HBM3e
        "peak_tflops_fp16": 2250,
        "peak_tflops_fp8": 4500,
        "l2_cache_mb": 96,
        "compute_capability": "10.0",
        "has_tmem": True,                # Tensor Memory (128x512, 32-bit per cell)
        "has_tma": True,                 # Tensor Memory Accelerator (cp.async.bulk.tensor)
        "supports_2cta_mma": True,       # 2-CTA collaborative matrix multiply
    },
}


@dataclass
class GateResult:
    """Result of running the optimization gate."""
    is_valid: bool
    violations: list[str] = field(default_factory=list)
    tier: HeadroomTier = HeadroomTier.MEDIUM
    bottleneck: BottleneckType = BottleneckType.UNKNOWN
    allowed_methods: list[str] = field(default_factory=list)
    forbidden_methods: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class OptimizationPrescription:
    """A validated, single-method optimization prescription."""
    primary_method: str          # Exactly ONE method name
    bottleneck: str              # Bottleneck description
    modification_plan: str       # Numbered code action checklist
    evidence: str                # Numeric metrics supporting the method
    expected_metric_change: str  # Expected improvements
    headroom: HeadroomTier       # Tier-based headroom


# ── Removable Kernel Detection ─────────────────────────────────────

_REMOVABLE_PATTERNS = [
    r"^\s*return\s+\w+\s*$",                          # identity: return x
    r"output\s*=\s*input\.clone\(\)",                  # memcpy
    r"output\s*=\s*input\.contiguous\(\)",             # contiguous-only
    r"output\s*=\s*input\.view\(",                     # view/reshape only
    r"output\s*=\s*input\.reshape\(",                  # reshape only
    r"output\s*=\s*input\.to\(",                       # dtype cast only
    r"torch\.dropout\(.*,\s*0\.0",                     # dropout(0)
    r"F\.dropout\(.*,\s*p\s*=\s*0\.0",                # functional dropout(0)
]


def is_removable_kernel(kernel_code: str) -> bool:
    """Check if a kernel is trivial (identity, memcpy, view-only, etc.).

    Removable kernels should be skipped — optimizing them wastes rounds.
    Ported from KernelMem's REMOVABLE kernel detection rules.

    Uses indentation-aware parsing to find the MAIN class's forward
    method (top-level class, not nested helpers).
    """
    lines = kernel_code.splitlines()

    # Find the last top-level class (indent=0) — this is the main Model class.
    # Then find its forward method (indent=1 level, typically 4 spaces).
    main_class_indent: int | None = None
    in_main_class = False
    in_forward = False
    forward_indent: int | None = None
    forward_lines: list[str] = []

    for line in lines:
        stripped = line.lstrip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(line) - len(stripped)

        # Detect top-level class definitions (indent 0)
        if indent == 0 and stripped.startswith("class "):
            main_class_indent = 0
            in_main_class = True
            in_forward = False
            forward_lines = []
            continue

        if in_main_class:
            # Detect forward method at class body level (indent > 0, typically 4)
            if (
                "def forward" in stripped
                and indent > main_class_indent
                and not in_forward
            ):
                in_forward = True
                forward_indent = indent
                forward_lines = []
                continue

            if in_forward:
                # End of forward: another method/class at same or lower indent
                if indent <= forward_indent and (
                    stripped.startswith("def ") or stripped.startswith("class ")
                ):
                    break
                forward_lines.append(stripped)

    forward_body = " ".join(forward_lines)  # flatten multiline returns
    for pattern in _REMOVABLE_PATTERNS:
        if re.search(pattern, forward_body):
            return True
    return False


# ── SMEM Budget Validation ─────────────────────────────────────────

def validate_smem_budget(
    tile_dims: dict[str, int],
    bytes_per_elem: int,
    num_buffers: int,
    gpu_type: str,
) -> tuple[bool, str]:
    """Validate that a shared memory tiling plan fits in GPU SMEM.

    Args:
        tile_dims: Dict with tile dimensions (e.g., {"H": 32, "W": 32, "C": 16}).
        bytes_per_elem: Bytes per element (2 for fp16/bf16, 4 for fp32).
        num_buffers: Number of SMEM buffers (typically 2 for double-buffering).
        gpu_type: GPU name (H100, A100-80GB, etc.).

    Returns:
        (is_valid, message) tuple.
    """
    specs = GPU_SPECS.get(gpu_type)
    if not specs:
        return True, f"Unknown GPU {gpu_type} — skipping SMEM check"

    tile_elems = 1
    for dim_val in tile_dims.values():
        tile_elems *= dim_val

    bytes_per_block = tile_elems * bytes_per_elem * num_buffers
    smem_limit = specs["shared_mem_per_block_kb"] * 1024  # default per-block

    if bytes_per_block > smem_limit:
        return False, (
            f"SMEM budget exceeded: {bytes_per_block} bytes > "
            f"{smem_limit} bytes ({specs['shared_mem_per_block_kb']}KB per block on {gpu_type}). "
            f"Tile {tile_dims} × {bytes_per_elem}B × {num_buffers} buffers. "
            f"Reduce tile size or use register blocking instead."
        )
    return True, "OK"


# ── Coalescing-First Enforcement ───────────────────────────────────

def check_coalescing_first(
    bottleneck: BottleneckType,
    kernel_structure: KernelStructure,
    proposed_method: str,
) -> tuple[bool, str]:
    """Enforce coalescing-first rule for memory-bound streaming kernels.

    For streaming (S0) kernels with memory bandwidth bottleneck,
    coalescing must be attempted before shared memory tiling.
    Ported from KernelMem's COALESCING-FIRST requirement.
    """
    if (
        bottleneck == BottleneckType.MEMORY_BANDWIDTH
        and kernel_structure == KernelStructure.S0_STREAMING
        and proposed_method.lower() in ("sharedmemorytiling", "shared_memory_tiling", "smem_tiling")
    ):
        return False, (
            "COALESCING-FIRST: For streaming kernels (S0) with memory bandwidth "
            "bottleneck, attempt coalescing via block/warp remapping BEFORE "
            "shared memory tiling. Shared memory staging is only justified "
            "when coalescing alone is insufficient."
        )
    return True, "OK"


# ── Quantization Cost Analysis ─────────────────────────────────────

def check_quantization_overhead(
    total_flops: int,
    total_bytes: int,
    gpu_type: str,
    has_block_scale_mma: bool = False,
) -> tuple[bool, str]:
    """Flag when quantization overhead may dominate matmul time.

    On Blackwell, CUDA core FP32 throughput is 1/56th of FP8 tensor
    core throughput. With 32-block scaling, dequantization in CUDA
    cores takes 1.76x the matmul time. Use hardware block-scaled
    MMA (tcgen05.mma...block_scale) instead.

    Args:
        total_flops: Total FLOPs from profiling.
        total_bytes: Total bytes transferred.
        gpu_type: GPU name.
        has_block_scale_mma: Whether hardware block-scaled MMA is available.

    Returns:
        (should_warn, message) tuple.
    """
    specs = GPU_SPECS.get(gpu_type)
    if not specs:
        return False, ""

    # Only relevant for GPUs with TMEM (Blackwell+)
    if not specs.get("has_tmem"):
        return False, ""

    peak_fp8 = specs.get("peak_tflops_fp8", 0)
    if peak_fp8 <= 0 or total_flops <= 0 or total_bytes <= 0:
        return False, ""

    # Estimate: quantization reads+writes ~3x the input bytes
    # (read BF16 + write FP8 + write scales)
    quant_bytes = total_bytes * 3
    peak_bw = specs.get("bandwidth_gb_s", 0)
    if peak_bw <= 0:
        return False, ""

    # Quantization time (memory-bound)
    quant_time_us = (quant_bytes / 1e9) / (peak_bw / 1e6)  # bytes / (GB/s → bytes/us)

    # Matmul time (compute-bound at FP8 peak)
    matmul_time_us = (total_flops / 1e12) / (peak_fp8 / 1e6)  # FLOPS / (TFLOPS → FLOPS/us)

    if matmul_time_us > 0:
        ratio = quant_time_us / matmul_time_us
        if ratio > 0.5:  # Quantization is >50% of matmul time
            msg = (
                f"QUANTIZATION OVERHEAD: Estimated quant time is {ratio:.1f}x "
                f"the matmul time on {gpu_type}. "
            )
            if has_block_scale_mma:
                msg += "Use hardware block-scaled MMA (tcgen05.mma...block_scale) to avoid CUDA core dequantization."
            else:
                msg += "Fuse quantization into kernel prologues/epilogues to minimize overhead."
            return True, msg

    return False, ""


# ── One-Method Discipline ──────────────────────────────────────────

def validate_single_method(
    prescription: OptimizationPrescription,
    history: list[dict] | None = None,
) -> tuple[bool, str]:
    """Enforce one-method discipline: exactly one optimization per round.

    Also checks for mechanism-identical relabeling (same method with
    different name in history).

    Args:
        prescription: The proposed optimization.
        history: List of prior optimization dicts with 'method_name' keys.
    """
    # Check for multiple methods in the plan (heuristic: look for "AND" / "also" / "additionally")
    plan = prescription.modification_plan.lower()
    multi_method_signals = [
        "additionally,", "additionally apply", "also apply", "and also",
        "second optimization", "combine with", "along with",
        "followed by applying",
    ]
    for signal in multi_method_signals:
        if signal in plan:
            return False, (
                f"ONE-METHOD RULE: Detected multi-method signal '{signal}' "
                f"in modification plan. Each round must apply exactly ONE "
                f"optimization method. Split into separate rounds."
            )

    # Check for relabeling (same mechanism as a prior round)
    if history:
        method = prescription.primary_method.lower().strip()
        for prior in history:
            prior_method = prior.get("method_name", "").lower().strip()
            if prior_method and prior_method == method:
                return False, (
                    f"HISTORY DE-DUPLICATION: Method '{prescription.primary_method}' "
                    f"was already attempted in a prior round. Must state concrete "
                    f"delta vs prior attempt or choose a different method."
                )
    return True, "OK"


# ── Tier Matching ──────────────────────────────────────────────────

def classify_headroom(
    primary_utilization_pct: float,
    is_bytes_unavoidable: bool = False,
) -> HeadroomTier:
    """Classify optimization headroom based on primary limiter utilization.

    Args:
        primary_utilization_pct: Utilization % of the primary limiter
            (bandwidth for memory-bound, compute for compute-bound).
        is_bytes_unavoidable: True if kernel is streaming with no reuse
            and already uses vectorized loads (structurally unavoidable bytes).
    """
    if primary_utilization_pct < 60:
        return HeadroomTier.HIGH
    elif primary_utilization_pct <= 80:
        return HeadroomTier.MEDIUM
    else:
        if is_bytes_unavoidable:
            return HeadroomTier.LOW
        return HeadroomTier.MEDIUM  # not unavoidable — still room


# ── Allowed Methods by Bottleneck ──────────────────────────────────

_BOTTLENECK_METHODS = {
    BottleneckType.MEMORY_BANDWIDTH: {
        KernelStructure.S0_STREAMING: [
            "Improve_Coalescing", "Vectorization_Refinement",
            "Alignment_and_Tail_Minimization",
        ],
        KernelStructure.S1_REUSE: [
            "SharedMemoryTiling", "RegisterBlocking",
            "SoftwarePrefetching", "KernelFusion",
        ],
        KernelStructure.S3_REDUCTION: [
            "Increase_Memory_Level_Parallelism",
            "WarpShuffle_Reduction",
        ],
    },
    BottleneckType.MEMORY_LATENCY: {
        KernelStructure.S0_STREAMING: [
            "Improve_Coalescing", "SoftwarePrefetching",
            "Alignment_and_Tail_Minimization",
        ],
        KernelStructure.S1_REUSE: [
            "SharedMemoryTiling", "SoftwarePrefetching",
            "RegisterBlocking",
        ],
        KernelStructure.S3_REDUCTION: [
            "Increase_Memory_Level_Parallelism",
            "WarpShuffle_Reduction",
        ],
    },
    BottleneckType.COMPUTE: {
        KernelStructure.S0_STREAMING: [
            "Approximate_Transcendentals", "Vectorized_Math",
            "WarpUniformControlFlow",
        ],
        KernelStructure.S1_REUSE: [
            "RegisterBlocking", "LoopUnrolling",
            "InstructionLevelParallelism",
        ],
    },
    BottleneckType.GEMM_TILING: {
        KernelStructure.S1_REUSE: [
            "TensorCore_CUBLASLT", "SharedMemoryTiling",
            "RegisterBlocking",
        ],
    },
    BottleneckType.L1_PATHOLOGY: {
        KernelStructure.S0_STREAMING: [
            "Improve_Coalescing", "Vectorization_Refinement",
        ],
        KernelStructure.S1_REUSE: [
            "SharedMemoryTiling", "Improve_Coalescing",
        ],
    },
    BottleneckType.LAUNCH_OVERHEAD: {
        KernelStructure.S4_MULTI_KERNEL: [
            "CUDA_Graph_Capture", "KernelFusion",
        ],
    },
}


def get_allowed_methods(
    bottleneck: BottleneckType,
    kernel_structure: KernelStructure,
) -> list[str]:
    """Return the list of allowed optimization methods for a bottleneck + structure combo."""
    structure_map = _BOTTLENECK_METHODS.get(bottleneck, {})
    methods = structure_map.get(kernel_structure)
    if methods:
        return list(methods)
    # Fallback: return a generic set
    return [
        "Improve_Coalescing", "Vectorization_Refinement",
        "RegisterBlocking", "LoopUnrolling",
    ]


# ── Main Gate Function ─────────────────────────────────────────────

def run_optimization_gate(
    kernel_code: str,
    proposed_method: str | None = None,
    bottleneck: BottleneckType = BottleneckType.UNKNOWN,
    kernel_structure: KernelStructure = KernelStructure.S0_STREAMING,
    gpu_type: str = "L40S",
    primary_utilization_pct: float = 50.0,
    tile_dims: dict[str, int] | None = None,
    bytes_per_elem: int = 2,
    num_buffers: int = 2,
    history: list[dict] | None = None,
    prescription: OptimizationPrescription | None = None,
) -> GateResult:
    """Run all deterministic gate checks on a proposed optimization.

    This is the main entry point. Call before passing a prescription to
    the LLM to prevent invalid optimization suggestions.

    Returns a GateResult with is_valid=True if all checks pass.
    """
    violations = []
    warnings = []

    # 1. Removable kernel check
    if is_removable_kernel(kernel_code):
        violations.append(
            "REMOVABLE KERNEL: This kernel is trivial (identity/memcpy/view-only). "
            "Skip optimization and select the next hottest non-removable kernel."
        )

    # 2. SMEM budget check (if tile dimensions provided)
    if tile_dims:
        valid, msg = validate_smem_budget(tile_dims, bytes_per_elem, num_buffers, gpu_type)
        if not valid:
            violations.append(f"SMEM BUDGET: {msg}")

    # 3. Coalescing-first check
    if proposed_method:
        valid, msg = check_coalescing_first(bottleneck, kernel_structure, proposed_method)
        if not valid:
            violations.append(msg)

    # 4. One-method discipline (if prescription provided)
    if prescription:
        valid, msg = validate_single_method(prescription, history)
        if not valid:
            violations.append(msg)

    # 5. Allowed methods check
    allowed = get_allowed_methods(bottleneck, kernel_structure)
    if proposed_method and proposed_method not in allowed:
        warnings.append(
            f"Method '{proposed_method}' is not in the recommended set for "
            f"{bottleneck.value} + {kernel_structure.name}: {allowed}. "
            f"Proceeding but flagging for review."
        )

    # 6. Headroom tier
    tier = classify_headroom(primary_utilization_pct)

    # 7. Tier-L honesty
    if tier == HeadroomTier.LOW:
        warnings.append(
            "TIER-L: Primary limiter is >80% utilized with structurally unavoidable bytes. "
            "Expected speedup should be small. Large speedup claims require justification."
        )

    return GateResult(
        is_valid=len(violations) == 0,
        violations=violations,
        tier=tier,
        bottleneck=bottleneck,
        allowed_methods=allowed,
        forbidden_methods=[],
        warnings=warnings,
    )
