"""Optimization pipeline integration — bridges KernelAgent's 6-agent
hardware-aware optimization pipeline with kernel-code's orchestration.

Architecture::

    Direct path (current default):
        TritonKernelAgent → VerificationWorker → Modal eval

    Optimization pipeline (requires local GPU + kernel_perf_agent):
        OptimizationManager → OptimizationWorker
            → KernelProfiler (NCU)
            → BottleneckAnalyzer
            → OptimizationOrchestrator
            → Benchmarking
            → VerificationWorker

The optimization pipeline provides hardware-aware optimization using
NCU profiling data to guide the LLM toward bottleneck-specific fixes.
It requires:
    1. kernel_perf_agent package (Meta internal, provides NCU roofline)
    2. Local NVIDIA GPU with NCU (NVIDIA Compute Profiler) installed
    3. CUDA toolkit in PATH

When these dependencies are unavailable, falls back to the Direct path.

Usage::

    from kernel_code.integration.opt_pipeline import create_optimizer

    optimizer = create_optimizer(
        engine="kernel-agent-opt",  # or "kernel-agent" for Direct
        reference_source=code,
        model_name="gpt-4o",
        hardware="L40S",
    )
    result = optimizer.run()
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kernel_code.live_display import LiveOptimizationDisplay
    from kernel_code.run_log import RunLogger

logger = logging.getLogger(__name__)


def is_opt_pipeline_available() -> bool:
    """Check if the optimization pipeline dependencies are available."""
    try:
        from kernel_perf_agent.kernel_opt.roofline.ncu_roofline import (  # noqa: F401
            RooflineConfig,
        )
    except ImportError:
        return False

    # Check for local GPU
    try:
        import torch
        if not torch.cuda.is_available():
            return False
    except ImportError:
        return False

    # Check for NCU binary
    import shutil
    if not shutil.which("ncu"):
        return False

    return True


def create_optimizer(
    engine: str,
    reference_source: str,
    model_name: str = "gpt-4o",
    num_workers: int = 4,
    max_rounds: int = 10,
    hardware: str = "L40S",
    live_display: "LiveOptimizationDisplay | None" = None,
    run_logger: "RunLogger | None" = None,
    use_modal: bool = True,
    problem_format: str = "auto",
) -> Any:
    """Create the appropriate optimizer based on engine selection.

    Args:
        engine: "kernel-agent" (Direct path) or "kernel-agent-opt"
                (hardware-aware optimization pipeline).

    Returns:
        An optimizer with a .run() method returning a result dict.
    """
    if engine == "kernel-agent-opt":
        if not is_opt_pipeline_available():
            logger.warning(
                "Optimization pipeline requested but dependencies unavailable "
                "(needs kernel_perf_agent + local GPU + NCU). "
                "Falling back to Direct path."
            )
            engine = "kernel-agent"
        else:
            return OptPipelineRunner(
                reference_source=reference_source,
                model_name=model_name,
                num_workers=num_workers,
                max_rounds=max_rounds,
                hardware=hardware,
                live_display=live_display,
                run_logger=run_logger,
            )

    # Default: Direct path via KernelAgentBridge
    from kernel_code.integration.kernel_agent_bridge import KernelAgentBridge

    return KernelAgentBridge(
        reference_source=reference_source,
        model_name=model_name,
        num_workers=num_workers,
        max_rounds=max_rounds,
        hardware=hardware,
        live_display=live_display,
        run_logger=run_logger,
        use_modal=use_modal,
        problem_format=problem_format,
    )


class OptPipelineRunner:
    """Runs the KernelAgent hardware-aware optimization pipeline.

    Uses OptimizationManager with OptimizationWorker for NCU-guided
    bottleneck analysis and targeted kernel optimization.

    Requires: kernel_perf_agent, local NVIDIA GPU, NCU profiler.
    """

    def __init__(
        self,
        reference_source: str,
        model_name: str = "gpt-4o",
        num_workers: int = 4,
        max_rounds: int = 10,
        hardware: str = "L40S",
        live_display: "LiveOptimizationDisplay | None" = None,
        run_logger: "RunLogger | None" = None,
    ) -> None:
        self._reference = reference_source
        self._model_name = model_name
        self._num_workers = num_workers
        self._max_rounds = max_rounds
        self._hardware = hardware
        self._live_display = live_display
        self._run_logger = run_logger

    def run(self) -> dict[str, Any]:
        """Run the optimization pipeline.

        Returns dict with: success, kernel_code, speedup, rounds, worker_id
        """
        import tempfile
        import time

        from kernel_agent.opt_manager import OptimizationManager
        from kernel_code.problem import detect_format, make_self_contained, Problem

        start_time = time.time()
        fmt = detect_format(self._reference)

        problem = Problem(reference_code=self._reference, format=fmt)
        self_contained = make_self_contained(problem)

        # Write reference to temp file for OptimizationManager
        tmpdir = Path(tempfile.mkdtemp(prefix="openkernel_opt_"))
        ref_path = tmpdir / "problem.py"
        ref_path.write_text(self_contained)

        # Build test code
        from kernel_code.problem import build_test_code
        problem.reference_code = self_contained
        test_code = build_test_code(problem)

        if self._live_display:
            self._live_display.update_phase(
                f"OptPipeline: {self._num_workers} workers × {self._max_rounds} rounds "
                f"(NCU-guided, {self._model_name})"
            )

        try:
            manager = OptimizationManager(
                strategy="beam_search",
                num_workers=self._num_workers,
                strategy_config={
                    "num_top_kernels": 2,
                    "num_bottlenecks": 2,
                },
            )

            # Generate initial kernel via Direct path
            from kernel_agent.agent import TritonKernelAgent
            agent = TritonKernelAgent(
                num_workers=1,
                max_rounds=3,
                model_name=self._model_name,
            )

            initial_result = agent.generate_kernel(
                problem_description=f"Optimize into Triton for {self._hardware}:\n```python\n{self_contained}\n```",
                test_code=test_code,
            )

            if not initial_result.get("success"):
                return {
                    "success": False,
                    "kernel_code": "",
                    "speedup": 0.0,
                    "error": "Failed to generate initial kernel",
                    "elapsed": time.time() - start_time,
                }

            initial_kernel = initial_result["kernel_code"]

            # Run optimization pipeline
            result = manager.run_optimization(
                initial_kernel=initial_kernel,
                problem_file=ref_path,
                test_code=test_code,
                max_rounds=self._max_rounds,
            )

            success = result.get("success", False)
            return {
                "success": success,
                "kernel_code": result.get("best_kernel", initial_kernel),
                "speedup": result.get("best_speedup", 0.0),
                "rounds": result.get("total_rounds", self._max_rounds),
                "elapsed": time.time() - start_time,
                "per_worker": [],
            }

        except Exception as exc:
            logger.error("Optimization pipeline failed: %s", exc)
            return {
                "success": False,
                "kernel_code": "",
                "speedup": 0.0,
                "error": str(exc),
                "elapsed": time.time() - start_time,
            }


# ---------------------------------------------------------------------------
# Fuser stub — for Level 2+ KernelBench multi-kernel problems
# ---------------------------------------------------------------------------


def needs_fusion(reference_code: str) -> bool:
    """Detect if a problem likely needs the Fuser pipeline.

    Level 2+ KernelBench problems contain complex PyTorch modules with
    multiple operations that should be fused into optimized subgraphs.

    Heuristics:
    - Multiple nn.Module subclasses (handles multi-base inheritance)
    - Complex forward() with many distinct ops (torch.* and F.*)
    - Sequential/ModuleList patterns
    """
    import re

    # Count nn.Module subclasses — handles multi-base like (nn.Module, Mixin)
    module_classes = len(re.findall(
        r"class\s+\w+\s*\([^)]*\bnn\.Module\b",
        reference_code,
    ))
    if module_classes > 2:
        return True

    # Count distinct ops: torch.* calls
    torch_ops = re.findall(
        r"torch\.(matmul|mm|bmm|conv\w*|relu|sigmoid|softmax|"
        r"layer_norm|layernorm|batch_norm|dropout|linear|"
        r"add|mul|cat|split|reshape|transpose|permute|view|"
        r"norm|cross|einsum|gather|scatter)",
        reference_code,
    )
    # Also count F.* calls (torch.nn.functional)
    f_ops = re.findall(
        r"F\.(relu|gelu|silu|sigmoid|softmax|log_softmax|"
        r"conv[12]d|linear|dropout|batch_norm|layer_norm|"
        r"max_pool[12]d|avg_pool[12]d|interpolate|pad|"
        r"cross_entropy|mse_loss|binary_cross_entropy)",
        reference_code,
    )
    distinct_ops = set(torch_ops) | set(f_ops)
    if len(distinct_ops) >= 4:
        return True

    # Sequential or ModuleList patterns
    if "nn.Sequential" in reference_code or "nn.ModuleList" in reference_code:
        return True

    return False


class FuserPipeline:
    """Stub for KernelAgent's Fuser decomposition pipeline.

    The Fuser decomposes complex PyTorch modules into fusable subgraphs,
    then optimizes each subgraph independently. Required for Level 2+
    KernelBench problems where the reference implementation contains
    multiple operations that should be fused.

    The Fuser directory from KernelAgent was not copied into our fork.
    To enable:
    1. Copy kernel_agent/Fuser/ from the upstream KernelAgent repo
    2. Import FuserPipeline and replace this stub
    3. Wire into the optimization loop for detected multi-op problems
    """

    def __init__(self, reference_code: str, model_name: str = "gpt-4o") -> None:
        self._reference = reference_code
        self._model = model_name

    def decompose(self) -> list[dict]:
        """Decompose the reference into fusable subgraphs.

        Returns a list of dicts, each with:
        - 'name': subgraph name
        - 'code': subgraph PyTorch code
        - 'inputs': input tensor specs
        - 'outputs': output tensor specs
        """
        raise NotImplementedError(
            "Fuser pipeline requires kernel_agent.Fuser — not yet integrated. "
            "Copy Fuser/ from upstream KernelAgent and update this stub. "
            "See: https://github.com/meta-pytorch/KernelAgent"
        )

    def fuse(self, optimized_subgraphs: list[dict]) -> str:
        """Fuse optimized subgraphs back into a single kernel.

        Args:
            optimized_subgraphs: List of dicts with 'name' and 'kernel_code'

        Returns:
            Complete fused kernel code
        """
        raise NotImplementedError(
            "Fuser pipeline requires kernel_agent.Fuser — not yet integrated."
        )
