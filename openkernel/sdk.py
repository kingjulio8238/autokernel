"""openkernel SDK -- simple API for embedding kernel optimization.

Provides synchronous, thin wrappers around the async openkernel engine so
that scripts, notebooks, and CI pipelines can optimize and evaluate GPU
kernels with minimal boilerplate.

Usage::

    import openkernel.sdk as ok

    # Optimize a kernel
    result = ok.optimize("softmax.py", backend="triton", max_iterations=20)
    print(f"Best speedup: {result.speedup}x")
    print(result.best_kernel)

    # Evaluate a kernel
    eval_result = ok.evaluate("my_kernel.py", "reference.py")
    print(f"Correct: {eval_result.correct}, Speedup: {eval_result.speedup}x")

    # Search skills
    skills = ok.search_skills("softmax reduction")
    for s in skills:
        print(f"{s['name']}: {s['approach'][:50]}")

    # Session context manager
    with ok.session("my-experiment") as s:
        r1 = s.optimize("kernel_a.py")
        r2 = s.optimize("kernel_b.py")
        s.summary()  # prints both results
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class OptimizeResult:
    """Simplified optimization result for SDK users."""

    speedup: float
    correct: bool
    iterations: int
    kept: int
    cost_usd: float
    best_kernel: str  # kernel source code
    summary: str  # one-line summary

    def _repr_html_(self) -> str:
        """Render as HTML in Jupyter notebooks."""
        color = "#276749" if self.speedup > 1.0 else "#c53030"
        return f'''
    <div style="font-family: monospace; padding: 12px; border: 1px solid #e0ddd8; border-radius: 4px; background: #f5f2ed;">
        <h3 style="margin: 0 0 8px 0; color: #1a1a1a;">Optimization Result</h3>
        <table style="border-collapse: collapse;">
            <tr><td style="padding: 2px 12px 2px 0; color: #6b6b6b;">Speedup</td>
                <td style="font-weight: bold; color: {color};">{self.speedup:.2f}x</td></tr>
            <tr><td style="padding: 2px 12px 2px 0; color: #6b6b6b;">Correct</td>
                <td>{"\u2713" if self.correct else "\u2717"}</td></tr>
            <tr><td style="padding: 2px 12px 2px 0; color: #6b6b6b;">Iterations</td>
                <td>{self.iterations} ({self.kept} kept)</td></tr>
            <tr><td style="padding: 2px 12px 2px 0; color: #6b6b6b;">Cost</td>
                <td>${self.cost_usd:.3f}</td></tr>
        </table>
        <details style="margin-top: 8px;">
            <summary style="cursor: pointer; color: #6b6b6b;">Best Kernel</summary>
            <pre style="background: #1a1a1a; color: #e0ddd8; padding: 8px; border-radius: 4px; font-size: 12px; overflow-x: auto;">{self.best_kernel[:500]}</pre>
        </details>
    </div>
    '''


@dataclass
class EvalResult:
    """Simplified eval result for SDK users."""

    correct: bool
    speedup: float
    runtime_us: float
    ref_runtime_us: float
    bottleneck: str

    def _repr_html_(self) -> str:
        """Render as HTML in Jupyter notebooks."""
        color = "#276749" if self.speedup > 1.0 else "#c53030"
        return f'''
    <div style="font-family: monospace; padding: 12px; border: 1px solid #e0ddd8; border-radius: 4px; background: #f5f2ed;">
        <h3 style="margin: 0 0 8px 0; color: #1a1a1a;">Eval Result</h3>
        <table style="border-collapse: collapse;">
            <tr><td style="padding: 2px 12px 2px 0; color: #6b6b6b;">Speedup</td>
                <td style="font-weight: bold; color: {color};">{self.speedup:.2f}x</td></tr>
            <tr><td style="padding: 2px 12px 2px 0; color: #6b6b6b;">Correct</td>
                <td>{"\u2713" if self.correct else "\u2717"}</td></tr>
            <tr><td style="padding: 2px 12px 2px 0; color: #6b6b6b;">Bottleneck</td>
                <td>{self.bottleneck}</td></tr>
            <tr><td style="padding: 2px 12px 2px 0; color: #6b6b6b;">Runtime</td>
                <td>{self.runtime_us:.1f} us (ref: {self.ref_runtime_us:.1f} us)</td></tr>
        </table>
    </div>
    '''


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_source(path: str | Path) -> str:
    """Read source code from a file path, returning the text verbatim."""
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Source file not found: {p}")
    return p.read_text()


def _run_async(coro: Any) -> Any:
    """Run an async coroutine synchronously.

    Handles the case where an event loop is already running (e.g. inside
    Jupyter notebooks) by using ``nest_asyncio`` if available, otherwise
    falls back to ``asyncio.run``.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        # Inside a running event loop (Jupyter, async REPL, etc.)
        try:
            import nest_asyncio  # type: ignore[import-untyped]

            nest_asyncio.apply()
        except ImportError:
            raise RuntimeError(
                "An asyncio event loop is already running. "
                "Install nest_asyncio (`pip install nest_asyncio`) to use "
                "the openkernel SDK inside notebooks or async contexts."
            )
        return asyncio.get_event_loop().run_until_complete(coro)
    else:
        return asyncio.run(coro)


def _build_config(
    backend: str = "triton",
    model: str | None = None,
    max_iterations: int = 50,
    gpu: str = "L40S",
    config_path: str | Path | None = None,
) -> "OpenKernelConfig":
    """Build an OpenKernelConfig from SDK-level arguments."""
    from openkernel.config import (
        Backend,
        GpuType,
        ModelConfig,
        ModalConfig,
        OpenKernelConfig,
    )

    if config_path is not None:
        cfg = OpenKernelConfig.from_yaml(config_path)
        # Still honour explicit overrides
        updates: dict[str, Any] = {}
        if backend != "triton":
            updates["backend"] = Backend(backend)
        if max_iterations != 50:
            updates["max_iterations"] = max_iterations
        if gpu != "L40S":
            updates["modal"] = cfg.modal.model_copy(
                update={"gpu_type": GpuType(gpu)}
            )
        if model is not None:
            updates["model"] = cfg.model.model_copy(
                update={"model_id": model}
            )
        if updates:
            cfg = cfg.model_copy(update=updates)
        return cfg

    model_cfg = ModelConfig()
    if model is not None:
        model_cfg = model_cfg.model_copy(update={"model_id": model})

    modal_cfg = ModalConfig(gpu_type=GpuType(gpu))

    return OpenKernelConfig(
        backend=Backend(backend),
        max_iterations=max_iterations,
        model=model_cfg,
        modal=modal_cfg,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def optimize(
    reference: str | Path,
    backend: str = "triton",
    model: str | None = None,
    max_iterations: int = 50,
    gpu: str = "L40S",
    config_path: str | Path | None = None,
    mock: bool = False,
) -> OptimizeResult:
    """Optimize a kernel. Synchronous wrapper around the async engine.

    Args:
        reference: Path to the reference kernel source file.
        backend: Target backend -- ``"triton"`` or ``"cuda"``.
        model: LLM model identifier (e.g. ``"openai/MiniMax-M2.5"``).
            When ``None`` the default from :class:`ModelConfig` is used.
        max_iterations: Maximum optimization iterations.
        gpu: GPU type for evaluation (``"L40S"``, ``"H100"``, etc.).
        config_path: Optional path to a YAML configuration file.
            When provided, the file is loaded first and explicit keyword
            arguments override individual fields.
        mock: If ``True``, use a mock evaluation function instead of
            real Modal GPU evaluation. Useful for testing and CI.

    Returns:
        :class:`OptimizeResult` with speedup, best kernel source, and
        cost information.
    """
    import openkernel as ok_engine

    reference_source = _read_source(reference)
    config = _build_config(
        backend=backend,
        model=model,
        max_iterations=max_iterations,
        gpu=gpu,
        config_path=config_path,
    )

    eval_fn = None
    if mock:
        from tests.mocks import create_mock_eval_fn

        eval_fn = create_mock_eval_fn(seed=42)

    result = _run_async(
        ok_engine.optimize(
            reference_source=reference_source,
            backend=backend,
            config=config,
            eval_fn=eval_fn,
        )
    )

    correct = result.final_speedup > 0.0
    return OptimizeResult(
        speedup=result.final_speedup,
        correct=correct,
        iterations=result.iterations_total,
        kept=result.intents_succeeded,
        cost_usd=result.total_cost_usd,
        best_kernel=result.final_kernel,
        summary=(
            f"{result.final_speedup:.2f}x speedup in "
            f"{result.iterations_total} iterations "
            f"({result.intents_succeeded}/{result.intents_explored} intents kept, "
            f"${result.total_cost_usd:.4f})"
        ),
    )


def evaluate(
    kernel: str | Path,
    reference: str | Path,
    mode: str = "fast",
    gpu: str = "L40S",
) -> EvalResult:
    """Evaluate a kernel against a reference. Synchronous.

    Args:
        kernel: Path to the kernel source file (defines ``ModelNew``).
        reference: Path to the reference source file (defines ``Model``,
            ``get_inputs``, ``get_init_inputs``).
        mode: Evaluation mode -- ``"fast"`` or ``"thorough"``.
        gpu: GPU type for evaluation.

    Returns:
        :class:`EvalResult` with correctness, speedup, and timing data.
    """
    import openkernel as ok_engine
    from openkernel.config import EvalMode, GpuType, ModalConfig, OpenKernelConfig

    kernel_source = _read_source(kernel)
    reference_source = _read_source(reference)

    config = OpenKernelConfig(
        eval_mode=EvalMode(mode),
        modal=ModalConfig(gpu_type=GpuType(gpu)),
    )

    raw = _run_async(
        ok_engine.evaluate(
            kernel_source=kernel_source,
            reference_source=reference_source,
            config=config,
        )
    )

    return EvalResult(
        correct=raw.correct,
        speedup=raw.speedup,
        runtime_us=raw.runtime_us,
        ref_runtime_us=raw.ref_runtime_us,
        bottleneck=raw.profile.bottleneck_type.value,
    )


def profile(
    kernel: str | Path,
    reference: str | Path,
) -> dict:
    """Profile a kernel and return profiling data.

    Runs a ``thorough`` evaluation (more performance trials) and returns
    the full profile metrics as a plain dictionary.

    Args:
        kernel: Path to the kernel source file.
        reference: Path to the reference source file.

    Returns:
        Dictionary with profiling metrics (bottleneck type, roofline
        position, cache efficiency, occupancy, etc.).
    """
    import openkernel as ok_engine
    from openkernel.config import EvalMode, OpenKernelConfig

    kernel_source = _read_source(kernel)
    reference_source = _read_source(reference)

    config = OpenKernelConfig(
        eval_mode=EvalMode.THOROUGH,
        enable_deep_profiling=True,
    )

    raw = _run_async(
        ok_engine.evaluate(
            kernel_source=kernel_source,
            reference_source=reference_source,
            config=config,
        )
    )

    p = raw.profile
    return {
        "correct": raw.correct,
        "speedup": raw.speedup,
        "runtime_us": raw.runtime_us,
        "ref_runtime_us": raw.ref_runtime_us,
        "bottleneck_type": p.bottleneck_type.value,
        "roofline_position": p.roofline_position,
        "cache_efficiency": p.cache_efficiency,
        "occupancy": p.occupancy,
        "bandwidth_utilization": p.bandwidth_utilization,
        "compute_utilization": p.compute_utilization,
        "top_stalls": p.top_stalls,
        "raw_metrics": p.raw_metrics,
    }


def search_skills(query: str, top_k: int = 5) -> list[dict]:
    """Search the optimization skill library.

    Performs keyword-based retrieval against the bundled skill library
    (``data/skills/*.json``).

    Args:
        query: Free-text search query (e.g. ``"softmax reduction"``).
        top_k: Maximum number of results to return.

    Returns:
        List of dictionaries, each with keys ``id``, ``name``,
        ``trigger``, ``approach``, ``backend``, ``tags``, and
        optionally ``code_template``.
    """
    from dataclasses import asdict

    from openkernel.memory.skill_library import SkillLibrary

    lib = SkillLibrary()
    lib.load()

    results = lib.search_skills(query, top_k=top_k)
    return [asdict(s) for s in results]


def compare(
    kernel_a: str | Path,
    kernel_b: str | Path,
    reference: str | Path,
) -> dict:
    """Compare two kernels against the same reference.

    Evaluates both kernels and returns a side-by-side comparison.

    Args:
        kernel_a: Path to the first kernel source file.
        kernel_b: Path to the second kernel source file.
        reference: Path to the reference source file.

    Returns:
        Dictionary with ``a``, ``b``, and ``comparison`` keys. Each
        kernel entry contains eval results; ``comparison`` contains
        the delta and which kernel is faster.
    """
    eval_a = evaluate(kernel_a, reference)
    eval_b = evaluate(kernel_b, reference)

    if eval_a.speedup > 0 and eval_b.speedup > 0:
        faster = "a" if eval_a.speedup >= eval_b.speedup else "b"
        delta = abs(eval_a.speedup - eval_b.speedup)
    elif eval_a.correct and not eval_b.correct:
        faster = "a"
        delta = eval_a.speedup
    elif eval_b.correct and not eval_a.correct:
        faster = "b"
        delta = eval_b.speedup
    else:
        faster = "neither"
        delta = 0.0

    return {
        "a": {
            "correct": eval_a.correct,
            "speedup": eval_a.speedup,
            "runtime_us": eval_a.runtime_us,
            "bottleneck": eval_a.bottleneck,
        },
        "b": {
            "correct": eval_b.correct,
            "speedup": eval_b.speedup,
            "runtime_us": eval_b.runtime_us,
            "bottleneck": eval_b.bottleneck,
        },
        "comparison": {
            "faster": faster,
            "speedup_delta": round(delta, 4),
        },
    }


# ---------------------------------------------------------------------------
# Session context manager
# ---------------------------------------------------------------------------


class Session:
    """Context manager for multi-kernel optimization sessions.

    Groups multiple ``optimize()`` calls under a named session and
    provides a summary of all results when done.

    Usage::

        with ok.session("my-experiment") as s:
            r1 = s.optimize("kernel_a.py")
            r2 = s.optimize("kernel_b.py")
            print(s.summary())
    """

    def __init__(self, name: str = "sdk-session") -> None:
        self._name = name
        self._results: list[tuple[str, OptimizeResult]] = []

    def __enter__(self) -> Session:
        return self

    def __exit__(self, *args: Any) -> None:
        pass

    def optimize(self, reference: str | Path, **kwargs: Any) -> OptimizeResult:
        """Optimize a kernel within this session.

        Accepts the same keyword arguments as the module-level
        :func:`optimize` function. The result is stored internally
        for :meth:`summary`.
        """
        result = optimize(reference, **kwargs)
        label = str(Path(reference).name)
        self._results.append((label, result))
        return result

    @property
    def results(self) -> list[tuple[str, OptimizeResult]]:
        """All (label, result) pairs collected in this session."""
        return list(self._results)

    def summary(self) -> str:
        """Return a formatted summary of all optimizations in this session.

        Also prints the summary to stdout for interactive use.
        """
        if not self._results:
            text = f"Session '{self._name}': no optimizations run."
            print(text)
            return text

        lines = [f"Session '{self._name}' -- {len(self._results)} kernel(s):"]
        total_cost = 0.0
        for label, r in self._results:
            status = "ok" if r.correct else "FAIL"
            lines.append(
                f"  {label}: {r.speedup:.2f}x [{status}] "
                f"({r.iterations} iters, ${r.cost_usd:.4f})"
            )
            total_cost += r.cost_usd

        best_label, best_r = max(self._results, key=lambda t: t[1].speedup)
        lines.append(f"  Best: {best_label} @ {best_r.speedup:.2f}x")
        lines.append(f"  Total cost: ${total_cost:.4f}")

        text = "\n".join(lines)
        print(text)
        return text


def session(name: str = "sdk-session") -> Session:
    """Create an optimization session context manager.

    Args:
        name: Human-readable name for the session (used in summaries).

    Returns:
        A :class:`Session` instance that can be used as a context manager.
    """
    return Session(name)
