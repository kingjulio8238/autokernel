"""openkernel CLI — command-line interface for kernel optimization."""

from __future__ import annotations

import asyncio
from pathlib import Path

import click

import openkernel
from openkernel.config import Backend, EvalMode, OpenKernelConfig


@click.group()
@click.version_option(version=openkernel.__version__, prog_name="openkernel")
def main() -> None:
    """openkernel — Self-recursive GPU kernel optimization engine."""


# ---------------------------------------------------------------------------
# openkernel optimize
# ---------------------------------------------------------------------------


@main.command()
@click.option(
    "--reference",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to reference kernel source file.",
)
@click.option(
    "--backend",
    type=click.Choice(["triton", "cuda"], case_sensitive=False),
    default="triton",
    show_default=True,
    help="Target backend.",
)
@click.option(
    "--model",
    "model_id",
    default=None,
    help="LLM model ID (e.g. claude-sonnet-4-20250514).",
)
@click.option(
    "--eval-mode",
    type=click.Choice(["fast", "thorough"], case_sensitive=False),
    default="fast",
    show_default=True,
    help="Evaluation mode.",
)
@click.option(
    "--max-iterations",
    type=int,
    default=None,
    help="Maximum optimization iterations (default: 100).",
)
def optimize(
    reference: Path,
    backend: str,
    model_id: str | None,
    eval_mode: str,
    max_iterations: int | None,
) -> None:
    """Optimize a GPU kernel from a reference implementation."""
    reference_source = reference.read_text()

    # Build config
    config_updates: dict = {
        "backend": Backend(backend),
        "eval_mode": EvalMode(eval_mode),
    }
    if max_iterations is not None:
        config_updates["max_iterations"] = max_iterations

    config = OpenKernelConfig(**config_updates)

    if model_id is not None:
        config = config.model_copy(
            update={"model": config.model.model_copy(update={"model_id": model_id})}
        )

    click.echo(f"openkernel v{openkernel.__version__}")
    click.echo(f"Backend:        {config.backend.value}")
    click.echo(f"Eval mode:      {config.eval_mode.value}")
    click.echo(f"Model:          {config.model.model_id}")
    click.echo(f"Max iterations: {config.max_iterations}")
    click.echo(f"Reference:      {reference}")
    click.echo()

    result = asyncio.run(
        openkernel.optimize(
            reference_source=reference_source,
            backend=backend,
            config=config,
        )
    )

    click.echo("=" * 60)
    click.echo("Optimization complete")
    click.echo("=" * 60)
    click.echo(f"Final speedup:       {result.final_speedup:.3f}x")
    click.echo(f"Intents explored:    {result.intents_explored}")
    click.echo(f"Intents succeeded:   {result.intents_succeeded}")
    click.echo(f"Intents failed:      {result.intents_failed}")
    click.echo(f"Total iterations:    {result.iterations_total}")
    click.echo(f"Wall time:           {result.wall_time_seconds:.1f}s")
    click.echo(f"Stagnation:          {'yes' if result.stagnation_triggered else 'no'}")

    if result.final_kernel:
        click.echo()
        click.echo("Best kernel:")
        click.echo("-" * 60)
        click.echo(result.final_kernel)


# ---------------------------------------------------------------------------
# openkernel evaluate
# ---------------------------------------------------------------------------


@main.command("evaluate")
@click.option(
    "--kernel",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to kernel source file.",
)
@click.option(
    "--reference",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to reference kernel source file.",
)
@click.option(
    "--eval-mode",
    type=click.Choice(["fast", "thorough"], case_sensitive=False),
    default="fast",
    show_default=True,
    help="Evaluation mode.",
)
def evaluate_cmd(
    kernel: Path,
    reference: Path,
    eval_mode: str,
) -> None:
    """Evaluate a kernel against a reference implementation."""
    kernel_source = kernel.read_text()
    reference_source = reference.read_text()

    config = OpenKernelConfig(eval_mode=EvalMode(eval_mode))

    click.echo(f"openkernel v{openkernel.__version__}")
    click.echo(f"Eval mode:  {config.eval_mode.value}")
    click.echo(f"Kernel:     {kernel}")
    click.echo(f"Reference:  {reference}")
    click.echo()

    result = asyncio.run(
        openkernel.evaluate(
            kernel_source=kernel_source,
            reference_source=reference_source,
            config=config,
        )
    )

    click.echo("=" * 60)
    click.echo("Evaluation result")
    click.echo("=" * 60)
    click.echo(f"Status:          {result.status.value}")
    click.echo(f"Correct:         {result.correct}")
    click.echo(f"Speedup:         {result.speedup:.3f}x")
    click.echo(f"Runtime:         {result.runtime_us:.1f} us")
    click.echo(f"Ref runtime:     {result.ref_runtime_us:.1f} us")
    click.echo(f"Eval time:       {result.eval_seconds:.1f}s")
    if result.error:
        click.echo(f"Error:           {result.error}")

    profile = result.profile
    if profile.bottleneck_type.value != "unknown":
        click.echo()
        click.echo("Profile:")
        click.echo(f"  Bottleneck:    {profile.bottleneck_type.value}")
        click.echo(f"  Occupancy:     {profile.occupancy:.1%}")
        click.echo(f"  Bandwidth:     {profile.bandwidth_utilization:.1%}")
        click.echo(f"  Compute:       {profile.compute_utilization:.1%}")


# ---------------------------------------------------------------------------
# openkernel info
# ---------------------------------------------------------------------------


@main.command()
def info() -> None:
    """Print version, config defaults, and available backends."""
    defaults = OpenKernelConfig()

    click.echo(f"openkernel v{openkernel.__version__}")
    click.echo()
    click.echo("Backends:")
    for b in Backend:
        click.echo(f"  - {b.value}")
    click.echo()
    click.echo("Eval modes:")
    for m in EvalMode:
        click.echo(f"  - {m.value}")
    click.echo()
    click.echo("Config defaults:")
    click.echo(f"  backend:              {defaults.backend.value}")
    click.echo(f"  eval_mode:            {defaults.eval_mode.value}")
    click.echo(f"  max_iterations:       {defaults.max_iterations}")
    click.echo(f"  max_retries_per_intent: {defaults.max_retries_per_intent}")
    click.echo(f"  stagnation_threshold: {defaults.stagnation_threshold}")
    click.echo(f"  model:                {defaults.model.model_id}")
    click.echo(f"  gpu:                  {defaults.modal.gpu_type.value}")


if __name__ == "__main__":
    main()
