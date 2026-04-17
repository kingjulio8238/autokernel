"""CLI entry point for kernel-code.

Commands:
    kernel-code            — launch the interactive shell (default)
    kernel-code shell      — launch the interactive shell (explicit)
    kernel-code optimize   — launch the optimization TUI (non-interactive)
    kernel-code dashboard  — open the Plotly Dash dashboard in a browser
"""

from __future__ import annotations

import webbrowser

import click


class _ShellDefaultGroup(click.Group):
    """Click group that launches the shell when no subcommand is given.

    If the user runs ``kernel-code`` with no arguments, or with arguments
    that don't match a known subcommand, fall through to the shell.
    """

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        # If no args at all, invoke the shell
        if not args:
            args = ["shell"]
        return super().parse_args(ctx, args)

    def resolve_command(self, ctx: click.Context, args: list[str]) -> tuple:
        try:
            return super().resolve_command(ctx, args)
        except click.UsageError:
            # Unknown subcommand -- fall through to shell
            return "shell", "shell", args


@click.group(cls=_ShellDefaultGroup)
@click.version_option(version="0.1.0", prog_name="kernel-code")
def main() -> None:
    """kernel code -- terminal-native GPU kernel optimization tool."""


@main.command()
@click.option("--session", type=str, default=None, help="Resume a specific session ID.")
def shell(session: str | None) -> None:
    """Launch the interactive kernel optimization shell (default)."""
    from kernel_code.shell import KernelCodeShell

    s = KernelCodeShell(session_id=session)
    try:
        s.run()
    except (SystemExit, KeyboardInterrupt):
        pass


@main.command()
@click.option("--reference", type=click.Path(exists=True), help="Path to reference kernel file.")
@click.option("--level", type=int, default=1, help="KernelBench level (1-3).")
@click.option("--problem", type=int, default=23, help="Problem number within level.")
@click.option(
    "--backend",
    type=click.Choice(["triton", "cuda"]),
    default="triton",
    help="Code-generation backend.",
)
@click.option("--iterations", type=int, default=20, help="Number of optimization iterations.")
@click.option("--session", type=str, default=None, help="Resume a specific session ID.")
@click.option(
    "--mock/--no-mock",
    default=True,
    help="Use mock data instead of the live engine (default: mock).",
)
@click.option(
    "--model",
    type=str,
    default=None,
    help="LLM model ID (e.g. claude-sonnet-4-20250514). Requires --no-mock.",
)
@click.option(
    "--gpu",
    type=click.Choice(["H100", "A100-80GB", "A100-40GB", "L40S"]),
    default="H100",
    help="GPU type for Modal eval.",
)
def optimize(
    reference: str | None,
    level: int,
    problem: int,
    backend: str,
    iterations: int,
    session: str | None,
    mock: bool,
    model: str | None,
    gpu: str,
) -> None:
    """Launch the kernel optimization TUI.

    With --mock (the default), generates synthetic data for development.
    With --no-mock, runs the real openkernel engine (requires API key / Modal).
    """
    from kernel_code.tui.app import KernelCodeApp

    if mock:
        # ---- Mock mode: run engine with test mocks, launch TUI ----
        from tests.mocks import MockInnerLoop, MockLLMCaller

        from kernel_code.mock_data import generate_mock_session

        session_path = generate_mock_session(num_iterations=iterations, session_id=session)
        click.echo(f"Session data: {session_path}")
        app = KernelCodeApp(session_path=session_path)
        app.run()
    else:
        # ---- Live mode: run the real openkernel engine ----
        from pathlib import Path

        from openkernel.config import Backend, GpuType, ModelConfig, ModalConfig, OpenKernelConfig

        from kernel_code.integration import OpenKernelBridge, TraceBridge

        # Build config from CLI flags
        config = OpenKernelConfig(
            backend=Backend.CUDA if backend == "cuda" else Backend.TRITON,
            max_iterations=iterations,
            modal=ModalConfig(gpu_type=GpuType(gpu)),
        )
        if model is not None:
            config.model = ModelConfig(model_id=model)

        problem_label = f"L{level}#{problem}"

        # Read reference source
        if reference is not None:
            reference_source = Path(reference).read_text()
        else:
            click.echo("Error: --reference is required in live mode (--no-mock).", err=True)
            raise SystemExit(1)

        # Create bridge
        bridge = OpenKernelBridge(
            config=config,
            session_id=session,
            problem_label=problem_label,
            hardware=gpu,
            backend=backend,
        )

        click.echo(f"Session: {bridge.session_id}")
        click.echo(f"Cache:   {bridge.cache_path}")
        click.echo(f"Model:   {config.model.model_id}")
        click.echo(f"GPU:     {gpu}")
        click.echo(f"Backend: {backend}")
        click.echo()

        # Start trace capture
        trace_bridge = TraceBridge(session_id=bridge.session_id, config=config)
        trace_bridge.start(
            problem_id=problem_label,
            hardware=gpu,
            backend=backend,
            model_id=config.model.model_id,
        )

        # Launch the TUI in the background -- it reads the cache file on a timer.
        # The bridge writes to the same file after each iteration.
        # We run optimization first, then launch the TUI with the final data.
        # (For truly live display, the TUI and engine would run in separate threads;
        #  this is the simple sequential version.)
        click.echo("Running optimization...")
        result = bridge.run_optimization(reference_source)

        # Record iterations into trace
        for it_data in bridge._iterations:
            trace_bridge.record_iteration(it_data)

        trace_bridge.finish(
            final_speedup=result.final_speedup,
            final_correct=result.final_speedup > 1.0,
        )

        click.echo(f"Optimization complete: {result.final_speedup:.2f}x speedup")
        click.echo(f"Intents explored: {result.intents_explored}")
        click.echo(f"Wall time: {result.wall_time_seconds:.1f}s")
        click.echo()

        # Now launch TUI to review the results
        app = KernelCodeApp(session_path=bridge.cache_path)
        app.run()


@main.command()
@click.option("--session", type=str, default=None, help="Session ID to display.")
@click.option("--port", type=int, default=8050, help="Port for the dashboard server.")
def dashboard(session: str | None, port: int) -> None:
    """Open the Plotly Dash dashboard in a browser."""
    from kernel_code.mock_data import generate_mock_session

    if session is None:
        session_path = generate_mock_session(num_iterations=20)
        session = session_path.stem
        click.echo(f"Generated mock session: {session}")

    click.echo(f"Starting dashboard on http://localhost:{port} ...")

    webbrowser.open(f"http://localhost:{port}")

    from kernel_code.dashboard.server import create_dash_app

    app = create_dash_app(session_id=session)
    app.run(debug=False, port=port)


if __name__ == "__main__":
    main()
