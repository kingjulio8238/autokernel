"""Optimization lifecycle hooks.

Provides a simple event-driven hook system for the /optimize flow.
Hooks are plain Python callables -- no shell commands or HTTP endpoints
(those can be layered on later if needed).

Events
------
- ``pre_optimize``  -- fired before optimization begins
- ``post_iterate``  -- fired after every iteration (keep, discard, or error)
- ``post_keep``     -- fired when an iteration becomes the new best
- ``post_discard``  -- fired when an iteration is discarded (worse than best)
- ``post_optimize`` -- fired after the optimization loop completes

Usage::

    hooks = create_default_hooks(session, skill_library=lib)
    hooks.register("post_keep", my_custom_callback)
    hooks.fire("pre_optimize", config=config, iterations=20)
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

from rich.console import Console

logger = logging.getLogger(__name__)


class HookRegistry:
    """Registry for optimization lifecycle hooks."""

    # Event types
    PRE_OPTIMIZE = "pre_optimize"
    POST_ITERATE = "post_iterate"
    POST_KEEP = "post_keep"
    POST_DISCARD = "post_discard"
    POST_OPTIMIZE = "post_optimize"

    _VALID_EVENTS = {
        PRE_OPTIMIZE,
        POST_ITERATE,
        POST_KEEP,
        POST_DISCARD,
        POST_OPTIMIZE,
    }

    def __init__(self) -> None:
        self._hooks: dict[str, list[Callable]] = {evt: [] for evt in self._VALID_EVENTS}

    def register(self, event: str, callback: Callable) -> None:
        """Register a hook for an event.

        Raises:
            ValueError: If *event* is not a recognised lifecycle event.
        """
        if event not in self._VALID_EVENTS:
            raise ValueError(
                f"Unknown hook event {event!r}. "
                f"Valid events: {', '.join(sorted(self._VALID_EVENTS))}"
            )
        self._hooks[event].append(callback)

    def fire(self, event: str, **kwargs: Any) -> list[Any]:
        """Fire all hooks for an event.  Returns a list of results.

        Hooks that raise are logged and skipped -- a failing hook must never
        abort the optimization itself.
        """
        results: list[Any] = []
        for hook in self._hooks.get(event, []):
            try:
                results.append(hook(**kwargs))
            except Exception:
                logger.exception("Hook %r failed for event %r", hook, event)
        return results


# ---------------------------------------------------------------------------
# Default hook factories
# ---------------------------------------------------------------------------


def _make_pre_optimize_log(console: Console) -> Callable:
    """Log start time and configuration when optimization begins."""

    def _hook(*, config: dict, iterations: int, **_kw: Any) -> float:
        start = time.time()
        backend = config.get("backend", "triton")
        hardware = config.get("hardware", "H100")
        console.print(
            f"[bold]Starting optimization:[/bold] {iterations} iterations, "
            f"backend={backend}, hardware={hardware}"
        )
        logger.info(
            "Optimization started: iterations=%d backend=%s hardware=%s",
            iterations,
            backend,
            hardware,
        )
        return start

    return _hook


def _make_pre_optimize_cost_confirm(console: Console) -> Callable:
    """Prompt for cost confirmation before a live run."""
    from kernel_code.permissions import confirm_cost, estimate_cost

    def _hook(*, config: dict, iterations: int, **_kw: Any) -> bool:
        gpu_type = config.get("hardware", "H100")
        estimated = estimate_cost(iterations, gpu_type=gpu_type)
        approved = confirm_cost(estimated, gpu_type, iterations, console=console)
        if not approved:
            console.print("[yellow]Optimization cancelled by user.[/yellow]")
        return approved

    return _hook


def _make_post_keep_log(console: Console) -> Callable:
    """Log when a new best kernel is found."""

    def _hook(
        *, speedup: float, iteration: int, intent: str, **_kw: Any
    ) -> None:
        console.print(
            f"  [green]New best:[/green] {speedup:.2f}x at iter {iteration} "
            f"({intent})"
        )

    return _hook


def _make_post_keep_skill_evidence(skill_library: Any) -> Callable:
    """Record successful optimization evidence in the skill library."""

    def _hook(
        *,
        speedup: float,
        iteration: int,
        intent: str,
        problem: str = "",
        hardware: str = "",
        skill_id: str | None = None,
        **_kw: Any,
    ) -> None:
        if skill_library is None or skill_id is None:
            return
        skill_library.update_evidence(
            skill_id,
            {
                "problem": problem,
                "speedup": speedup,
                "hardware": hardware,
                "iteration": iteration,
                "intent": intent,
            },
        )

    return _hook


def _make_post_keep_evolution(template_evolver: Any) -> Callable:
    """Record a winning kernel in the template evolver for flywheel feedback."""

    def _hook(
        *,
        speedup: float,
        iteration: int,
        intent: str,
        kernel_code: str = "",
        hardware: str = "",
        backend: str = "",
        skill_id: str | None = None,
        **_kw: Any,
    ) -> None:
        if template_evolver is None or not kernel_code:
            return
        # Use skill_id if available, otherwise fall back to a generic label
        sid = skill_id or "unknown"
        template_evolver.record_win(
            skill_id=sid,
            kernel_code=kernel_code,
            speedup=speedup,
            hardware=hardware,
            backend=backend,
        )

    return _hook


def _make_post_discard_log(console: Console) -> Callable:
    """Log when an iteration is discarded."""

    def _hook(
        *,
        speedup: float,
        best_speedup: float,
        intent: str,
        **_kw: Any,
    ) -> None:
        console.print(
            f"  [red]Discarded:[/red] {speedup:.2f}x < best {best_speedup:.2f}x "
            f"({intent})"
        )

    return _hook


def _make_post_optimize_save(session_mod: Any) -> Callable:
    """Save best kernel and results via session.save_optimization_result."""

    def _hook(
        *,
        session: Any,
        run_id: str,
        reference_path: str,
        backend: str,
        config_path: str | None,
        cache_session_id: str,
        best_kernel_code: str,
        best_speedup: float,
        best_iteration: int,
        iterations_total: int,
        iterations_kept: int,
        cost: float,
        wall_time_seconds: float,
        hardware: str,
        top_iterations: list[dict],
        **_kw: Any,
    ) -> Any:
        return session_mod.save_optimization_result(
            session=session,
            run_id=run_id,
            reference_path=reference_path,
            backend=backend,
            config_path=config_path,
            cache_session_id=cache_session_id,
            best_kernel_code=best_kernel_code,
            best_speedup=best_speedup,
            best_iteration=best_iteration,
            iterations_total=iterations_total,
            iterations_kept=iterations_kept,
            cost=cost,
            wall_time_seconds=wall_time_seconds,
            hardware=hardware,
            top_iterations=top_iterations,
        )

    return _hook


def _make_post_optimize_summary(console: Console) -> Callable:
    """Print a final summary line after optimization."""

    def _hook(
        *,
        best_speedup: float,
        iterations_kept: int,
        iterations_total: int,
        cost: float,
        **_kw: Any,
    ) -> None:
        console.print()
        console.print(
            f"[bold green]Best: {best_speedup:.2f}x[/bold green] | "
            f"{iterations_kept}/{iterations_total} kept | "
            f"${cost:.2f}"
        )

    return _hook


def _make_post_optimize_dashboard_link(console: Console) -> Callable:
    """Print a dashboard link after optimization."""
    from kernel_code.session import get_dashboard_url

    def _hook(*, cache_session_id: str, **_kw: Any) -> None:
        url = get_dashboard_url(cache_session_id)
        console.print(f"  Dashboard: [cyan]{url}[/cyan]")
        console.print()

    return _hook


def _make_post_optimize_cache_save(file_cache: Any, console: Console) -> Callable:
    """Save reference file hash to cache after optimization completes."""

    def _hook(
        *,
        reference_path: str = "",
        **_kw: Any,
    ) -> None:
        if file_cache is None or not reference_path:
            return
        from pathlib import Path

        ref = Path(reference_path)
        if ref.exists():
            file_cache.track_file(ref)
            logger.info(
                "Cached reference file state: %s (%d eval entries)",
                reference_path,
                file_cache.eval_cache_size,
            )

    return _hook


def _make_pre_optimize_cache_check(file_cache: Any, console: Console) -> Callable:
    """On resume: check if reference file changed since last session."""

    def _hook(
        *,
        config: dict,
        iterations: int,
        reference_path: str = "",
        **_kw: Any,
    ) -> None:
        if file_cache is None or not reference_path:
            return
        from pathlib import Path

        ref = Path(reference_path)
        if not ref.exists():
            return

        if file_cache.has_file_changed(ref):
            console.print(
                "[yellow]Reference file changed since last session -- "
                "eval cache invalidated for new content.[/yellow]"
            )
        else:
            cached_evals = file_cache.eval_cache_size
            if cached_evals > 0:
                console.print(
                    f"[dim]Using cached eval (identical kernel) -- "
                    f"{cached_evals} cached result(s) available.[/dim]"
                )

    return _hook


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_default_hooks(
    session_mod: Any = None,
    skill_library: Any = None,
    template_evolver: Any = None,
    file_cache: Any = None,
    console: Console | None = None,
) -> HookRegistry:
    """Create a HookRegistry with sensible default hooks.

    Args:
        session_mod: The ``kernel_code.session`` module (or any object with a
            ``save_optimization_result`` callable).  If ``None``, the
            ``post_optimize`` save hook is omitted.
        skill_library: A :class:`~openkernel.memory.skill_library.SkillLibrary`
            instance.  If provided, successful iterations record evidence.
        template_evolver: A :class:`~kernel_code.template_evolution.TemplateEvolver`
            instance.  If provided, winning kernels are recorded for template
            evolution (flywheel feedback).
        file_cache: A :class:`~kernel_code.file_cache.FileStateCache` instance.
            If provided, reference file state is tracked and eval results
            are cached across sessions to save GPU costs.
        console: Rich Console to use for output.  Defaults to a new Console.

    Returns:
        A fully-wired :class:`HookRegistry`.
    """
    con = console or Console()
    hooks = HookRegistry()

    # -- pre_optimize -------------------------------------------------------
    hooks.register(HookRegistry.PRE_OPTIMIZE, _make_pre_optimize_log(con))
    hooks.register(HookRegistry.PRE_OPTIMIZE, _make_pre_optimize_cost_confirm(con))
    if file_cache is not None:
        hooks.register(HookRegistry.PRE_OPTIMIZE, _make_pre_optimize_cache_check(file_cache, con))

    # -- post_keep ----------------------------------------------------------
    hooks.register(HookRegistry.POST_KEEP, _make_post_keep_log(con))
    if skill_library is not None:
        hooks.register(HookRegistry.POST_KEEP, _make_post_keep_skill_evidence(skill_library))
    if template_evolver is not None:
        hooks.register(HookRegistry.POST_KEEP, _make_post_keep_evolution(template_evolver))

    # -- post_discard -------------------------------------------------------
    hooks.register(HookRegistry.POST_DISCARD, _make_post_discard_log(con))

    # -- post_optimize ------------------------------------------------------
    if session_mod is not None:
        hooks.register(HookRegistry.POST_OPTIMIZE, _make_post_optimize_save(session_mod))
    hooks.register(HookRegistry.POST_OPTIMIZE, _make_post_optimize_summary(con))
    hooks.register(HookRegistry.POST_OPTIMIZE, _make_post_optimize_dashboard_link(con))
    if file_cache is not None:
        hooks.register(HookRegistry.POST_OPTIMIZE, _make_post_optimize_cache_save(file_cache, con))

    return hooks
