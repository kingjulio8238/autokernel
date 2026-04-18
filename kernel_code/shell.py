"""Interactive kernel optimization shell (REPL).

Provides a persistent interactive environment for kernel optimization,
combining slash commands with natural-language AI queries.

Usage::

    from kernel_code.shell import KernelCodeShell
    shell = KernelCodeShell()
    shell.run()

Or from the CLI::

    kernel-code          # launches the shell
    kernel-code shell    # explicit subcommand
"""

from __future__ import annotations

import asyncio
import json
import shlex
import uuid
import webbrowser
from collections.abc import Callable
from pathlib import Path

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

# prompt_toolkit — optional dependency for polished input with history & completion.
# Falls back to bare input() if not installed.
try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.completion import Completer, Completion
    from prompt_toolkit.formatted_text import HTML
    from prompt_toolkit.key_binding import KeyBindings

    _HAS_PROMPT_TOOLKIT = True
except ImportError:  # pragma: no cover
    _HAS_PROMPT_TOOLKIT = False

# ------------------------------------------------------------------
# Multi-line key bindings (prompt_toolkit)
# ------------------------------------------------------------------

if _HAS_PROMPT_TOOLKIT:
    _kb = KeyBindings()

    @_kb.add('escape', 'enter')  # Alt+Enter to insert newline
    def _alt_enter(event):
        event.current_buffer.insert_text('\n')

    @_kb.add('enter')
    def _enter(event):
        buf = event.current_buffer
        text = buf.text
        # If text ends with colon or backslash, stay in multi-line mode
        if text.rstrip().endswith(':') or text.rstrip().endswith('\\'):
            buf.insert_text('\n')
        else:
            buf.validate_and_handle()

from kernel_code.advisor import AdvisorState, should_advise, get_advice
from kernel_code.agent_loop import AgentLoop
from kernel_code.cost_dashboard import CostTracker
from kernel_code.file_cache import FileStateCache
from kernel_code.git_integration import (
    create_optimization_branch,
    get_current_branch,
    get_optimization_log,
    return_to_original_branch,
)
from kernel_code.messages import ConversationHistory
from kernel_code.errors import format_error
from kernel_code.hooks import HookRegistry, create_default_hooks
from kernel_code.progress import AgentProgress, OptimizationProgress
from kernel_code.onboarding import needs_onboarding, run_onboarding
from kernel_code.permissions import BudgetTracker, confirm_cost, estimate_cost
from kernel_code.settings import (
    KernelCodeSettings,
    load_settings,
    save_project_setting,
    settings_to_config,
    _FIELD_TYPES,
)
from kernel_code.skill_trigger import suggest_skills, format_skill_suggestions
from kernel_code.template_evolution import TemplateEvolver

# Project root -- cache lives at repo root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SESSIONS_DIR = _PROJECT_ROOT / "cache" / "sessions"

# History file for prompt_toolkit (lives alongside other dot-files)
_HISTORY_PATH = _PROJECT_ROOT / ".kernel-code" / "history.txt"


# ------------------------------------------------------------------
# Tab completer (prompt_toolkit)
# ------------------------------------------------------------------

if _HAS_PROMPT_TOOLKIT:

    class KernelCodeCompleter(Completer):
        """Tab completion for kernel-code slash commands and flags."""

        def __init__(self, shell: "KernelCodeShell") -> None:
            self._shell = shell

        def get_completions(self, document, complete_event):
            text = document.text_before_cursor

            # Complete slash commands
            if text.startswith("/"):
                commands = [
                    "/optimize",
                    "/show",
                    "/skills",
                    "/skill:",
                    "/compare",
                    "/compact",
                    "/context",
                    "/diff",
                    "/dashboard",
                    "/git",
                    "/history",
                    "/config",
                    "/evolve",
                    "/cost",
                    "/setup",
                    "/doctor",
                    "/theme",
                    "/help",
                    "/quit",
                    "/exit",
                ]
                for cmd in commands:
                    if cmd.startswith(text):
                        yield Completion(cmd, start_position=-len(text))

            # Complete after /show
            if text.startswith("/show "):
                for sub in ["best", "results", "run"]:
                    full = f"/show {sub}"
                    if full.startswith(text):
                        yield Completion(full, start_position=-len(text))

            # Complete after /skill:
            if text.startswith("/skill:"):
                prefix = text[7:]
                for skill in self._shell._skill_library:
                    sid = skill.get("id", "")
                    if sid.startswith(prefix):
                        yield Completion(f"/skill:{sid}", start_position=-len(text))

            # Complete --flags after /optimize
            if "/optimize" in text and text.endswith("--"):
                for flag in [
                    "--reference",
                    "--backend",
                    "--config",
                    "--parallel",
                    "--mock",
                    "--no-mock",
                    "--iterations",
                    "--gpu",
                    "--git",
                ]:
                    yield Completion(flag, start_position=-2)


class KernelCodeShell:
    """Interactive kernel optimization shell."""

    def __init__(self, session_id: str | None = None) -> None:
        self._console = Console()
        self._session_id = session_id or uuid.uuid4().hex[:8]
        self._runs: list[dict] = []
        self._best_run: dict | None = None
        self._session_data: dict = {}
        self._total_cost: float = 0.0
        self._budget = BudgetTracker()
        self._cost_tracker = CostTracker()
        self._active_skill: dict | None = None
        self._skill_library: list[dict] = self._load_skills()

        # Track whether the session_id was explicitly provided (for resume)
        self._explicit_session_id = session_id is not None

        # File-state cache -- avoids redundant GPU evals and file re-parsing
        self._file_cache = FileStateCache(
            cache_dir=_PROJECT_ROOT / ".kernel-code" / "cache"
        )

        # Template evolution — tracks winning kernels for flywheel feedback
        self._template_evolver = TemplateEvolver(
            evolution_dir=_PROJECT_ROOT / ".kernel-code" / "evolution"
        )

        self._hooks: HookRegistry = create_default_hooks(
            template_evolver=self._template_evolver,
            file_cache=self._file_cache,
            console=self._console,
        )

        # Load hierarchical settings (global -> project -> local)
        self._settings: KernelCodeSettings = load_settings()

        # Apply budget cap from settings
        if self._settings.max_budget is not None:
            self._budget = BudgetTracker(max_budget=self._settings.max_budget)

        # Discover project-level KERNEL.md / kernel.toml config
        from kernel_code.kernel_config import (
            KernelConfig,
            discover_kernel_config,
            inject_config_context,
        )

        self._kernel_config: KernelConfig | None = discover_kernel_config()

        # Progress reporters
        self._opt_progress = OptimizationProgress(console=self._console)
        self._agent_progress = AgentProgress(console=self._console)

        # Typed conversation history for multi-turn context
        self._conversation = ConversationHistory()

        # Agentic loop for natural language (lazy-init on first NL input)
        self._agent_loop: AgentLoop | None = None

        # prompt_toolkit session (history + completion + styled prompt)
        self._prompt_session: PromptSession | None = None  # type: ignore[type-arg]
        if _HAS_PROMPT_TOOLKIT:
            _HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
            self._prompt_session = PromptSession(
                history=FileHistory(str(_HISTORY_PATH)),
                completer=KernelCodeCompleter(self),
                key_bindings=_kb,
                multiline=False,
                prompt_continuation="... ",
            )

        # Command dispatch table
        self._commands: dict[str, Callable[[str], None]] = {
            "/optimize": self._cmd_optimize,
            "/show": self._cmd_show,
            "/compare": self._cmd_compare,
            "/dashboard": self._cmd_dashboard,
            "/git": self._cmd_git,
            "/history": self._cmd_history,
            "/cost": self._cmd_cost,
            "/skills": self._cmd_skills,
            "/compact": self._cmd_compact,
            "/context": self._cmd_context,
            "/diff": self._cmd_diff,
            "/config": self._cmd_config,
            "/setup": self._cmd_setup,
            "/evolve": self._cmd_evolve,
            "/doctor": self._cmd_doctor,
            "/theme": self._cmd_theme,
            "/help": self._cmd_help,
            "/quit": self._cmd_quit,
            "/exit": self._cmd_quit,
        }

    # ------------------------------------------------------------------
    # Session resume
    # ------------------------------------------------------------------

    def _resume_session(self, session_id: str) -> bool:
        """Resume a previous session. Returns True if successful."""
        from kernel_code.session import load_session

        try:
            session = load_session(session_id)
        except FileNotFoundError:
            self._console.print(f"[red]Session {session_id} not found.[/red]")
            return False

        # Restore session state
        self._session_id = session.session_id

        # Restore conversation history
        if session.conversation_history:
            self._conversation = ConversationHistory.from_list(
                session.conversation_history
            )

        # Restore cost tracker
        if hasattr(session, "_cost_data") and session._cost_data:
            self._cost_tracker = CostTracker.from_dict(session._cost_data)

        # Load the last run's session data from cache
        if session.runs:
            last_run = session.runs[-1]
            cache_path = Path(
                f"cache/sessions/{last_run.cache_session_id}.json"
            )
            if cache_path.exists():
                self._session_data = json.loads(cache_path.read_text())

        self._console.print(f"[green]Resumed session {session_id}[/green]")
        if session.runs:
            self._console.print(
                f"  Runs: {len(session.runs)} | "
                f"Last: {session.runs[-1].best_speedup:.2f}x"
            )
        else:
            self._console.print("  No runs yet")
        return True

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Main REPL loop."""
        if needs_onboarding():
            run_onboarding(console=self._console)
            # Reload settings after onboarding creates them
            self._settings = load_settings()
            if self._settings.max_budget is not None:
                self._budget = BudgetTracker(max_budget=self._settings.max_budget)
            # Re-discover kernel config in case KERNEL.md was created
            from kernel_code.kernel_config import discover_kernel_config

            self._kernel_config = discover_kernel_config()

        # Resume session if a session_id was explicitly provided (--resume/--continue)
        resumed = False
        if self._explicit_session_id:
            from kernel_code.session import load_session

            try:
                load_session(self._session_id)
                resumed = self._resume_session(self._session_id)
            except FileNotFoundError:
                # session_id was provided but doesn't exist on disk -- treat as new
                pass

        # If not explicitly resumed, check for a recent session to offer resumption
        if not resumed:
            from kernel_code.session import load_latest_session, should_resume

            if should_resume():
                latest = load_latest_session()
                if latest is not None:
                    num_runs = len(latest.runs)
                    self._console.print(
                        f"[dim]Found recent session "
                        f"[bold]{latest.session_id}[/bold] "
                        f"({num_runs} run{'s' if num_runs != 1 else ''}, "
                        f"backend: {latest.current_backend})[/dim]"
                    )
                    try:
                        answer = (
                            self._console.input(
                                "[dim]Resume this session? (y/n) [/dim]"
                            )
                            .strip()
                            .lower()
                        )
                    except (EOFError, KeyboardInterrupt):
                        answer = "n"
                    if answer in ("y", "yes"):
                        resumed = self._resume_session(latest.session_id)

        self._print_welcome()
        while True:
            user_input = self._prompt()
            if user_input is None:  # EOF / Ctrl-D
                self._console.print()
                break
            self._handle_input(user_input)

    def _prompt(self) -> str | None:
        """Display the prompt and read user input. Returns None on EOF.

        Uses prompt_toolkit when available (gives persistent history, tab
        completion, and styled prompt).  Falls back to Rich console.input()
        if prompt_toolkit is not installed.
        """
        if self._prompt_session is not None:
            try:
                text = self._prompt_session.prompt(
                    HTML("<green>kernel-code</green> <gray>&gt;</gray> ")
                )
                return text
            except EOFError:
                return None
            except KeyboardInterrupt:
                self._console.print()
                return ""

        # Fallback: bare Rich input
        try:
            text = self._console.input(
                "[bold cyan]kernel-code[/bold cyan] [dim]>[/dim] "
            )
            return text
        except EOFError:
            return None
        except KeyboardInterrupt:
            self._console.print()
            return ""

    def _handle_input(self, user_input: str) -> None:
        """Route user input to the appropriate handler."""
        stripped = user_input.strip()
        if not stripped:
            return

        # Record user message in conversation history
        self._conversation.add_user(stripped)

        try:
            if stripped.startswith("/"):
                self._dispatch_command(stripped)
            else:
                self._handle_natural_language(stripped)
        except SystemExit:
            raise
        except Exception as exc:
            format_error(
                exc, context=f"Input: {stripped}", console=self._console
            )

    # ------------------------------------------------------------------
    # Command dispatch
    # ------------------------------------------------------------------

    def _dispatch_command(self, raw: str) -> None:
        """Parse and dispatch a slash command."""
        parts = raw.split(None, 1)
        cmd = parts[0].lower()
        args_str = parts[1] if len(parts) > 1 else ""

        # Handle /skill:NAME prefix
        if cmd.startswith("/skill:"):
            skill_name = cmd[len("/skill:") :]
            self._cmd_skill_load(skill_name)
            return

        handler = self._commands.get(cmd)
        if handler is None:
            self._console.print(
                f"[red]Unknown command:[/red] {escape(cmd)}. "
                "Type [bold]/help[/bold] for available commands."
            )
            return

        handler(args_str)

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    def _cmd_optimize(self, args_str: str) -> None:
        """/optimize --reference FILE [--backend triton|cuda] [--config YAML] [--mock] [--iterations N] [--parallel]"""
        try:
            tokens = shlex.split(args_str)
        except ValueError as exc:
            self._console.print(f"[red]Parse error:[/red] {exc}")
            return

        # Simple argument parsing -- defaults from settings
        reference = None
        backend = self._settings.default_backend
        config_path = None
        mock = True
        iterations = 20
        level = 1
        problem = 23
        parallel = False
        git_enabled = False

        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok in ("--reference", "-r") and i + 1 < len(tokens):
                reference = tokens[i + 1]
                i += 2
            elif tok in ("--backend", "-b") and i + 1 < len(tokens):
                backend = tokens[i + 1]
                i += 2
            elif tok in ("--config", "-c") and i + 1 < len(tokens):
                config_path = tokens[i + 1]
                i += 2
            elif tok in ("--iterations", "-n") and i + 1 < len(tokens):
                iterations = int(tokens[i + 1])
                i += 2
            elif tok == "--mock":
                mock = True
                i += 1
            elif tok == "--no-mock":
                mock = False
                i += 1
            elif tok == "--parallel":
                parallel = True
                i += 1
            elif tok in ("--level",) and i + 1 < len(tokens):
                level = int(tokens[i + 1])
                i += 2
            elif tok in ("--problem",) and i + 1 < len(tokens):
                problem = int(tokens[i + 1])
                i += 2
            elif tok == "--git":
                git_enabled = True
                i += 1
            else:
                # Treat bare argument as reference file if no flag
                if reference is None and not tok.startswith("-"):
                    reference = tok
                i += 1

        if not mock and reference is None:
            self._console.print(
                "[red]Error:[/red] --reference is required in live mode (--no-mock). "
                "Use [bold]/optimize --reference FILE --no-mock[/bold]"
            )
            return

        # Cost gate -- estimate and confirm before running
        # Parallel mode runs two backends, so double the cost estimate
        gpu_type = self._settings.default_gpu
        cost_multiplier = 2 if parallel else 1
        est = estimate_cost(iterations, gpu_type=gpu_type) * cost_multiplier

        if not self._budget.check(est):
            self._console.print(
                f"[red]Budget limit reached.[/red] "
                f"Spent ${self._budget.total_spent:.2f}, "
                f"this run would add ~${est:.2f}."
            )
            return

        # Auto-approve if cost is below the configured threshold
        if est < self._settings.auto_confirm_under:
            pass  # skip confirmation
        elif not confirm_cost(
            est,
            gpu_type,
            iterations * cost_multiplier,
            console=self._console,
        ):
            self._console.print("[dim]Optimization cancelled.[/dim]")
            return

        self._console.print()

        # ------------------------------------------------------------------
        # Git integration: create optimization branch and wire hooks
        # ------------------------------------------------------------------
        git_branch: str | None = None
        original_branch: str | None = None
        if git_enabled:
            original_branch = get_current_branch()
            problem_name = f"L{level}-P{problem}"
            git_branch = create_optimization_branch(problem_name)
            if git_branch:
                self._console.print(
                    f"[bold cyan]Git:[/bold cyan] created branch [green]{git_branch}[/green]"
                )
                # Re-create hooks with git support so kept variants are committed
                self._hooks = create_default_hooks(
                    template_evolver=self._template_evolver,
                    file_cache=self._file_cache,
                    console=self._console,
                    git_enabled=True,
                )
            else:
                self._console.print(
                    "[yellow]Git:[/yellow] could not create branch "
                    "(not a git repo or checkout failed). Continuing without git."
                )
                git_enabled = False

        if parallel:
            self._run_parallel_optimization(
                reference=reference,
                config_path=config_path,
                iterations=iterations,
                mock=mock,
                level=level,
                problem=problem,
            )
        elif mock:
            self._run_mock_optimization(iterations=iterations)
        else:
            self._run_live_optimization(
                reference=reference,
                backend=backend,
                config_path=config_path,
                iterations=iterations,
                level=level,
                problem=problem,
            )

        # ------------------------------------------------------------------
        # Git integration: post-run summary
        # ------------------------------------------------------------------
        if git_enabled and git_branch:
            log = get_optimization_log(max_entries=10)
            self._console.print()
            self._console.print(
                f"[bold cyan]Git:[/bold cyan] optimization history on [green]{git_branch}[/green]"
            )
            self._console.print(f"[dim]{log}[/dim]")
            self._console.print(
                "[dim]Use [bold]/git[/bold] to view the full log, "
                "or switch back manually when ready.[/dim]"
            )

    def _cmd_git(self, args_str: str) -> None:
        """/git -- show optimization branch log."""
        from kernel_code.git_integration import is_git_repo

        if not is_git_repo():
            self._console.print("[dim]Not inside a git repository.[/dim]")
            return

        branch = get_current_branch()
        log = get_optimization_log(max_entries=30)

        self._console.print()
        self._console.print(f"[bold cyan]Branch:[/bold cyan] {branch}")
        self._console.print()
        self._console.print(log)
        self._console.print()

    def _cmd_show(self, args_str: str) -> None:
        """/show best | results | run N"""
        tokens = args_str.strip().split()
        if not tokens:
            self._console.print(
                "[dim]Usage:[/dim] /show best | results | run N"
            )
            return

        sub = tokens[0].lower()

        if sub == "best":
            self._show_best()
        elif sub == "results":
            self._show_results()
        elif sub == "run" and len(tokens) >= 2:
            try:
                run_num = int(tokens[1])
                self._show_run(run_num)
            except ValueError:
                self._console.print(
                    "[red]Error:[/red] run number must be an integer."
                )
        else:
            self._console.print(
                "[dim]Usage:[/dim] /show best | results | run N"
            )

    def _cmd_compare(self, _args_str: str) -> None:
        """/compare -- compare best kernel to baseline."""
        if not self._runs:
            self._console.print(
                "[dim]No optimization runs yet. Run /optimize first.[/dim]"
            )
            return

        self._show_comparison()

    def _cmd_dashboard(self, _args_str: str) -> None:
        """/dashboard -- open browser dashboard for current session."""
        if not self._runs:
            self._console.print(
                "[dim]No optimization runs yet. Run /optimize first.[/dim]"
            )
            return

        url = f"http://localhost:8050/session/{self._session_id}"
        self._console.print(f"[cyan]Opening dashboard:[/cyan] {url}")

        from kernel_code.mock_data import generate_mock_session

        # Ensure session data exists on disk for the dashboard to read
        session_path = _SESSIONS_DIR / f"{self._session_id}.json"
        if not session_path.exists() and self._session_data:
            session_path.parent.mkdir(parents=True, exist_ok=True)
            self._persist_cost_data()
            session_path.write_text(
                json.dumps(self._session_data, indent=2)
            )

        webbrowser.open(url)
        self._console.print(
            "[dim]Dashboard server needs to be running separately:[/dim] "
            "[bold]kernel-code dashboard[/bold]"
        )

    def _cmd_history(self, _args_str: str) -> None:
        """/history -- list all runs in current session."""
        if not self._runs:
            self._console.print(
                "[dim]No optimization runs in this session yet.[/dim]"
            )
            return

        table = Table(
            title=f"Session {self._session_id} -- History",
            show_header=True,
            header_style="bold",
            border_style="dim",
            pad_edge=False,
        )
        table.add_column("#", style="dim", width=4, justify="right")
        table.add_column("Speedup", width=8, justify="right")
        table.add_column("Status", width=10)
        table.add_column("Intent", ratio=1)

        for run in self._runs:
            num = str(run["iteration"])
            speedup = (
                f"{run['speedup']:.2f}x" if run["speedup"] > 0 else "--"
            )
            status = run["status"]
            intent = run.get("intent", "")

            if status == "keep":
                status_str = "[green]keep[/green]"
            elif status == "discard":
                status_str = "[red]discard[/red]"
            elif status in ("compile_error", "error"):
                status_str = "[red]error[/red]"
            elif status == "incorrect":
                status_str = "[yellow]incorrect[/yellow]"
            else:
                status_str = f"[dim]{escape(status)}[/dim]"

            # Highlight best row
            if (
                self._best_run
                and run["iteration"] == self._best_run["iteration"]
            ):
                num = f"[bold green]{num}[/bold green]"
                speedup = f"[bold green]{speedup}[/bold green]"

            table.add_row(num, speedup, status_str, escape(intent))

        self._console.print()
        self._console.print(table)
        self._console.print()

    def _cmd_help(self, _args_str: str) -> None:
        """/help -- show available commands."""
        help_text = Table(
            title="Commands",
            show_header=True,
            header_style="bold",
            border_style="dim",
            pad_edge=False,
        )
        help_text.add_column("Command", style="bold cyan", width=36)
        help_text.add_column("Description", ratio=1)

        commands = [
            (
                "/optimize --reference FILE [opts]",
                "Run kernel optimization",
            ),
            ("  --backend triton|cuda", "  Code-generation backend"),
            ("  --config YAML", "  Configuration file"),
            (
                "  --iterations N",
                "  Number of iterations (default: 20)",
            ),
            (
                "  --mock / --no-mock",
                "  Use mock data (default: --mock)",
            ),
            (
                "  --parallel",
                "  Run both Triton and CUDA, compare results",
            ),
            (
                "  --git",
                "  Track optimization on an openkernel/* git branch",
            ),
            (
                "/show best",
                "Display best kernel with syntax highlighting",
            ),
            (
                "/show results",
                "Display summary table of current session",
            ),
            ("/show run N", "Display details of run N"),
            (
                "/compare",
                "Compare current best to baseline (side-by-side)",
            ),
            ("/compact", "Force conversation compaction"),
            ("/context", "Show context window token breakdown"),
            (
                "/diff",
                "Show diff between reference and best kernel",
            ),
            ("/git", "Show optimization branch git log"),
            (
                "/dashboard",
                "Open browser dashboard for current session",
            ),
            ("/history", "List all runs in current session"),
            (
                "/cost",
                "Show full cost breakdown (LLM + GPU + per-run)",
            ),
            (
                "/config",
                "Show resolved settings (with source files)",
            ),
            ("/config set KEY VALUE", "Update a project setting"),
            (
                "/skills",
                "List all available optimization skills",
            ),
            (
                "/skill:NAME",
                "Load a skill as seed for next /optimize",
            ),
            (
                "/evolve",
                "Show template evolution status for all skills",
            ),
            (
                "/evolve approve SKILL_ID",
                "Approve and apply an evolved template",
            ),
            ("/setup", "Re-run the first-run setup wizard"),
            (
                "/doctor",
                "Check environment health (Modal, API keys, skills, deps)",
            ),
            ("/theme", "Show/change terminal theme"),
            ("/help", "Show this help message"),
            ("/quit, /exit", "Exit the shell"),
        ]
        for cmd, desc in commands:
            help_text.add_row(cmd, desc)

        self._console.print()
        self._console.print(help_text)
        self._console.print()
        self._console.print(
            "[dim]Anything not starting with / is sent as a natural language question.[/dim]"
        )
        self._console.print()

    def _cmd_skills(self, _args_str: str) -> None:
        """/skills -- list all available optimization skills."""
        if not self._skill_library:
            self._console.print(
                "[dim]No skills found in data/skills/.[/dim]"
            )
            return

        self._console.print()
        self._console.print("[bold]Available optimization skills:[/bold]")

        table = Table(
            show_header=True,
            header_style="bold",
            border_style="dim",
            pad_edge=False,
        )
        table.add_column("ID", style="cyan", width=26)
        table.add_column("Name", width=32)
        table.add_column("Backend", width=10)

        for skill in self._skill_library:
            table.add_row(
                skill.get("id", ""),
                skill.get("name", ""),
                skill.get("backend", ""),
            )

        self._console.print(table)
        self._console.print()
        self._console.print(
            "[dim]Use /skill:ID to load a skill (e.g. /skill:triton_tiled_gemm)[/dim]"
        )
        self._console.print()

    def _cmd_skill_load(self, skill_name: str) -> None:
        """/skill:NAME -- load a specific skill into the optimization context."""
        skill_name = skill_name.strip()
        if not skill_name:
            self._console.print(
                "[dim]Usage:[/dim] /skill:NAME  (e.g. /skill:triton_tiled_gemm)"
            )
            return

        # Find by id (exact match)
        match = None
        for skill in self._skill_library:
            if skill.get("id") == skill_name:
                match = skill
                break

        # Fallback: partial match on id or name
        if match is None:
            for skill in self._skill_library:
                sid = skill.get("id", "").lower()
                sname = skill.get("name", "").lower()
                if (
                    skill_name.lower() in sid
                    or skill_name.lower() in sname
                ):
                    match = skill
                    break

        if match is None:
            self._console.print(
                f"[red]Skill not found:[/red] {escape(skill_name)}. "
                "Use [bold]/skills[/bold] to see available skills."
            )
            return

        self._active_skill = match

        self._console.print()
        self._console.print(
            f"[bold green]Loaded:[/bold green] {escape(match.get('name', ''))}"
        )
        self._console.print(
            f"[bold]Trigger:[/bold] {escape(match.get('trigger', ''))}"
        )
        self._console.print(
            f"[bold]Approach:[/bold] {escape(match.get('approach', ''))}"
        )

        template = match.get("code_template")
        if template:
            self._console.print()
            self._console.print("[bold]Template:[/bold]")
            syntax = Syntax(
                template,
                "python",
                theme="monokai",
                line_numbers=True,
                padding=1,
            )
            self._console.print(syntax)
        self._console.print()

    @staticmethod
    def _load_skills() -> list[dict]:
        """Load all skill JSON files from data/skills/."""
        skills_dir = _PROJECT_ROOT / "data" / "skills"
        skills: list[dict] = []
        if not skills_dir.exists():
            return skills
        for path in sorted(skills_dir.glob("*.json")):
            try:
                with open(path) as f:
                    skills.append(json.load(f))
            except (json.JSONDecodeError, OSError):
                continue
        return skills

    def _cmd_config(self, args_str: str) -> None:
        """/config -- show current settings.  /config set KEY VALUE -- update project settings."""
        tokens = args_str.strip().split()

        if not tokens:
            # Show all resolved settings with origins
            self._show_config()
            return

        if tokens[0].lower() == "set" and len(tokens) >= 3:
            key = tokens[1]
            value = " ".join(tokens[2:])
            self._set_config(key, value)
            return

        self._console.print(
            "[dim]Usage:[/dim] /config  or  /config set KEY VALUE"
        )

    def _show_config(self) -> None:
        """Display resolved settings, showing which file each value came from."""
        self._console.print()

        table = Table(
            title="Resolved Settings",
            show_header=True,
            header_style="bold",
            border_style="dim",
            pad_edge=True,
        )
        table.add_column("Setting", style="cyan", width=22)
        table.add_column("Value", width=28)
        table.add_column("Source", style="dim", ratio=1)

        for key in _FIELD_TYPES:
            value = getattr(self._settings, key, None)
            origin = self._settings._origins.get(key, "default")
            value_str = (
                str(value) if value is not None else "[dim]None[/dim]"
            )
            table.add_row(key, value_str, origin)

        self._console.print(table)

        if self._settings.source_files:
            self._console.print()
            self._console.print("[dim]Loaded from:[/dim]")
            for f in self._settings.source_files:
                self._console.print(f"  [dim]{f}[/dim]")

        self._console.print()

    def _set_config(self, key: str, raw_value: str) -> None:
        """Update a single setting in the project settings file."""
        if key not in _FIELD_TYPES:
            self._console.print(
                f"[red]Unknown setting:[/red] {escape(key)}. "
                f"Valid keys: {', '.join(sorted(_FIELD_TYPES))}"
            )
            return

        path = save_project_setting(key, raw_value)

        # Reload settings so the change takes effect immediately
        self._settings = load_settings()
        if self._settings.max_budget is not None:
            self._budget = BudgetTracker(
                max_budget=self._settings.max_budget
            )

        new_value = getattr(self._settings, key, None)
        self._console.print(
            f"[green]Updated:[/green] {key} = {new_value}  "
            f"[dim](saved to {path})[/dim]"
        )

    def _cmd_setup(self, _args_str: str) -> None:
        """/setup -- re-run the first-run onboarding wizard."""
        run_onboarding(console=self._console)
        # Reload settings after onboarding
        self._settings = load_settings()
        if self._settings.max_budget is not None:
            self._budget = BudgetTracker(
                max_budget=self._settings.max_budget
            )
        from kernel_code.kernel_config import discover_kernel_config

        self._kernel_config = discover_kernel_config()

    def _cmd_evolve(self, args_str: str) -> None:
        """/evolve -- show evolution status.  /evolve approve SKILL_ID -- apply evolved template."""
        tokens = args_str.strip().split()

        if not tokens:
            self._show_evolution_status()
            return

        if tokens[0].lower() == "approve" and len(tokens) >= 2:
            skill_id = tokens[1]
            self._approve_evolution(skill_id)
            return

        if tokens[0].lower() == "propose" and len(tokens) >= 2:
            skill_id = tokens[1]
            self._propose_evolution(skill_id)
            return

        self._console.print(
            "[dim]Usage:[/dim] /evolve  |  /evolve approve SKILL_ID  |  /evolve propose SKILL_ID"
        )

    def _show_evolution_status(self) -> None:
        """Display template evolution status for all tracked skills."""
        status = self._template_evolver.get_evolution_status()

        if not status:
            self._console.print(
                "[dim]No evolution data yet. Winning kernels are recorded "
                "automatically during /optimize runs.[/dim]"
            )
            return

        self._console.print()
        self._console.print("[bold]Template Evolution Status[/bold]")
        self._console.print()

        table = Table(
            show_header=True,
            header_style="bold",
            border_style="dim",
            pad_edge=False,
        )
        table.add_column("Skill ID", style="cyan", width=26)
        table.add_column("Wins", width=6, justify="right")
        table.add_column("Avg Speedup", width=12, justify="right")
        table.add_column("Status", width=14)
        table.add_column("Top Patterns", ratio=1)

        for entry in status:
            wins = str(entry["wins"])
            avg_spd = f"{entry['avg_speedup']:.2f}x"
            patterns = (
                ", ".join(entry["top_patterns"][:3])
                if entry["top_patterns"]
                else "[dim]--[/dim]"
            )

            if entry["approved"]:
                status_str = "[green]applied[/green]"
            elif entry["has_proposal"]:
                status_str = "[yellow]pending approval[/yellow]"
            elif entry["ready_to_evolve"]:
                status_str = "[bold cyan]ready[/bold cyan]"
            else:
                remaining = 5 - entry["wins"]
                status_str = f"[dim]{remaining} more win{'s' if remaining != 1 else ''}[/dim]"

            table.add_row(
                entry["skill_id"], wins, avg_spd, status_str, patterns
            )

        self._console.print(table)
        self._console.print()

        # Hint for actionable skills
        ready = [
            e
            for e in status
            if e["ready_to_evolve"] and not e["approved"]
        ]
        if ready:
            if any(e["has_proposal"] for e in ready):
                self._console.print(
                    "[dim]Use [bold]/evolve approve SKILL_ID[/bold] to apply a pending proposal.[/dim]"
                )
            else:
                self._console.print(
                    "[dim]Use [bold]/evolve propose SKILL_ID[/bold] to generate an evolved template.[/dim]"
                )
            self._console.print()

    def _propose_evolution(self, skill_id: str) -> None:
        """Generate an evolved template proposal using the LLM."""
        if not self._template_evolver.should_evolve(skill_id):
            state = self._template_evolver._states.get(skill_id)
            wins = len(state.wins) if state else 0
            self._console.print(
                f"[red]Not enough wins:[/red] {skill_id} has {wins}/5 wins needed."
            )
            return

        # Find the original template from the skill library
        original_template = None
        for skill in self._skill_library:
            if skill.get("id") == skill_id:
                original_template = skill.get("code_template", "")
                break

        self._console.print(
            f"[dim]Generating evolved template for {escape(skill_id)}...[/dim]"
        )

        from openkernel.config import ModelConfig
        from openkernel.llm.provider import LLMProvider

        provider = LLMProvider(ModelConfig())
        evolved = asyncio.run(
            self._template_evolver.propose_evolution(
                skill_id,
                llm_provider=provider,
                original_template=original_template,
            )
        )

        self._console.print()
        self._console.print(
            f"[bold green]Proposed evolution for {escape(skill_id)}:[/bold green]"
        )
        self._console.print()
        syntax = Syntax(
            evolved,
            "python",
            theme="monokai",
            line_numbers=True,
            padding=1,
        )
        self._console.print(syntax)
        self._console.print()
        self._console.print(
            f"[dim]Run [bold]/evolve approve {escape(skill_id)}[/bold] to apply this template.[/dim]"
        )
        self._console.print()

    def _approve_evolution(self, skill_id: str) -> None:
        """Approve and apply an evolved template to the skill library."""
        state = self._template_evolver._states.get(skill_id)
        if state is None or state.evolved_template is None:
            self._console.print(
                f"[red]No pending evolution for {escape(skill_id)}.[/red] "
                f"Run [bold]/evolve propose {escape(skill_id)}[/bold] first."
            )
            return

        # We need a SkillLibrary instance to apply the evolution
        from openkernel.memory.skill_library import SkillLibrary

        lib = SkillLibrary(skills_dir=_PROJECT_ROOT / "data" / "skills")
        lib.load()

        success = self._template_evolver.approve_evolution(skill_id, lib)
        if success:
            # Reload the shell's skill list
            self._skill_library = self._load_skills()
            self._console.print(
                f"[bold green]Applied:[/bold green] Evolved template for "
                f"{escape(skill_id)} saved to data/skills/."
            )
        else:
            self._console.print(
                f"[red]Failed to apply evolution for {escape(skill_id)}.[/red]"
            )

    def _cmd_doctor(self, args_str: str) -> None:
        """Diagnose installation and environment health."""
        import os
        import sys
        from pathlib import Path

        checks = []

        # Python version
        v = sys.version_info
        ok = v >= (3, 10)
        checks.append((
            "Python",
            f"{v.major}.{v.minor}.{v.micro}",
            ok,
            "Need 3.10+" if not ok else "",
        ))

        # Modal
        modal_ok = (
            Path.home().joinpath(".modal.toml").is_file()
            or Path.home().joinpath(".modal").is_dir()
            or bool(os.environ.get("MODAL_TOKEN_ID"))
        )
        checks.append((
            "Modal",
            "authenticated" if modal_ok else "not configured",
            modal_ok,
            "Run: modal setup",
        ))

        # LLM API keys
        groq = bool(os.environ.get("GROQ_API_KEY"))
        minimax = bool(os.environ.get("MINIMAX_API_KEY"))
        anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
        llm_ok = groq or minimax or anthropic
        llm_name = (
            "Groq"
            if groq
            else ("MiniMax" if minimax else ("Anthropic" if anthropic else "none"))
        )
        checks.append((
            "LLM API key",
            llm_name if llm_ok else "not set",
            llm_ok,
            "export GROQ_API_KEY=... (free tier)",
        ))

        # Optional keys
        hf = bool(os.environ.get("HF_TOKEN"))
        checks.append((
            "HF Hub token",
            "set" if hf else "not set (optional)",
            hf or True,
            "export HF_TOKEN=...",
        ))

        # Skills
        skill_count = (
            len(self._skill_library) if self._skill_library else 0
        )
        checks.append((
            "Skills loaded",
            str(skill_count),
            skill_count >= 10,
            f"Expected 10+, got {skill_count}",
        ))

        # KERNEL.md
        km = self._kernel_config is not None
        checks.append((
            "KERNEL.md",
            "found" if km else "not found (optional)",
            km or True,
            "Create KERNEL.md in project root",
        ))

        # Settings
        settings_count = (
            len(self._settings.source_files)
            if hasattr(self._settings, "source_files")
            else 0
        )
        checks.append(
            ("Settings", f"{settings_count} files loaded", True, "")
        )

        # prompt_toolkit
        try:
            import prompt_toolkit

            pt_ok = True
        except ImportError:
            pt_ok = False
        checks.append((
            "prompt_toolkit",
            "installed" if pt_ok else "not installed",
            pt_ok,
            "pip install prompt-toolkit",
        ))

        # Eval cache
        cache_size = (
            self._file_cache.eval_cache_size
            if hasattr(self, "_file_cache") and self._file_cache
            else 0
        )
        checks.append(("Eval cache", f"{cache_size} entries", True, ""))

        # Print results
        table = Table(
            title="Environment Health Check",
            show_header=False,
            padding=(0, 1),
        )
        table.add_column("Status", width=3)
        table.add_column("Component", style="bold")
        table.add_column("Details")
        table.add_column("Fix", style="dim")

        for name, detail, ok, fix in checks:
            status = (
                "[green]\u2713[/green]"
                if ok
                else "[red]\u2717[/red]"
            )
            table.add_row(status, name, detail, fix if not ok else "")

        self._console.print(table)

        all_ok = all(ok for _, _, ok, _ in checks)
        if all_ok:
            self._console.print("\n[green]All checks passed.[/green]")
        else:
            self._console.print(
                f"\n[yellow]{sum(1 for _, _, ok, _ in checks if not ok)} issue(s) found.[/yellow]"
            )

    def _cmd_theme(self, args_str: str) -> None:
        """Show current theme info."""
        self._console.print("[bold]Current theme:[/bold] warm-dark")
        self._console.print(
            "[dim]Based on humanoid-terminal design system[/dim]"
        )
        self._console.print()
        self._console.print("[dim]Themes available:[/dim]")
        self._console.print("  warm-dark [dim](default)[/dim]")
        self._console.print()
        self._console.print(
            "[dim]Custom themes can be configured in .kernel-code/settings.yaml[/dim]"
        )

    def _cmd_cost(self, _args_str: str) -> None:
        """/cost -- show full cost breakdown dashboard."""
        self._cost_tracker.format_dashboard(console=self._console)

    def _cmd_compact(self, _args_str: str) -> None:
        """Force conversation compaction."""
        if not self._conversation or self._conversation.message_count == 0:
            self._console.print("[dim]Nothing to compact.[/dim]")
            return

        before = self._conversation.token_estimate
        summary = self._conversation.compact()
        after = self._conversation.token_estimate

        reduction = (
            ((before - after) / before * 100) if before > 0 else 0
        )
        self._console.print(
            f"[green]Compacted:[/green] {before:,} \u2192 {after:,} tokens ({reduction:.0f}% reduction)"
        )
        self._console.print(
            "[dim]Kept: last 10 messages, optimization events, best kernel[/dim]"
        )

    def _cmd_context(self, _args_str: str) -> None:
        """Show context window breakdown."""
        from kernel_code.compaction import estimate_tokens
        from kernel_code.context_viz import render_context_breakdown
        from kernel_code.kernel_config import inject_config_context

        # Estimate each component
        system_prompt = 800  # rough estimate
        kernel_md = (
            estimate_tokens(inject_config_context(self._kernel_config))
            if self._kernel_config
            else 0
        )
        active_skill = (
            estimate_tokens(str(self._active_skill))
            if self._active_skill
            else 0
        )
        conversation = (
            self._conversation.token_estimate
            if self._conversation
            else 0
        )
        session_iters = estimate_tokens(
            str(self._session_data.get("iterations", []))
        )
        best_kernel = (
            estimate_tokens(
                self._best_run.get("kernel_code_snippet", "")
            )
            if self._best_run
            else 0
        )

        budget = 4096  # default, adjust based on model

        components = [
            ("System prompt", system_prompt),
            ("KERNEL.md", kernel_md),
            ("Active skill", active_skill),
            ("Conversation", conversation),
            ("Session history", session_iters),
            ("Best kernel", best_kernel),
        ]

        # Detect whether conversation has been compacted (summary msg present)
        compacted = (
            self._conversation is not None
            and self._conversation.message_count > 0
            and any(
                (m.content or "").lower().startswith("compacted")
                for m in self._conversation.get_messages()
                if m.role == "system"
            )
        )

        render_context_breakdown(
            components=components,
            budget=budget,
            compacted=compacted,
            console=self._console,
        )

    def _cmd_diff(self, _args_str: str) -> None:
        """Show diff between reference and best optimized kernel."""
        import difflib

        if not self._session_data.get("iterations"):
            self._console.print(
                "[dim]No optimization results. Run /optimize first.[/dim]"
            )
            return

        # Get reference code
        reference = self._session_data.get("reference_code", "")
        if (
            not reference
            and hasattr(self, "_reference_path")
            and self._reference_path
        ):
            try:
                reference = Path(self._reference_path).read_text()
            except Exception:
                pass

        if not reference:
            self._console.print(
                "[dim]Reference code not available.[/dim]"
            )
            return

        # Get best kernel
        best_iter = None
        best_speedup = 0
        for it in self._session_data.get("iterations", []):
            if (
                it.get("decision") == "keep"
                and it.get("speedup", 0) > best_speedup
            ):
                best_speedup = it["speedup"]
                best_iter = it

        if not best_iter or not best_iter.get("kernel_code_snippet"):
            self._console.print(
                "[dim]No optimized kernel available.[/dim]"
            )
            return

        optimized = best_iter["kernel_code_snippet"]

        # Generate diff
        diff = difflib.unified_diff(
            reference.splitlines(keepends=True),
            optimized.splitlines(keepends=True),
            fromfile="reference.py (baseline)",
            tofile=f"optimized.py ({best_speedup:.2f}x)",
        )
        diff_text = "".join(diff)

        if not diff_text:
            self._console.print(
                "[dim]No differences (kernel matches reference).[/dim]"
            )
            return

        self._console.print(Syntax(diff_text, "diff", theme="monokai"))

    def _cmd_quit(self, _args_str: str) -> None:
        """/quit or /exit -- exit the shell."""
        self._console.print("[dim]Goodbye.[/dim]")
        raise SystemExit(0)

    # ------------------------------------------------------------------
    # Natural language
    # ------------------------------------------------------------------

    def _handle_natural_language(self, text: str) -> None:
        """Handle natural language input through the agentic loop.

        The AgentLoop sends the user's question plus session context and
        available tools to the LLM.  The LLM may call tools (up to 3 per
        turn) to gather data before producing a final text answer.

        Uses streaming to display the final response token-by-token.
        Falls back to non-streaming if streaming fails.
        """
        # Lazy-initialise the agent loop so we only create the LLM
        # provider when actually needed.
        if self._agent_loop is None:
            self._agent_loop = AgentLoop(
                session_context=self._session_data,
                console=self._console,
                conversation=self._conversation,
            )
        else:
            # Keep context in sync after new optimization runs
            self._agent_loop.update_context(self._session_data)

        self._console.print()

        answer: str | None = None
        try:
            asyncio.run(self._stream_response(text))
        except Exception:
            # Fall back to non-streaming if streaming fails
            with self._agent_progress.thinking():
                answer = asyncio.run(self._agent_loop.run(text))
            self._console.print(escape(answer))

        # Record assistant response in conversation history
        if answer:
            self._conversation.add_assistant(answer)

        # Sync LLM cost from provider after each turn
        self._sync_llm_cost()

        self._console.print()

    def _sync_llm_cost(self) -> None:
        """Sync cumulative LLM cost from the agent loop's provider into the cost tracker."""
        if self._agent_loop is not None:
            provider = self._agent_loop.provider
            self._cost_tracker.sync_from_provider(
                model_id=provider._config.model_id,
                provider_tokens=provider.tokens_used,
                provider_cost=provider.cost_usd,
            )

    def _persist_cost_data(self) -> None:
        """Embed cost tracker state into session_data for persistence."""
        self._session_data["_cost_tracker"] = self._cost_tracker.to_dict()

    def _restore_cost_data(self) -> None:
        """Restore cost tracker state from session_data after resume."""
        cost_data = self._session_data.get("_cost_tracker")
        if cost_data:
            self._cost_tracker = CostTracker.from_dict(cost_data)

    async def _stream_response(self, text: str) -> None:
        """Run the agent loop with streaming and print tokens as they arrive.

        Shows a spinner while tool calls are being processed.
        Once the final response starts streaming, the spinner is removed and
        tokens are printed directly.
        """
        got_first_token = False
        status = self._agent_progress.thinking()
        status.start()
        try:
            async for token in self._agent_loop.run_stream(text):
                if not got_first_token:
                    # Stop the spinner before printing the first token
                    status.stop()
                    got_first_token = True
                self._console.print(escape(token), end="")
        finally:
            if not got_first_token:
                status.stop()
        if got_first_token:
            # Print a final newline after streaming completes
            self._console.print()

    # ------------------------------------------------------------------
    # Optimization runners
    # ------------------------------------------------------------------

    def _run_mock_optimization(self, iterations: int = 20) -> None:
        """Run optimization with mock data."""
        from kernel_code.mock_data import generate_mock_session

        # Fire pre_optimize hooks
        self._hooks.fire(
            HookRegistry.PRE_OPTIMIZE,
            config={"backend": "mock", "hardware": "mock"},
            iterations=iterations,
        )

        self._console.print(
            "[bold]Running optimization (mock mode)...[/bold]"
        )
        self._console.print()

        session_path = generate_mock_session(
            num_iterations=iterations, session_id=self._session_id
        )

        # Load the generated session data
        self._session_data = json.loads(session_path.read_text())
        self._restore_cost_data()
        iters = self._session_data.get("iterations", [])
        self._runs = iters
        run_cost = estimate_cost(len(iters))
        self._total_cost += run_cost
        self._budget.record(run_cost)

        # Record cost in the cost tracker
        run_num = len(self._cost_tracker._runs) + 1
        gpu_type = self._settings.default_gpu
        avg_eval_seconds = 15.0
        gpu_seconds = len(iters) * avg_eval_seconds
        llm_cost_est = len(iters) * 0.003
        gpu_cost_est = run_cost - llm_cost_est
        self._cost_tracker.record_run_cost(
            run_id=f"#{run_num}",
            llm_cost=llm_cost_est,
            gpu_cost=gpu_cost_est,
            iterations=len(iters),
        )
        # Record GPU usage
        for _ in range(len(iters)):
            self._cost_tracker.record_gpu_eval(gpu_type, avg_eval_seconds)

        # Find the best run and fire per-iteration hooks with progress reporting
        best = None
        best_speedup = 0.0
        for it in iters:
            self._opt_progress.start_iteration(
                it["iteration"], it.get("intent", "")
            )
            if it["status"] == "keep":
                is_new_best = (
                    best is None or it["speedup"] > best["speedup"]
                )
                if is_new_best:
                    best = it
                    best_speedup = it["speedup"]
                self._opt_progress.kept(
                    it["speedup"], is_new_best=is_new_best
                )
                self._conversation.add_optimization_event(
                    "kept",
                    speedup=it["speedup"],
                    iteration=it["iteration"],
                )
                self._hooks.fire(
                    HookRegistry.POST_KEEP,
                    speedup=it["speedup"],
                    iteration=it["iteration"],
                    intent=it.get("intent", ""),
                )
            elif it["status"] == "discard":
                self._opt_progress.discarded(
                    it["speedup"], best_speedup
                )
                self._conversation.add_optimization_event(
                    "discarded",
                    speedup=it["speedup"],
                    iteration=it["iteration"],
                )
                self._hooks.fire(
                    HookRegistry.POST_DISCARD,
                    speedup=it["speedup"],
                    best_speedup=best_speedup,
                    intent=it.get("intent", ""),
                )
            elif it["status"] in ("compile_error", "error"):
                self._opt_progress.error(
                    it["status"], it.get("error", "unknown error")
                )
                self._conversation.add_optimization_event(
                    "error",
                    error=it.get("error", "unknown"),
                    iteration=it["iteration"],
                )
            elif it["status"] == "incorrect":
                self._opt_progress.error(
                    "incorrect", "correctness check failed"
                )
                self._conversation.add_optimization_event(
                    "error",
                    error="correctness check failed",
                    iteration=it["iteration"],
                )
            # Suggest skills based on bottleneck (every 5th iteration to avoid noise)
            profile = it.get("profile", {})
            bn = profile.get("bottleneck_type", "")
            if bn and bn != "unknown" and it["iteration"] % 5 == 0:
                skill_suggestions = suggest_skills(
                    bottleneck_type=bn,
                    problem_description=it.get("intent", ""),
                    skill_library=self._skill_library,
                    top_k=2,
                )
                if skill_suggestions:
                    self._console.print(
                        f"[dim]{format_skill_suggestions(skill_suggestions, bn)}[/dim]"
                    )

            self._hooks.fire(
                HookRegistry.POST_ITERATE,
                iteration=it["iteration"],
                speedup=it["speedup"],
                status=it["status"],
                intent=it.get("intent", ""),
            )
        self._best_run = best

        kept = sum(1 for it in iters if it["status"] == "keep")

        # Report completion
        self._opt_progress.complete(
            best_speedup=self._session_data.get("best_speedup", 0.0),
            iterations=len(iters),
            kept=kept,
            cost=self._total_cost,
        )

        # Launch TUI for visualization
        self._console.print("[dim]Launching TUI...[/dim]")
        self._console.print()

        try:
            from kernel_code.tui.app import KernelCodeApp

            app = KernelCodeApp(session_path=session_path)
            app.run()
        except Exception as exc:
            self._console.print(
                f"[yellow]TUI exited:[/yellow] {escape(str(exc))}"
            )

        # Persist cost tracker data into session
        self._persist_cost_data()

        # Fire post_optimize hooks
        self._hooks.fire(
            HookRegistry.POST_OPTIMIZE,
            best_speedup=self._session_data.get("best_speedup", 0.0),
            iterations_kept=kept,
            iterations_total=len(iters),
            cost=self._total_cost,
            cache_session_id=self._session_id,
        )

        # Post-optimization summary
        self._print_post_optimization_summary(
            best_speedup=self._session_data.get("best_speedup", 0.0),
            kept=kept,
            total=len(iters),
            cost=self._total_cost,
            session_total=self._budget.total_spent,
        )

        # Skill suggestions based on final bottleneck state
        if best:
            final_profile = best.get("profile", {})
            final_bn = final_profile.get("bottleneck_type", "")
            if final_bn and final_bn != "unknown":
                final_suggestions = suggest_skills(
                    bottleneck_type=final_bn,
                    problem_description=best.get("intent", ""),
                    skill_library=self._skill_library,
                    top_k=3,
                )
                if final_suggestions:
                    self._console.print(
                        format_skill_suggestions(
                            final_suggestions, final_bn
                        )
                    )
                    self._console.print()

    def _run_live_optimization(
        self,
        reference: str,
        backend: str,
        config_path: str | None,
        iterations: int,
        level: int,
        problem: int,
    ) -> None:
        """Run optimization with the live openkernel engine."""
        from pathlib import Path as _Path

        ref_path = _Path(reference)
        if not ref_path.exists():
            self._console.print(
                f"[red]Error:[/red] file not found: {escape(reference)}"
            )
            return

        reference_source = ref_path.read_text()

        # Check if reference file changed since last optimization
        ref_changed = self._file_cache.has_file_changed(ref_path)
        if not ref_changed:
            self._console.print(
                "[dim]Reference file unchanged since last run -- "
                "cached eval results will be reused where possible.[/dim]"
            )
        # Track current state regardless
        self._file_cache.track_file(ref_path)

        # Fire pre_optimize hooks
        self._hooks.fire(
            HookRegistry.PRE_OPTIMIZE,
            config={"backend": backend, "hardware": "H100"},
            iterations=iterations,
        )

        self._console.print(
            "[bold]Running optimization (live mode)...[/bold]"
        )
        self._console.print(f"  Reference: {escape(reference)}")
        self._console.print(f"  Backend:   {backend}")
        self._console.print(f"  Iterations: {iterations}")
        if self._file_cache.eval_cache_size > 0:
            self._console.print(
                f"  Eval cache: {self._file_cache.eval_cache_size} entries"
            )
        self._console.print()

        from openkernel.config import (
            Backend as _Backend,
            GpuType,
            ModalConfig,
            OpenKernelConfig,
        )

        # Build config from settings hierarchy, then layer CLI overrides
        settings_kwargs = settings_to_config(self._settings)
        config = OpenKernelConfig(
            **settings_kwargs,
            max_iterations=iterations,
        )
        # CLI --backend overrides settings
        config.backend = (
            _Backend.CUDA if backend == "cuda" else _Backend.TRITON
        )
        if config_path:
            config = OpenKernelConfig.from_yaml(config_path)

        from kernel_code.integration import OpenKernelBridge

        problem_label = f"L{level}#{problem}"
        bridge = OpenKernelBridge(
            config=config,
            session_id=self._session_id,
            problem_label=problem_label,
            hardware="H100",
            backend=backend,
            hooks=self._hooks,
            progress=self._opt_progress,
            file_cache=self._file_cache,
        )

        self._console.print(f"  Session:   {bridge.session_id}")
        self._console.print(f"  Cache:     {bridge.cache_path}")
        self._console.print()

        result = bridge.run_optimization(reference_source)

        # Load the final session data from cache
        if bridge.cache_path.exists():
            self._session_data = json.loads(
                bridge.cache_path.read_text()
            )
            self._runs = self._session_data.get("iterations", [])
            gpu_type = config.modal.gpu_type.value
            run_cost = estimate_cost(
                len(self._runs), gpu_type=gpu_type
            )
            self._total_cost += run_cost
            self._budget.record(run_cost)

            # Record in cost tracker
            run_num = len(self._cost_tracker._runs) + 1
            avg_eval_seconds = 15.0
            llm_cost_est = len(self._runs) * 0.003
            gpu_cost_est = run_cost - llm_cost_est
            self._cost_tracker.record_run_cost(
                run_id=f"#{run_num}",
                llm_cost=llm_cost_est,
                gpu_cost=gpu_cost_est,
                iterations=len(self._runs),
            )
            for _ in range(len(self._runs)):
                self._cost_tracker.record_gpu_eval(
                    gpu_type, avg_eval_seconds
                )

            best = None
            for it in self._runs:
                if it["status"] == "keep":
                    if best is None or it["speedup"] > best["speedup"]:
                        best = it
            self._best_run = best

        kept = sum(
            1 for it in self._runs if it["status"] == "keep"
        )

        # Launch TUI for review
        self._console.print(
            "[dim]Launching TUI for review...[/dim]"
        )
        try:
            from kernel_code.tui.app import KernelCodeApp

            app = KernelCodeApp(session_path=bridge.cache_path)
            app.run()
        except Exception as exc:
            self._console.print(
                f"[yellow]TUI exited:[/yellow] {escape(str(exc))}"
            )

        # Persist cost tracker data into session
        self._persist_cost_data()

        # Fire post_optimize hooks
        self._hooks.fire(
            HookRegistry.POST_OPTIMIZE,
            best_speedup=result.final_speedup,
            iterations_kept=kept,
            iterations_total=len(self._runs),
            cost=self._total_cost,
            cache_session_id=bridge.session_id,
        )

        # Post-optimization summary
        self._print_post_optimization_summary(
            best_speedup=result.final_speedup,
            kept=kept,
            total=len(self._runs),
            cost=self._total_cost,
            session_total=self._budget.total_spent,
        )

        # Save best kernel to file
        if result.final_kernel:
            out_name = f"{ref_path.stem}_optimized.py"
            out_path = _Path.cwd() / out_name
            out_path.write_text(result.final_kernel)
            self._console.print(
                f"[green]Best kernel saved:[/green] {out_name}"
            )

    def _run_parallel_optimization(
        self,
        reference: str | None,
        config_path: str | None,
        iterations: int,
        mock: bool,
        level: int,
        problem: int,
    ) -> None:
        """Run optimization with both Triton and CUDA backends, compare results."""
        from kernel_code.parallel import (
            print_comparison,
            run_parallel_backends,
        )

        self._console.print(
            "[bold]Running parallel backend exploration (Triton + CUDA)...[/bold]"
        )
        self._console.print(
            f"  Mode: {'mock' if mock else 'live'}"
        )
        self._console.print(
            f"  Iterations per backend: {iterations}"
        )
        self._console.print()

        # Build base config for live mode
        config_base = None
        if not mock:
            from openkernel.config import OpenKernelConfig

            if config_path:
                config_base = OpenKernelConfig.from_yaml(config_path)
            else:
                settings_kwargs = settings_to_config(self._settings)
                config_base = OpenKernelConfig(
                    **settings_kwargs,
                    max_iterations=iterations,
                )

        results = run_parallel_backends(
            reference_path=reference or "",
            config_base=config_base,
            iterations=iterations,
            mock=mock,
            console=self._console,
        )

        triton_result = results.get("triton", {})
        cuda_result = results.get("cuda", {})
        winner = results.get("winner", "none")

        # Print comparison table
        if triton_result and cuda_result:
            print_comparison(
                triton_result, cuda_result, self._console
            )

        # Record cost (both backends)
        run_cost = estimate_cost(iterations) * 2
        self._total_cost += run_cost
        self._budget.record(run_cost)

        # Record in cost tracker (parallel = 2 backends)
        gpu_type = self._settings.default_gpu
        avg_eval_seconds = 15.0
        llm_cost_est = iterations * 2 * 0.003
        gpu_cost_est = run_cost - llm_cost_est
        run_num = len(self._cost_tracker._runs) + 1
        self._cost_tracker.record_run_cost(
            run_id=f"#{run_num}",
            llm_cost=llm_cost_est,
            gpu_cost=gpu_cost_est,
            iterations=iterations * 2,
        )
        for _ in range(iterations * 2):
            self._cost_tracker.record_gpu_eval(
                gpu_type, avg_eval_seconds
            )

        # Load the winner's session data into the shell state
        winner_session = results.get("winner_session", {})
        if winner_session:
            self._session_data = winner_session
            self._runs = winner_session.get("iterations", [])

            best = None
            for it in self._runs:
                if it["status"] == "keep":
                    if (
                        best is None
                        or it["speedup"] > best["speedup"]
                    ):
                        best = it
            self._best_run = best

            kept = sum(
                1 for it in self._runs if it["status"] == "keep"
            )

            self._console.print(
                f"[dim]Loaded [bold]{winner}[/bold] results into session "
                f"({kept}/{len(self._runs)} kept, "
                f"best {winner_session.get('best_speedup', 0.0):.2f}x).[/dim]"
            )
            self._console.print()

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def _print_welcome(self) -> None:
        """Print the welcome banner."""
        self._console.print()
        self._console.print(
            "[bold]openkernel[/bold] v0.1 -- interactive kernel optimization shell"
        )
        self._console.print()
        self._console.print(
            f"  Session: [cyan]{self._session_id}[/cyan] (new)"
        )
        if self._settings.source_files:
            self._console.print(
                f"  Settings: [green]{len(self._settings.source_files)} file(s) loaded[/green]"
            )
        if (
            self._kernel_config
            and self._kernel_config.source_path
        ):
            self._console.print(
                f"  Loaded KERNEL.md from: [green]{self._kernel_config.source_path}[/green]"
            )
        self._console.print(
            "  Type [bold]/help[/bold] for commands, or ask a question in natural language."
        )
        self._console.print()

    def _print_post_optimization_summary(
        self,
        best_speedup: float,
        kept: int,
        total: int,
        cost: float,
        session_total: float | None = None,
    ) -> None:
        """Print the post-optimization summary block."""
        self._console.print()

        # Summary line
        summary = Text()
        summary.append("Best: ", style="dim")
        summary.append(f"{best_speedup:.2f}x", style="bold green")
        summary.append(" | ", style="dim")
        summary.append(f"{kept}/{total} kept", style="dim")
        summary.append(" | ", style="dim")
        summary.append(
            self._cost_tracker.format_summary(), style="dim"
        )
        self._console.print(summary)

        # Dashboard link
        url = f"http://localhost:8050/session/{self._session_id}"
        self._console.print(f"  Dashboard: [cyan]{url}[/cyan]")

        # Hint
        self._console.print()
        self._console.print(
            "[dim]Type a question about the results, or /help for commands.[/dim]"
        )
        self._console.print()

    def _show_best(self) -> None:
        """Display the best kernel with syntax highlighting."""
        if self._best_run is None:
            self._console.print(
                "[dim]No best kernel yet. Run /optimize first.[/dim]"
            )
            return

        code = self._best_run.get("kernel_code_snippet", "")
        if not code:
            self._console.print(
                "[dim]Best kernel code not available.[/dim]"
            )
            return

        self._console.print()
        self._console.print(
            f"[bold green]Best kernel[/bold green] "
            f"(run #{self._best_run['iteration']}, "
            f"{self._best_run['speedup']:.2f}x speedup)"
        )
        self._console.print()

        syntax = Syntax(
            code,
            "python",
            theme="monokai",
            line_numbers=True,
            padding=1,
        )
        self._console.print(syntax)
        self._console.print()

    def _show_results(self) -> None:
        """Display summary table of current session."""
        if not self._runs:
            self._console.print(
                "[dim]No results yet. Run /optimize first.[/dim]"
            )
            return

        kept = sum(
            1 for r in self._runs if r["status"] == "keep"
        )
        discarded = sum(
            1 for r in self._runs if r["status"] == "discard"
        )
        errors = sum(
            1
            for r in self._runs
            if r["status"] in ("compile_error", "error")
        )
        best_speedup = self._session_data.get("best_speedup", 0.0)

        self._console.print()

        # Summary panel
        summary_table = Table(
            title=f"Session {self._session_id} -- Results Summary",
            show_header=False,
            border_style="dim",
            pad_edge=True,
        )
        summary_table.add_column("Key", style="dim", width=16)
        summary_table.add_column("Value", ratio=1)

        summary_table.add_row(
            "Total runs", str(len(self._runs))
        )
        summary_table.add_row(
            "Kept", f"[green]{kept}[/green]"
        )
        summary_table.add_row(
            "Discarded", f"[red]{discarded}[/red]"
        )
        summary_table.add_row(
            "Errors",
            f"[red]{errors}[/red]" if errors else "0",
        )
        summary_table.add_row(
            "Best speedup",
            f"[bold green]{best_speedup:.2f}x[/bold green]",
        )
        summary_table.add_row(
            "Est. cost", f"${self._total_cost:.2f}"
        )

        self._console.print(summary_table)
        self._console.print()

    def _show_run(self, run_num: int) -> None:
        """Display details of a specific run."""
        run = None
        for r in self._runs:
            if r["iteration"] == run_num:
                run = r
                break

        if run is None:
            self._console.print(
                f"[red]Run #{run_num} not found.[/red] "
                f"Available: 1-{len(self._runs)}"
            )
            return

        self._console.print()

        # Run detail panel
        detail = Table(
            title=f"Run #{run_num}",
            show_header=False,
            border_style="dim",
            pad_edge=True,
        )
        detail.add_column("Key", style="dim", width=16)
        detail.add_column("Value", ratio=1)

        # Status with color
        status = run["status"]
        if status == "keep":
            status_str = "[green]keep[/green]"
        elif status == "discard":
            status_str = "[red]discard[/red]"
        elif status in ("compile_error", "error"):
            status_str = "[red]error[/red]"
        else:
            status_str = f"[yellow]{escape(status)}[/yellow]"

        detail.add_row("Status", status_str)
        detail.add_row(
            "Speedup",
            f"[bold]{run['speedup']:.2f}x[/bold]"
            if run["speedup"] > 0
            else "--",
        )
        detail.add_row("Intent", escape(run.get("intent", "")))

        if run.get("runtime_us"):
            detail.add_row(
                "Runtime", f"{run['runtime_us']:.1f} us"
            )
        if run.get("ref_runtime_us"):
            detail.add_row(
                "Ref runtime", f"{run['ref_runtime_us']:.1f} us"
            )

        # Profile
        profile = run.get("profile", {})
        if profile:
            bw = profile.get("bandwidth_util", 0)
            cu = profile.get("compute_util", 0)
            ce = profile.get("cache_efficiency", 0)
            occ = profile.get("occupancy", 0)
            bn = profile.get("bottleneck_type", "unknown")

            detail.add_row("Bandwidth util", f"{bw:.0%}")
            detail.add_row("Compute util", f"{cu:.0%}")
            detail.add_row("Cache efficiency", f"{ce:.0%}")
            detail.add_row("Occupancy", f"{occ:.2f}")
            detail.add_row("Bottleneck", escape(bn))

        if run.get("error"):
            detail.add_row(
                "Error", f"[red]{escape(run['error'])}[/red]"
            )

        self._console.print(detail)

        # Kernel code
        code = run.get("kernel_code_snippet", "")
        if code:
            self._console.print()
            syntax = Syntax(
                code,
                "python",
                theme="monokai",
                line_numbers=True,
                padding=1,
            )
            self._console.print(syntax)

        self._console.print()

    def _show_comparison(self) -> None:
        """Compare current best to baseline (side-by-side)."""
        if not self._best_run:
            self._console.print("[dim]No best kernel yet.[/dim]")
            return

        # Find baseline (first keep or first run)
        baseline = None
        for r in self._runs:
            if r["iteration"] == 1 or r.get(
                "intent", ""
            ).startswith("baseline"):
                baseline = r
                break
        if baseline is None:
            baseline = self._runs[0]

        best = self._best_run

        self._console.print()

        # Comparison table
        table = Table(
            title="Baseline vs Best",
            show_header=True,
            header_style="bold",
            border_style="dim",
        )
        table.add_column("Metric", style="dim", width=18)
        table.add_column(
            "Baseline", width=20, justify="right"
        )
        table.add_column("Best", width=20, justify="right")
        table.add_column("Delta", width=14, justify="right")

        # Speedup
        base_spd = baseline.get("speedup", 0.0)
        best_spd = best.get("speedup", 0.0)
        delta_spd = best_spd - base_spd
        delta_color = (
            "green"
            if delta_spd > 0
            else "red" if delta_spd < 0 else "dim"
        )
        table.add_row(
            "Speedup",
            f"{base_spd:.2f}x",
            f"[bold green]{best_spd:.2f}x[/bold green]",
            f"[{delta_color}]+{delta_spd:.2f}x[/{delta_color}]"
            if delta_spd >= 0
            else f"[{delta_color}]{delta_spd:.2f}x[/{delta_color}]",
        )

        # Runtime
        base_rt = baseline.get("runtime_us", 0.0)
        best_rt = best.get("runtime_us", 0.0)
        if base_rt > 0 and best_rt > 0:
            delta_rt = best_rt - base_rt
            delta_color = (
                "green"
                if delta_rt < 0
                else "red" if delta_rt > 0 else "dim"
            )
            table.add_row(
                "Runtime (us)",
                f"{base_rt:.1f}",
                f"{best_rt:.1f}",
                f"[{delta_color}]{delta_rt:+.1f}[/{delta_color}]",
            )

        # Profile metrics
        base_prof = baseline.get("profile", {})
        best_prof = best.get("profile", {})

        for key, label in [
            ("bandwidth_util", "Bandwidth"),
            ("compute_util", "Compute"),
            ("cache_efficiency", "Cache eff."),
            ("occupancy", "Occupancy"),
        ]:
            bv = base_prof.get(key, 0.0)
            bsv = best_prof.get(key, 0.0)
            delta = bsv - bv
            delta_color = (
                "green"
                if delta > 0
                else "red" if delta < 0 else "dim"
            )
            table.add_row(
                label,
                f"{bv:.0%}",
                f"{bsv:.0%}",
                f"[{delta_color}]{delta:+.0%}[/{delta_color}]",
            )

        # Intent
        table.add_row(
            "Intent",
            escape(baseline.get("intent", "")[:30]),
            escape(best.get("intent", "")[:30]),
            "",
        )

        self._console.print(table)

        # Side-by-side code
        base_code = baseline.get("kernel_code_snippet", "")
        best_code = best.get("kernel_code_snippet", "")

        if base_code or best_code:
            self._console.print()

            cols = Table.grid(expand=True)
            cols.add_column(ratio=1)
            cols.add_column(ratio=1)

            left = Syntax(
                base_code or "# (no code)",
                "python",
                theme="monokai",
                line_numbers=True,
            )
            right = Syntax(
                best_code or "# (no code)",
                "python",
                theme="monokai",
                line_numbers=True,
            )

            left_panel = Panel(
                left, title="Baseline", border_style="dim"
            )
            right_panel = Panel(
                right, title="Best", border_style="green"
            )

            cols.add_row(left_panel, right_panel)
            self._console.print(cols)

        self._console.print()
