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
import os
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

from kernel_code.next_steps import (
    NextStep,
    generate_next_steps_llm,
    generate_next_steps_rule_based,
    format_next_steps,
)
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
    inject_api_keys,
    get_configured_providers,
    _FIELD_TYPES,
    _API_KEY_ENV_MAP,
)
from kernel_code.skill_trigger import suggest_skills, format_skill_suggestions
from kernel_code.iteration_formatter import format_iteration_line, format_optimization_header
from kernel_code.summary_card import render_optimization_summary
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
        """Tab completion with descriptions for kernel-code commands."""

        def __init__(self, shell: "KernelCodeShell") -> None:
            self._shell = shell

        def get_completions(self, document, complete_event):
            text = document.text_before_cursor

            # Complete slash commands with descriptions
            if text.startswith("/"):
                commands = [
                    ("/optimize", "Optimize kernel (runs until target or budget)"),
                    ("/show", "Show results (best | results | run N)"),
                    ("/skills", "List optimization skills"),
                    ("/skill:", "Load a skill by name"),
                    ("/compare", "Compare baseline vs optimized"),
                    ("/diff", "Show kernel diff"),
                    ("/compact", "Compact conversation context"),
                    ("/context", "Show context window usage"),
                    ("/dashboard", "Open browser dashboard"),
                    ("/git", "Show git optimization log"),
                    ("/history", "Show run history"),
                    ("/config", "View/edit settings"),
                    ("/profile", "Profile reference kernel (bottleneck analysis)"),
                    ("/roofline", "Show roofline plot (--me, --mem)"),
                    ("/problem", "Load & browse kernel problems"),
                    ("/models", "Browse & select LLM models"),
                    ("/evolve", "Template evolution status"),
                    ("/cost", "Cost breakdown dashboard"),
                    ("/advisor", "Optimization suggestions"),
                    ("/doctor", "Environment health check"),
                    ("/setup", "Re-run onboarding"),
                    ("/theme", "Terminal theme info"),
                    ("/help", "Show all commands"),
                    ("/quit", "Exit kernel code"),
                ]
                for cmd, desc in commands:
                    if cmd.startswith(text):
                        yield Completion(
                            cmd,
                            start_position=-len(text),
                            display_meta=desc,
                        )

            # Complete after /show
            if text.startswith("/show "):
                subs = [
                    ("best", "Best optimized kernel"),
                    ("results", "Summary table + chart"),
                    ("run", "Details of run N"),
                ]
                for sub, desc in subs:
                    full = f"/show {sub}"
                    if full.startswith(text):
                        yield Completion(
                            full,
                            start_position=-len(text),
                            display_meta=desc,
                        )

            # Complete after /skill:
            if text.startswith("/skill:"):
                prefix = text[7:]
                for skill in self._shell._skill_library:
                    sid = skill.get("id", "")
                    sname = skill.get("name", sid)
                    if sid.startswith(prefix):
                        yield Completion(
                            f"/skill:{sid}",
                            start_position=-len(text),
                            display_meta=sname,
                        )

            # Complete @file references — search for files in project
            if "@" in text:
                # Find the @ and the partial path after it
                at_idx = text.rfind("@")
                partial = text[at_idx + 1:]
                if not partial.startswith("/"):
                    # Search for matching files
                    import glob
                    project = str(_PROJECT_ROOT)
                    patterns = [
                        f"{project}/{partial}*.py",
                        f"{project}/**/{partial}*.py",
                    ]
                    seen = set()
                    for pat in patterns:
                        for match in sorted(glob.glob(pat, recursive=True))[:10]:
                            rel = os.path.relpath(match, project)
                            if rel not in seen:
                                seen.add(rel)
                                yield Completion(
                                    f"@{rel}",
                                    start_position=-(len(partial) + 1),
                                    display_meta="file",
                                )

            # Complete --flags after /optimize
            if "/optimize" in text and text.endswith("--"):
                for flag, desc in [
                    ("--reference", "Path to reference kernel"),
                    ("--backend", "triton or cuda"),
                    ("--config", "YAML config file"),
                    ("--parallel", "Try both backends"),
                    ("--mock", "Use mock data (testing)"),
                    ("--iterations", "Max iterations"),
                    ("--gpu", "GPU type (H100/A100/L40S)"),
                    ("--git", "Track with git commits"),
                ]:
                    yield Completion(
                        flag,
                        start_position=-2,
                        display_meta=desc,
                    )


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

        # Inject API keys from settings into environment
        inject_api_keys(self._settings)

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

        # Pending next-step suggestions (set after each optimization)
        self._pending_next_steps: list[NextStep] = []

        # prompt_toolkit session (history + completion + styled prompt)
        self._prompt_session: PromptSession | None = None  # type: ignore[type-arg]
        if _HAS_PROMPT_TOOLKIT:
            _HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
            from prompt_toolkit.styles import Style as PTStyle

            # Style the completion dropdown to match warm theme
            pt_style = PTStyle.from_dict({
                "completion-menu": "bg:#24231f #e0ddd8",
                "completion-menu.completion": "bg:#24231f #e0ddd8",
                "completion-menu.completion.current": "bg:#3d3a36 #4ade80 bold",
                "completion-menu.meta.completion": "bg:#24231f #a09890 italic",
                "completion-menu.meta.completion.current": "bg:#3d3a36 #4ade80 italic",
                "scrollbar.background": "bg:#24231f",
                "scrollbar.button": "bg:#3d3a36",
            })

            self._prompt_session = PromptSession(
                history=FileHistory(str(_HISTORY_PATH)),
                completer=KernelCodeCompleter(self),
                key_bindings=_kb,
                multiline=False,
                prompt_continuation="... ",
                style=pt_style,
                complete_in_thread=True,
            )

        # Command dispatch table
        self._commands: dict[str, Callable[[str], None]] = {
            "/optimize": self._cmd_optimize_unified,
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
            "/models": self._cmd_models,
            "/problem": self._cmd_problem,
            "/profile": self._cmd_profile_ref,
            "/roofline": self._cmd_roofline,
            "/advisor": self._cmd_advisor,
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

        # A2 hero welcome card
        from kernel_code.welcome import render_welcome, detect_hw, pick_motd
        from kernel_code.settings import inject_api_keys as _inject
        _inject(self._settings)
        _hw = detect_hw(self._settings)
        _returning = bool(self._explicit_session_id) or resumed
        _motd = pick_motd(_returning)
        render_welcome(self._console, returning=_returning, hw=_hw, motd=_motd)

        # Status banner (problem, context, skills)
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

    def _resolve_file_tags(self, text: str) -> tuple[str, list[str]]:
        """Resolve @file references — inject file contents into context.

        Returns (processed_text, list_of_loaded_files).
        """
        import re

        tagged_files: list[str] = []
        file_contents: list[str] = []

        # Find all @path references
        for match in re.finditer(r"@([\w./_-]+\.py)", text):
            rel_path = match.group(1)
            full_path = _PROJECT_ROOT / rel_path
            if full_path.is_file():
                content = full_path.read_text()
                tagged_files.append(rel_path)
                file_contents.append(f"\n--- {rel_path} ---\n{content}\n--- end {rel_path} ---\n")

        if not tagged_files:
            return text, []

        # Remove @tags from text and append file contents
        clean_text = re.sub(r"@[\w./_-]+\.py", "", text).strip()
        if file_contents:
            clean_text = clean_text + "\n" + "\n".join(file_contents)

        return clean_text, tagged_files

    def _handle_input(self, user_input: str) -> None:
        """Route user input to the appropriate handler."""
        stripped = user_input.strip()
        if not stripped:
            return

        # Resolve @file tags — inject file content into context
        if "@" in stripped and ".py" in stripped:
            stripped, tagged_files = self._resolve_file_tags(stripped)
            if tagged_files:
                for f in tagged_files:
                    self._console.print(f"  \u23bf  [#999999]Loaded @{f}[/#999999]")

        # Ignore single-char non-command inputs (stray y/n from prompts)
        if len(stripped) <= 2 and stripped.lower() in ("n", "no", "y", "yes"):
            return

        # Record user message in conversation history
        self._conversation.add_user(stripped)

        try:
            # Handle next-step shortcut: "1", "2", or "3" after optimization
            if stripped in ("1", "2", "3") and self._pending_next_steps:
                self._pick_next_step(int(stripped))
                return
            # Smart optimize: "optimize @file.py for H100 2x $10"
            if self._is_optimize_intent(stripped):
                self._smart_optimize(stripped)
                return
            # Detect refinement hint: "try autotune", "use shared memory"
            if self._is_refinement_request(stripped):
                self._handle_refinement(stripped)
                return
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

    def _cmd_optimize_unified(self, args_str: str) -> None:
        """/optimize [@file] [target] [$budget] [--engine native] [--mock]"""
        args = args_str.strip()

        # --mock → old single-round mock path
        if "--mock" in args:
            self._cmd_optimize(args)
            return

        # --engine native → old single-round live path
        if "--engine native" in args:
            self._cmd_optimize(args)
            return

        # Default: smart optimize (autopilot until target)
        # The smart optimize is triggered from _handle_input when text
        # contains "optimize". For /optimize slash command, call it directly.
        self._smart_optimize(f"optimize {args}")

    def _cmd_optimize(self, args_str: str) -> None:
        """/optimize --reference FILE [--backend triton|cuda] [--config YAML] [--iterations N] [--parallel] [--mock]"""
        try:
            tokens = shlex.split(args_str)
        except ValueError as exc:
            self._console.print(f"[red]Parse error:[/red] {exc}")
            return

        # Simple argument parsing -- defaults from settings
        reference = None
        backend = self._settings.default_backend
        config_path = None
        mock = False
        iterations = self._settings.max_rounds
        level = 1
        problem = 23
        parallel = False
        git_enabled = False
        engine = self._settings.engine

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
            elif tok == "--engine" and i + 1 < len(tokens):
                engine = tokens[i + 1]
                i += 2
            else:
                # Treat bare argument as reference file if no flag
                if reference is None and not tok.startswith("-"):
                    reference = tok
                i += 1

        # Auto-detect reference.py if no --reference given
        if not mock and reference is None:
            default_ref = _PROJECT_ROOT / "reference.py"
            if default_ref.is_file():
                reference = str(default_ref)
            else:
                self._console.print(
                    "[red]Error:[/red] no reference file found. "
                    "Load a problem first: [bold]/problem 1.5[/bold] or "
                    "[bold]/problem load FILE[/bold]"
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
        elif engine == "kernel-agent":
            self._run_kernel_agent_optimization(
                reference=reference,
                iterations=iterations,
            )
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
                "/optimize [@file] [target] [$budget]",
                "Optimize kernel — runs until target or budget",
            ),
            ("  Example: /optimize @relu.py 2x $5", ""),
            ("  --engine native", "  Single-round mode (debugging)"),
            (
                "  --mock",
                "  Use mock data for testing",
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
            ("/problem", "Show/load kernel problem (/problem 1.5, /problem load FILE)"),
            ("/models", "Browse & select LLM models"),
            ("/advisor", "Show 3 next optimization suggestions"),
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

    def _has_provider_key(self, provider: str, env_key: str) -> bool:
        """Check if an API key is configured for a provider."""
        import os
        # Check env var (non-empty string)
        env_val = os.environ.get(env_key or "")
        if env_val:
            return True
        # Check settings file
        settings_val = getattr(self._settings, f"{provider}_api_key", None)
        if settings_val:
            return True
        return False

    def _cmd_models(self, args_str: str) -> None:
        """Arrow-key model picker — navigate with up/down, Enter to select, Esc to cancel."""
        import os
        from openkernel.llm.models import load_recommended_models
        from kernel_code.settings import save_api_key

        all_models = load_recommended_models()
        models = [m for m in all_models if m.get("env_key")]
        if not models:
            self._console.print("[#ef4444]No routable models found[/#ef4444]")
            return

        current_model = self._settings.default_model

        # Find initial cursor position (current active model)
        cursor = 0
        for i, m in enumerate(models):
            if m["id"] == current_model:
                cursor = i
                break

        from prompt_toolkit.application import Application
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.layout import Layout
        from prompt_toolkit.layout.containers import Window
        from prompt_toolkit.layout.controls import FormattedTextControl

        selected_idx = [cursor]
        result = [None]

        def _fmt_ctx(ctx: int) -> str:
            if ctx >= 1_000_000:
                return f"{ctx // 1_000_000}M"
            return f"{ctx // 1_000}K"

        def _get_list_text():
            lines: list[tuple[str, str]] = []
            lines.append(("bold", " Models\n\n"))

            # Column layout:
            #   ▸ Name                 Provider   In/Out $/M   Ctx    Key
            lines.append(("#666666", "     "))
            lines.append(("#666666", f"{'Name':<24}{'Provider':<11}{'Cost $/M':<13}{'Context':<9}Key\n"))

            for i, m in enumerate(models):
                name = m.get("name", m["id"])
                provider = m.get("provider", "?")
                env_key = m.get("env_key", "")
                is_cursor = i == selected_idx[0]

                has_key = self._has_provider_key(provider, env_key)

                # Cost columns
                cost_tier = m.get("cost_tier", "")
                if cost_tier == "free":
                    cost_str = "free"
                else:
                    cost_in = m.get("cost_per_m_input", 0)
                    cost_out = m.get("cost_per_m_output", 0)
                    cost_str = f"${cost_in:.1f}/${cost_out:.1f}"

                ctx_str = _fmt_ctx(m.get("context_window", 0))
                key_sym = "\u2713" if has_key else "\u2717"
                key_style = "#4ade80" if has_key else "#ef4444"

                if is_cursor:
                    lines.append(("bold #4ade80", "  \u25b8 "))
                    lines.append(("bold #4ade80", f"{name:<24}"))
                    lines.append(("#4ade80", f"{provider:<11}"))
                    lines.append(("#4ade80", f"{cost_str:<13}"))
                    lines.append(("#4ade80", f"{ctx_str:<9}"))
                    lines.append((key_style + " bold", f"{key_sym}\n"))
                else:
                    lines.append(("", "    "))
                    lines.append(("", f"{name:<24}"))
                    lines.append(("#888888", f"{provider:<11}"))
                    lines.append(("#888888", f"{cost_str:<13}"))
                    lines.append(("#888888", f"{ctx_str:<9}"))
                    lines.append((key_style, f"{key_sym}\n"))

            # Detail panel for highlighted model
            m = models[selected_idx[0]]
            strengths = m.get("strengths", [])
            lines.append(("", "\n"))
            lines.append(("bold", f"  {m.get('name', m['id'])}\n"))
            for s in strengths[:3]:
                lines.append(("#aaaaaa", f"    {s}\n"))

            lines.append(("", "\n"))
            lines.append(("#666666 italic", " \u2191\u2193 navigate  enter select  esc cancel"))
            return lines

        control = FormattedTextControl(_get_list_text)
        kb = KeyBindings()

        @kb.add("up")
        def _up(event):
            selected_idx[0] = (selected_idx[0] - 1) % len(models)

        @kb.add("down")
        def _down(event):
            selected_idx[0] = (selected_idx[0] + 1) % len(models)

        @kb.add("enter")
        def _enter(event):
            result[0] = selected_idx[0]
            event.app.exit()

        @kb.add("escape")
        def _escape(event):
            result[0] = None
            event.app.exit()

        @kb.add("c-c")
        def _ctrl_c(event):
            result[0] = None
            event.app.exit()

        app: Application = Application(
            layout=Layout(Window(control)),
            key_bindings=kb,
            full_screen=False,
        )
        app.run()

        if result[0] is None:
            return

        selected = models[result[0]]
        provider = selected.get("provider", "")
        env_key = selected.get("env_key", "")

        # Check API key — ask if missing
        if not self._has_provider_key(provider, env_key):
            self._console.print()
            self._console.print(
                f"[#fbbf24]No API key for {selected['name']}[/#fbbf24]"
            )
            self._console.print(
                f"[white]Provider: {provider}  |  env var: {env_key}[/white]"
            )
            self._console.print()
            try:
                key = self._console.input(
                    "[bold white]Paste API key: [/bold white]"
                ).strip()
            except (EOFError, KeyboardInterrupt):
                key = ""
            if not key:
                self._console.print("[white]Cancelled[/white]")
                return

            key_setting = f"{provider}_api_key"
            if key_setting in _API_KEY_ENV_MAP:
                path = save_api_key(key_setting, key)
                os.environ[_API_KEY_ENV_MAP[key_setting]] = key
                self._console.print(f"[#4ade80]Saved to {path}[/#4ade80]")
            else:
                os.environ[env_key] = key
                self._console.print("[#4ade80]Key set for this session[/#4ade80]")

        # Apply the model
        save_project_setting("default_model", selected["id"])
        save_project_setting("default_provider", provider)
        self._settings = load_settings()
        inject_api_keys(self._settings)
        # Force agent loop to re-create with the new model on next use
        self._agent_loop = None
        self._console.print(
            f"[#4ade80]Active model \u2192 {selected['name']}[/#4ade80]"
        )

    def _cmd_autopilot(self, args_str: str) -> None:
        """/autopilot — interactive setup then autonomous optimization."""
        from kernel_code.goal_spec import GoalSpec

        self._console.print()
        self._console.print("[bold]Autopilot Setup[/bold]")
        self._console.print("[white]  Configure your optimization goal. Press Enter to accept defaults.[/white]")
        self._console.print()

        def _ask(label: str, default: str, hint: str = "") -> str:
            # Escape default value to prevent Rich markup interpretation
            safe_default = escape(default)
            prompt = f"  [bold white]{label}[/bold white]"
            if hint:
                prompt += f" [#888888]({hint})[/#888888]"
            prompt += f" \\[{safe_default}]: "
            try:
                val = self._console.input(prompt).strip()
            except (EOFError, KeyboardInterrupt):
                return ""
            return val if val else default

        # --- Step 1: Reference file ---
        ref_path = _PROJECT_ROOT / "reference.py"
        ref_problem = ""
        if ref_path.is_file():
            ref_lines = ref_path.read_text().split("\n", 5)
            for line in ref_lines:
                if "Problem" in line or "KernelBench" in line:
                    ref_problem = line.strip().strip('"').strip("'").strip()
                    break

        if ref_problem:
            self._console.print(f"  [#4ade80]Reference:[/#4ade80] {ref_problem}")
        else:
            self._console.print("  [yellow]No reference loaded[/yellow]")

        ref_input = _ask("Reference file", str(ref_path) if ref_path.is_file() else "", "path to PyTorch reference")
        if not ref_input:
            self._console.print("[white]Cancelled[/white]")
            return
        ref_path = _PROJECT_ROOT / ref_input if not ref_input.startswith("/") else __import__("pathlib").Path(ref_input)
        if not ref_path.is_file():
            self._console.print(f"[red]File not found:[/red] {escape(ref_input)}")
            return

        # --- Step 2: Target speedup ---
        target_str = _ask("Target speedup", "2.0", "e.g. 1.5, 2.0, 3.0")
        if not target_str:
            return
        target = float(target_str)

        # --- Step 3: Hardware ---
        hw = _ask("Hardware", self._settings.default_gpu, "L40S, H100, A100-80GB")
        if not hw:
            return

        # --- Step 4: Backend ---
        be = _ask("Backend", self._settings.default_backend, "triton or cuda")
        if not be:
            return

        # --- Step 5: Model ---
        model = self._settings.default_model
        provider = self._settings.default_provider
        model_input = _ask("Model", model, "Enter or /models to change")
        if model_input == "/models":
            self._cmd_models("")
            model = self._settings.default_model
            provider = self._settings.default_provider
        elif model_input:
            model = model_input

        # --- Step 6: Budget ---
        budget_str = _ask("Budget", f"{self._settings.max_budget or 5.00:.2f}", "max USD spend")
        if not budget_str:
            return
        budget = float(budget_str.lstrip("$"))

        # --- Step 7: Time limit ---
        time_str = _ask("Time limit", "none", "e.g. 30m, 1h, or none")
        time_limit = None
        if time_str and time_str != "none":
            if time_str.endswith("m"):
                time_limit = int(time_str[:-1]) * 60
            elif time_str.endswith("h"):
                time_limit = int(time_str[:-1]) * 3600
            else:
                time_limit = int(time_str)

        # --- Step 8: Rounds ---
        rounds_str = _ask("Max rounds", "5", "outer loop iterations")
        if not rounds_str:
            return
        rounds = int(rounds_str)

        # --- Build and validate ---
        goal = GoalSpec(
            target_speedup=target,
            max_budget_usd=budget,
            max_time_seconds=time_limit,
            max_rounds=rounds,
            reference_path=str(ref_path),
            hardware=hw,
            backend=be,
            model=model,
            provider=provider,
        )

        errors = goal.validate()
        if errors:
            for e in errors:
                self._console.print(f"[red]Error:[/red] {e}")
            return

        # --- Confirm ---
        self._console.print()
        self._console.print("[bold]Review[/bold]")
        self._console.print(f"  Reference:  {ref_problem or ref_path.name}")
        self._console.print(f"  Target:     {target}x speedup")
        self._console.print(f"  Hardware:   {hw}  |  Backend: {be}")
        self._console.print(f"  Model:      {model}")
        self._console.print(f"  Budget:     ${budget:.2f}")
        if time_limit:
            self._console.print(f"  Time limit: {time_limit // 60}m")
        self._console.print(f"  Rounds:     up to {rounds}")
        est = estimate_cost(goal.estimated_max_iterations, gpu_type=hw)
        self._console.print(f"  Est. cost:  ~${est:.2f} (up to {goal.estimated_max_iterations} iterations)")
        self._console.print()

        try:
            answer = self._console.input(
                "[bold white]Launch autopilot? (y/n): [/bold white]"
            ).strip().lower()
        except (EOFError, KeyboardInterrupt):
            return
        if answer not in ("y", "yes"):
            self._console.print("[white]Cancelled[/white]")
            return

        # --- Run ---
        from kernel_code.live_display import LiveOptimizationDisplay

        live_display = LiveOptimizationDisplay(
            console=self._console,
            problem=ref_problem or ref_path.name,
            hardware=hw,
            backend=be,
        )
        live_display.set_target(target)

        # Run logger
        from kernel_code.run_log import RunLogger
        run_logger = RunLogger()
        run_logger.start_run(
            command="/optimize",
            config={
                "command": "/optimize",
                "model": model,
                "hardware": hw,
                "backend": be,
                "reference": str(ref_path),
                "target": target,
                "budget": budget,
                "rounds": rounds,
            },
        )

        live_display.start()

        from kernel_code.auto_optimizer import MetaOptimizer

        optimizer = MetaOptimizer(
            goal=goal,
            settings=self._settings,
            console=self._console,
            live_display=live_display,
            run_logger=run_logger,
        )

        try:
            result = optimizer.run()
        finally:
            live_display.finish(
                stop_reason=result.stop_reason if 'result' in dir() else "interrupted"
            )

        # --- Results ---
        self._console.print()
        if result.target_reached:
            self._console.print(
                f"  [bold #4ade80]TARGET REACHED: {result.best_speedup:.2f}x[/bold #4ade80]"
            )
        else:
            self._console.print(
                f"  [white]Best: {result.best_speedup:.2f}x "
                f"(target: {target:.1f}x)[/white]"
            )
        self._console.print(
            f"  [white]{result.rounds_completed} rounds, "
            f"{result.total_iterations} iterations, "
            f"${result.total_cost_usd:.2f}, "
            f"{int(result.elapsed_seconds)}s[/white]"
        )

        if result.best_kernel:
            out_name = "reference_optimized.py"
            out_path = _PROJECT_ROOT / out_name
            out_path.write_text(result.best_kernel)
            self._console.print(
                f"  [#4ade80]Best kernel saved: {out_name}[/#4ade80]"
            )

        # Finalize run log
        run_logger.end_run(
            best_speedup=result.best_speedup,
            best_kernel=result.best_kernel,
            stop_reason=result.stop_reason,
            total_cost=result.total_cost_usd,
        )
        self._console.print(
            f"  [white]Run log:[/white] [#888888]{run_logger.log_path}[/#888888]"
        )
        self._console.print()

    def _cmd_problem(self, args_str: str) -> None:
        """Browse KernelBench problems, load one, or load a custom reference file.

        Usage:
            /problem                — show current problem
            /problem browse [LEVEL] — list KernelBench problems
            /problem LEVEL.ID       — load KernelBench problem (e.g. 1.5)
            /problem load FILE      — load a custom reference file
        """
        args = args_str.strip()

        # /problem — show current
        if not args:
            self._show_current_problem()
            return

        # /problem load FILE — custom reference
        if args.startswith("load "):
            filepath = args[5:].strip()
            self._load_custom_reference(filepath)
            return

        # /problem browse [LEVEL] — list KernelBench problems
        if args.startswith("browse"):
            level_str = args[6:].strip()
            level = int(level_str) if level_str else None
            self._browse_problems(level)
            return

        # /problem LEVEL.ID — load KernelBench problem
        if "." in args:
            try:
                level_s, problem_s = args.split(".", 1)
                level = int(level_s)
                problem_id = int(problem_s)
            except ValueError:
                self._console.print("[#ef4444]Invalid format. Use LEVEL.ID (e.g. 1.5)[/#ef4444]")
                return
            self._load_kernelbench_problem(level, problem_id)
            return

        # /problem N — treat as level to browse
        try:
            level = int(args)
            self._browse_problems(level)
        except ValueError:
            self._console.print(
                "[#ef4444]Unknown subcommand.[/#ef4444] "
                "Try: [bold]/problem browse[/bold], [bold]/problem 1.5[/bold], "
                "or [bold]/problem load FILE[/bold]"
            )

    def _show_current_problem(self) -> None:
        """Show what problem is currently loaded."""
        ref_path = _PROJECT_ROOT / "reference.py"
        if not ref_path.is_file():
            self._console.print("[yellow]No problem loaded.[/yellow]")
            self._console.print()
            self._console.print("  [bold]/problem browse[/bold]      — browse KernelBench problems")
            self._console.print("  [bold]/problem 1.5[/bold]         — load Level 1, Problem 5")
            self._console.print("  [bold]/problem load FILE[/bold]   — load your own reference")
            return

        # Parse problem info from reference.py docstring
        content = ref_path.read_text()
        lines = content.split("\n", 10)
        problem_name = "Custom kernel"
        for line in lines:
            if "KernelBench" in line or "Problem" in line:
                problem_name = line.strip().strip('"').strip("'").strip()
                break

        self._console.print()
        self._console.print(f"  [bold white]{problem_name}[/bold white]")
        self._console.print(f"  [white]Reference: reference.py ({len(content)} chars)[/white]")

        # Show kernel.py status
        kernel_path = _PROJECT_ROOT / "kernel.py"
        if kernel_path.is_file():
            kernel_content = kernel_path.read_text()
            has_model_new = "ModelNew" in kernel_content
            self._console.print(
                f"  [white]Kernel:    kernel.py "
                f"({'ModelNew defined' if has_model_new else 'no ModelNew'})[/white]"
            )

        # Show forward signature
        import re
        match = re.search(r"def forward\(self,(.*?)\).*?:", content, re.DOTALL)
        if match:
            params = match.group(1).strip()
            self._console.print(f"  [white]Signature: forward(self, {params})[/white]")

        self._console.print()
        self._console.print("  Run [bold]/optimize[/bold] to start optimizing this kernel.")

    def _browse_problems(self, level: int | None = None) -> None:
        """List available KernelBench problems."""
        try:
            from kernelbench.dataset import construct_kernelbench_dataset
        except ImportError:
            self._console.print(
                "[#ef4444]kernelbench not installed.[/#ef4444] "
                "Install with: [bold]pip install kernelbench[/bold]"
            )
            return

        levels = [level] if level else [1, 2, 3, 4]

        for lvl in levels:
            try:
                dataset = construct_kernelbench_dataset(level=lvl, source="local")
            except Exception:
                try:
                    dataset = construct_kernelbench_dataset(level=lvl, source="huggingface")
                except Exception as exc:
                    self._console.print(f"[#ef4444]Could not load Level {lvl}: {exc}[/#ef4444]")
                    continue

            problems = dataset.problems if hasattr(dataset, "problems") else []
            self._console.print(f"\n  [bold white]Level {lvl}[/bold white] ({len(problems)} problems)")

            for p in problems[:20]:  # cap display at 20
                pid = p.problem_id if hasattr(p, "problem_id") else "?"
                name = p.name if hasattr(p, "name") else str(p)
                self._console.print(f"    [#4ade80]{lvl}.{pid}[/#4ade80]  {name}")

            if len(problems) > 20:
                self._console.print(f"    [white]... and {len(problems) - 20} more[/white]")

        self._console.print()
        self._console.print("  Load one: [bold]/problem 1.5[/bold]")

    def _load_kernelbench_problem(self, level: int, problem_id: int) -> None:
        """Load a KernelBench problem into reference.py and kernel.py."""
        try:
            from kernelbench.dataset import construct_kernelbench_dataset
        except ImportError:
            self._console.print(
                "[#ef4444]kernelbench not installed.[/#ef4444] "
                "Install with: [bold]pip install kernelbench[/bold]"
            )
            return

        self._console.print(f"[white]Loading Level {level}, Problem {problem_id}...[/white]")

        try:
            dataset = construct_kernelbench_dataset(level=level, source="local")
        except Exception:
            try:
                dataset = construct_kernelbench_dataset(level=level, source="huggingface")
            except Exception as exc:
                self._console.print(f"[#ef4444]Could not load dataset: {exc}[/#ef4444]")
                return

        try:
            problem = dataset.get_problem_by_id(problem_id)
        except Exception as exc:
            self._console.print(f"[#ef4444]Problem {problem_id} not found: {exc}[/#ef4444]")
            return

        # Write reference.py
        header = (
            f'"""\nKernelBench Level {level}, Problem {problem_id}: {problem.name}\n'
            f'READ-ONLY — do not modify this file. The agent modifies kernel.py instead.\n'
            f'Loaded via /problem {level}.{problem_id}\n"""\n\n'
        )
        ref_path = _PROJECT_ROOT / "reference.py"
        ref_path.write_text(header + problem.code)

        # Generate passthrough kernel.py
        from scripts.setup_problem import extract_forward_signature, make_passthrough_kernel
        forward_params = extract_forward_signature(problem.code)
        kernel_code = make_passthrough_kernel(problem.code, forward_params)
        kernel_path = _PROJECT_ROOT / "kernel.py"
        kernel_path.write_text(kernel_code)

        self._console.print(f"[#4ade80]Loaded: {problem.name}[/#4ade80]")
        self._console.print(f"  [white]reference.py — target to beat[/white]")
        self._console.print(f"  [white]kernel.py — passthrough baseline (~1.0x)[/white]")
        self._console.print()
        self._console.print("  Run [bold]/optimize[/bold] to start.")

    def _load_custom_reference(self, filepath: str) -> None:
        """Load a custom reference file for optimization."""
        from pathlib import Path as _Path

        src = _Path(filepath).resolve()
        if not src.is_file():
            self._console.print(f"[#ef4444]File not found: {escape(filepath)}[/#ef4444]")
            return

        content = src.read_text()

        # Detect format
        from kernel_code.problem import detect_format, FORMAT_KERNELBENCH, FORMAT_GPUMODE

        fmt = detect_format(content)

        # Copy to reference.py (preserving original for the engine)
        ref_path = _PROJECT_ROOT / "reference.py"
        ref_path.write_text(content)

        # Also copy adjacent files for GPU Mode format (task.py, utils.py)
        if fmt == FORMAT_GPUMODE:
            for extra in ["task.py", "utils.py"]:
                extra_src = src.parent / extra
                if extra_src.is_file():
                    (_PROJECT_ROOT / extra).write_text(extra_src.read_text())

        fmt_label = {"kernelbench": "KernelBench", "gpumode": "GPU Mode", "custom": "Custom"}.get(fmt, fmt)
        self._console.print(f"  [#d77757]Loaded:[/#d77757] [bold white]{src.name}[/bold white]")
        self._console.print(f"  \u23bf  format: {fmt_label}")
        self._console.print(f"  \u23bf  reference.py updated")
        self._console.print()
        self._console.print("  Run [bold]/optimize[/bold] to start.")

    def _cmd_advisor(self, args_str: str) -> None:
        """Show optimization suggestions based on current session state."""
        if not self._session_data.get("iterations"):
            self._console.print(
                "[yellow]No optimization data yet.[/yellow] "
                "Run [bold]/optimize[/bold] first."
            )
            return

        # Generate fresh suggestions
        steps = generate_next_steps_rule_based(self._session_data)
        try:
            steps = asyncio.run(generate_next_steps_llm(self._session_data))
        except Exception:
            pass  # keep rule-based

        self._pending_next_steps = steps
        self._console.print(format_next_steps(steps))

    def _pick_next_step(self, choice: int) -> None:
        """Handle user picking a next-step suggestion (1, 2, or 3)."""
        if not self._pending_next_steps:
            return
        idx = choice - 1
        if idx < 0 or idx >= len(self._pending_next_steps):
            return

        step = self._pending_next_steps[idx]
        self._pending_next_steps = []  # clear after selection

        self._console.print()
        self._console.print(
            f"[bold #4ade80]Selected:[/bold #4ade80] {step.title}"
        )

        # Load the matching skill if one was suggested
        if step.skill_id:
            self._cmd_skill_load(step.skill_id)

        # Build the optimization intent from the suggestion
        intent = f"{step.title}: {step.approach}"
        self._console.print(
            f"[white]Intent:[/white] {escape(intent)}"
        )
        self._console.print()

        # Re-run optimization with the suggested approach as context
        # Use the same reference file and backend from the last session
        ref = self._session_data.get("reference_file", "")
        backend = self._session_data.get("backend", self._settings.default_backend)
        iters = self._session_data.get("num_iterations", 10)

        if not ref:
            self._console.print(
                "[yellow]No reference file from last run. "
                "Run [bold]/optimize --reference FILE[/bold] instead.[/yellow]"
            )
            return

        self._console.print(
            f"[bold]Starting optimization:[/bold] {step.title}"
        )
        self._cmd_optimize(
            f"--reference {ref} --backend {backend} "
            f"--iterations {iters}"
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

    def _cmd_profile_ref(self, args_str: str) -> None:
        """Profile the reference kernel on Modal — shows runtime, bottleneck, SOL."""
        from pathlib import Path as _Path

        ref_path = _PROJECT_ROOT / "reference.py"
        if not ref_path.is_file():
            self._console.print("  [#ff6b80]No reference.py found. Run /problem first.[/#ff6b80]")
            return

        reference_source = ref_path.read_text()

        # Detect format and make self-contained
        from kernel_code.problem import detect_format, make_self_contained, Problem
        fmt = detect_format(reference_source)

        if fmt == "gpumode":
            p = None
            try:
                from kernel_code.problem import load_problem
                p = load_problem(ref_path)
            except Exception:
                p = Problem(reference_code=reference_source, format=fmt)
            reference_source = make_self_contained(p)

        self._console.print()
        self._console.print("  [bold white]\u2500\u2500 Profiling reference \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500[/bold white]")

        # For KernelBench: run the Model as both ref and kernel (passthrough = 1.0x baseline)
        # For GPU Mode: run ref_kernel as both ref and kernel
        if fmt == "kernelbench":
            # Build a passthrough ModelNew
            import re
            match = re.search(r"def forward\(self,(.*?)\).*?:", reference_source, re.DOTALL)
            params = match.group(1).strip() if match else "*args"
            param_names = ", ".join(p.split(":")[0].strip() for p in params.split(",") if p.strip())
            kernel_source = f'''import torch
import torch.nn as nn

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, {params}) -> torch.Tensor:
        return torch.matmul({param_names})
'''
            pf = "kernelbench"
        else:
            # GPU Mode: use ref_kernel as kernel_function
            kernel_source = reference_source.replace("def ref_kernel(", "def kernel_function(")
            pf = "gpumode"

        self._console.print("  \u23bf  [#999999]Evaluating on Modal...[/#999999]")

        try:
            import modal
            eval_fn = modal.Function.from_name("openkernel-eval", "eval_kernel_on_gpu")
            result = eval_fn.remote(
                kernel_source=kernel_source,
                reference_source=reference_source,
                eval_mode="fast",
                problem_format=pf,
            )
        except Exception as exc:
            self._console.print(f"  [#ff6b80]Profile failed: {exc}[/#ff6b80]")
            return

        # Store profile for use in subsequent /optimize
        self._last_profile = result

        # Display profile
        from kernel_code.kernel_profile import render_kernel_profile
        render_kernel_profile(
            speedup=result.get("speedup", 0.0),
            ref_runtime_us=result.get("ref_runtime_us", 0.0),
            kernel_runtime_us=result.get("runtime_us", 0.0),
            profile=result.get("profile", {}),
            hardware=self._settings.default_gpu,
            console=self._console,
        )

        # Show guidance
        ref_us = result.get("ref_runtime_us", 0.0)
        if ref_us > 0:
            self._console.print(f"  \u23bf  [#999999]Reference runtime: {ref_us:.0f}\u03bcs on {self._settings.default_gpu}[/#999999]")
            self._console.print(f"  \u23bf  [#999999]Run /optimize to generate a faster kernel[/#999999]")
        self._console.print()

    def _cmd_roofline(self, args_str: str) -> None:
        """/roofline [--me] [--mem] — show roofline plot."""
        from kernel_code.roofline_view import render_roofline

        args = args_str.strip().lower()
        if "--me" in args:
            # Show user's best kernel on the roofline
            best = self._best_run.get("speedup", 0.0) if self._best_run else 0.0
            render_roofline(
                self._console, view="me",
                user_speedup=best, user_label="your kernel",
                hardware=self._settings.default_gpu,
            )
        elif "--mem" in args:
            render_roofline(
                self._console, view="mem",
                hardware=self._settings.default_gpu,
            )
        else:
            render_roofline(
                self._console, view="compute",
                hardware=self._settings.default_gpu,
            )

    def _cmd_diff(self, _args_str: str) -> None:
        """Show diff between reference and optimized kernel."""
        import difflib
        from rich.syntax import Syntax

        ref_path = _PROJECT_ROOT / "reference.py"
        opt_path = _PROJECT_ROOT / "reference_optimized.py"

        # Try files first (most reliable)
        if ref_path.is_file() and opt_path.is_file():
            reference = ref_path.read_text()
            optimized = opt_path.read_text()
        else:
            self._console.print(
                "  [#999999]No optimized kernel found. Run /optimize first.[/#999999]"
            )
            return

        diff = difflib.unified_diff(
            reference.splitlines(keepends=True),
            optimized.splitlines(keepends=True),
            fromfile="reference.py",
            tofile="reference_optimized.py",
        )
        diff_text = "".join(diff)

        if not diff_text:
            self._console.print(
                "  [#999999]No differences (kernel matches reference).[/#999999]"
            )
            return

        self._console.print()
        self._console.print(
            "  [bold white]\u2500\u2500 Diff: reference \u2192 optimized \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500[/bold white]"
        )
        self._console.print(Syntax(diff_text, "diff", theme="monokai"))
        self._console.print()

    def _cmd_quit(self, _args_str: str) -> None:
        """/quit or /exit -- exit the shell."""
        self._console.print("[dim]Goodbye.[/dim]")
        raise SystemExit(0)

    # ------------------------------------------------------------------
    # Natural language
    # ------------------------------------------------------------------

    def _is_optimize_intent(self, text: str) -> bool:
        """Detect if the user wants to optimize something."""
        lower = text.lower()
        triggers = [
            "optimize", "make faster", "speed up", "improve performance",
            "make this faster", "optimize this", "kernel for",
        ]
        return any(t in lower for t in triggers)

    def _is_refinement_request(self, text: str) -> bool:
        """Detect if the user wants to refine the existing kernel."""
        opt_path = _PROJECT_ROOT / "reference_optimized.py"
        if not opt_path.is_file():
            return False
        refinement_hints = [
            "try ", "use ", "add ", "increase ", "decrease ", "change ",
            "autotune", "shared memory", "block size", "vectorize",
            "tile", "fuse", "tensor core", "coalesce", "unroll",
        ]
        lower = text.lower()
        return any(h in lower for h in refinement_hints)

    def _smart_optimize(self, text: str) -> None:
        """Smart optimizer — parse goal from NL, validate, ask for missing, run.

        Handles: "optimize @file.py for H100 2x $10"
        Also: "optimize" (uses defaults), "optimize @file.py" (fills rest from settings)
        """
        from kernel_code.goal_parser import parse_goal, validate_goal

        self._console.print()

        # Parse what the user gave us
        goal = parse_goal(text)

        # Suppress asyncio noise from litellm
        import logging as _logging
        _logging.getLogger("asyncio").setLevel(_logging.CRITICAL)
        _logging.getLogger("openkernel.llm.provider").setLevel(_logging.CRITICAL)

        # Fill from settings only what's truly implicit
        if not goal.file:
            ref_path = _PROJECT_ROOT / "reference.py"
            if ref_path.is_file():
                goal.file = "reference.py"
        if not goal.backend:
            goal.backend = self._settings.default_backend
        if not goal.model:
            goal.model = self._settings.default_model

        # --- Ask for what wasn't explicitly stated ---
        def _ask(label: str, default: str, hint: str = "") -> str:
            safe = escape(default)
            prompt = f"  [bold white]{label}[/bold white]"
            if hint:
                prompt += f" [#999999]({hint})[/#999999]"
            prompt += f" \\[{safe}]: "
            try:
                val = self._console.input(prompt).strip()
            except (EOFError, KeyboardInterrupt):
                return ""
            return val if val else default

        # File
        if not goal.file or not (_PROJECT_ROOT / goal.file).is_file():
            goal.file = _ask("Kernel file", "", "path to .py file")
            if not goal.file:
                return

        # Target speedup
        if "target" not in goal.explicit:
            val = _ask("Target speedup", "2.0", "e.g. 1.5, 2.0, 3.0")
            if not val:
                return
            goal.target_speedup = float(val)

        # Hardware
        if "hardware" not in goal.explicit:
            val = _ask("Hardware", self._settings.default_gpu, "L40S, H100, A100-80GB")
            if not val:
                return
            goal.hardware = val

        # Budget
        if "budget" not in goal.explicit:
            default_budget = f"{self._settings.max_budget or 5.00:.2f}"
            val = _ask("Budget", default_budget, "max USD spend")
            if not val:
                return
            goal.budget_usd = float(val.lstrip("$"))

        # Time limit
        if "time" not in goal.explicit:
            val = _ask("Time limit", "none", "e.g. 30m, 1h, or none")
            if val and val != "none":
                if val.endswith("m"):
                    goal.time_limit_seconds = int(val[:-1]) * 60
                elif val.endswith("h"):
                    goal.time_limit_seconds = int(val[:-1]) * 3600

        # --- Validate ---
        from kernel_code.goal_parser import validate_goal
        errors = validate_goal(goal, _PROJECT_ROOT)
        if errors:
            for e in errors:
                self._console.print(f"  [#ff6b80]{e}[/#ff6b80]")
            return

        # --- Show plan + confirm ---
        from kernel_code.problem import load_problem, detect_format
        ref_path = _PROJECT_ROOT / goal.file
        try:
            problem = load_problem(ref_path)
            problem_name = problem.name or ref_path.name
        except Exception:
            problem_name = ref_path.name

        self._console.print()
        self._console.print(f"  [bold white]\u2500\u2500 Optimization Plan \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500[/bold white]")
        self._console.print(f"  File:     [bold white]{problem_name}[/bold white]")
        self._console.print(f"  Target:   [bold white]{goal.target_speedup:.1f}x[/bold white] speedup")
        self._console.print(f"  Hardware: [white]{goal.hardware}[/white]  \u00b7  Backend: [white]{goal.backend}[/white]")
        self._console.print(f"  Model:    [white]{goal.model}[/white]")
        self._console.print(f"  Budget:   [white]${goal.budget_usd:.2f}[/white]")
        if goal.time_limit_seconds:
            self._console.print(f"  Time:     [white]{goal.time_limit_seconds // 60}m[/white]")
        self._console.print()

        try:
            answer = self._console.input(
                "  [bold white]Proceed? (y/n): [/bold white]"
            ).strip().lower()
        except (EOFError, KeyboardInterrupt):
            return
        if answer not in ("y", "yes"):
            self._console.print("  [#999999]Cancelled[/#999999]")
            return

        # --- Save as reference.py if it's a different file ---
        if goal.file != "reference.py":
            src = (_PROJECT_ROOT / goal.file).read_text()
            (_PROJECT_ROOT / "reference.py").write_text(src)
            # Copy supporting files for GPU Mode
            fmt = detect_format(src)
            if fmt == "gpumode":
                src_dir = (_PROJECT_ROOT / goal.file).parent
                for extra in ["task.py", "utils.py"]:
                    ep = src_dir / extra
                    if ep.is_file():
                        (_PROJECT_ROOT / extra).write_text(ep.read_text())
            self._console.print(f"  \u23bf  [#999999]Loaded {goal.file} \u2192 reference.py[/#999999]")

        # --- Apply settings ---
        if goal.hardware != self._settings.default_gpu:
            from kernel_code.settings import save_project_setting
            save_project_setting("default_gpu", goal.hardware)
            self._settings.default_gpu = goal.hardware

        # --- Auto-profile ---
        self._console.print(f"  \u23bf  [#999999]Profiling reference...[/#999999]")
        self._cmd_profile_ref("")

        # --- Run autopilot ---
        from kernel_code.goal_spec import GoalSpec
        from kernel_code.auto_optimizer import MetaOptimizer
        from kernel_code.live_display import LiveOptimizationDisplay

        spec = GoalSpec(
            target_speedup=goal.target_speedup,
            max_budget_usd=goal.budget_usd,
            max_time_seconds=goal.time_limit_seconds or None,
            max_rounds=self._settings.max_autopilot_rounds,
            reference_path=str(_PROJECT_ROOT / "reference.py"),
            hardware=goal.hardware,
            backend=goal.backend,
            model=goal.model,
            provider=self._settings.default_provider,
        )

        live_display = LiveOptimizationDisplay(
            console=self._console,
            problem=problem_name,
            hardware=goal.hardware,
            backend=goal.backend,
        )
        live_display.set_target(goal.target_speedup)

        from kernel_code.run_log import RunLogger
        run_logger = RunLogger()
        run_logger.start_run(
            command=f"smart optimize: {text[:60]}",
            config={
                "model": goal.model, "hardware": goal.hardware,
                "backend": goal.backend, "target": goal.target_speedup,
                "budget": goal.budget_usd, "file": goal.file,
            },
        )

        live_display.start()
        optimizer = MetaOptimizer(
            goal=spec,
            settings=self._settings,
            console=self._console,
            live_display=live_display,
            run_logger=run_logger,
        )

        try:
            result = optimizer.run()
        finally:
            live_display.finish(
                stop_reason=result.stop_reason if 'result' in dir() else "interrupted"
            )

        # --- Results ---
        from kernel_code.kernel_profile import render_kernel_profile
        render_kernel_profile(
            speedup=result.best_speedup,
            console=self._console,
        )

        if result.target_reached:
            self._console.print(
                f"  [bold #4eba65]TARGET REACHED: {result.best_speedup:.2f}x[/bold #4eba65]"
            )
        else:
            self._console.print(
                f"  [white]Best: {result.best_speedup:.2f}x (target: {goal.target_speedup:.1f}x)[/white]"
            )

        if result.best_kernel:
            out_path = _PROJECT_ROOT / "reference_optimized.py"
            out_path.write_text(result.best_kernel)
            self._console.print(f"  [#4eba65]Saved: reference_optimized.py[/#4eba65]")

        run_logger.end_run(
            best_speedup=result.best_speedup,
            best_kernel=result.best_kernel,
            stop_reason=result.stop_reason,
            total_cost=result.total_cost_usd,
        )

        self._console.print(f"  \u23bf  [#999999]/diff to see changes[/#999999]")
        self._console.print(f"  \u23bf  [#999999]Run log: {run_logger.log_path}[/#999999]")
        self._console.print()

    def _handle_inline_optimize(self, text: str) -> None:
        """Handle pasted code — wrap in reference format and optimize."""
        import re

        self._console.print()

        # Extract code from @file injection or pasted text
        code = text

        # Check for injected @file content: "--- path ---\ncontent\n--- end path ---"
        file_match = re.search(r"--- (\S+) ---\n(.*?)\n--- end \1 ---", code, re.DOTALL)
        if file_match:
            code = file_match.group(2).strip()
        else:
            # Extract after trigger phrase
            for trigger in ["optimize this:", "optimize:", "make this faster:", "speed up:"]:
                if trigger in code.lower():
                    idx = code.lower().index(trigger) + len(trigger)
                    code = code[idx:]
                    break

            # Strip markdown fences
            match = re.search(r"```(?:python)?\s*\n?(.*?)```", code, re.DOTALL)
            if match:
                code = match.group(1)

        code = code.strip()

        if not code:
            self._console.print("[#ff6b80]No code detected to optimize.[/#ff6b80]")
            return

        # Detect if it's a full reference (has class Model) or a snippet
        from kernel_code.problem import detect_format
        fmt = detect_format(code)

        if "class Model" not in code and "def ref_kernel" not in code:
            # Wrap bare function in a Model class
            # Try to extract function signature
            func_match = re.search(r"def (\w+)\((.*?)\).*?:", code)
            if func_match:
                func_name = func_match.group(1)
                params = func_match.group(2)
                # Build a reference wrapper
                code = f'''import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    {code}

# Default inputs — adjust sizes as needed
def get_inputs():
    return [torch.randn(1024, 1024, device="cuda")]

def get_init_inputs():
    return []
'''
                self._console.print(f"  \u23bf  [#999999]Wrapped as Model with forward()[/#999999]")
            else:
                self._console.print("[#ff6b80]Could not parse function from pasted code.[/#ff6b80]")
                return

        # Save as reference.py
        ref_path = _PROJECT_ROOT / "reference.py"
        ref_path.write_text(code)
        self._console.print(f"  \u23bf  [#999999]Saved to reference.py[/#999999]")
        self._console.print()

        # Run optimization
        self._cmd_optimize("")

    def _handle_refinement(self, hint: str) -> None:
        """Refine the existing optimized kernel with a hint from the KE."""
        from pathlib import Path as _Path

        opt_path = _PROJECT_ROOT / "reference_optimized.py"
        ref_path = _PROJECT_ROOT / "reference.py"

        if not opt_path.is_file() or not ref_path.is_file():
            self._console.print("  [#999999]No optimized kernel to refine. Run /optimize first.[/#999999]")
            return

        current_kernel = opt_path.read_text()
        reference = ref_path.read_text()

        self._console.print()
        self._console.print(f"  [bold white]\u2500\u2500 Refining kernel \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500[/bold white]")
        self._console.print(f"  \u23bf  Hint: {hint[:60]}")
        self._console.print()

        # Build a refinement prompt that includes the current kernel
        from kernel_code.problem import detect_format, make_self_contained, Problem, load_problem

        fmt = detect_format(reference)
        if fmt == "gpumode":
            try:
                p = load_problem(ref_path)
                reference = make_self_contained(p)
            except Exception:
                pass

        from kernel_code.integration.kernel_agent_bridge import KernelAgentBridge
        from kernel_code.live_display import LiveOptimizationDisplay

        # Modify the reference to include the current kernel + refinement hint
        refinement_ref = (
            f"# CURRENT KERNEL TO REFINE:\n"
            f"# {hint}\n"
            f"#\n"
            f"# Improve this kernel based on the hint above.\n"
            f"# Keep what works, fix what doesn't.\n\n"
            f"{reference}\n\n"
            f"# === CURRENT KERNEL (to be improved) ===\n"
            f"# {hint}\n"
            f"{current_kernel}"
        )

        live_display = LiveOptimizationDisplay(
            console=self._console,
            problem=f"Refining: {hint[:40]}",
            hardware=self._settings.default_gpu,
            backend=self._settings.default_backend,
        )

        bridge = KernelAgentBridge(
            reference_source=reference,
            model_name=self._settings.default_model,
            num_workers=self._settings.num_workers,
            max_rounds=self._settings.max_rounds,
            hardware=self._settings.default_gpu,
            live_display=live_display,
        )

        result = {}
        live_display.start()
        try:
            result = bridge.run()
        finally:
            stop = "improved" if result.get("success") else "no improvement found"
            live_display.finish(stop_reason=stop)

        speedup = result.get("speedup", 0.0)
        kernel_code = result.get("kernel_code", "")

        from kernel_code.kernel_profile import render_kernel_profile
        render_kernel_profile(
            speedup=speedup,
            ref_runtime_us=result.get("ref_runtime_us", 0.0),
            kernel_runtime_us=result.get("kernel_runtime_us", 0.0),
            profile=result.get("profile", {}),
            hardware=self._settings.default_gpu,
            console=self._console,
        )

        if result.get("success") and kernel_code:
            from kernel_agent.model_wrapper import wrap_in_model_new
            if fmt == "kernelbench":
                kernel_code = wrap_in_model_new(kernel_code, reference)
            opt_path.write_text(kernel_code)
            self._console.print(f"  [#4eba65]Refined kernel saved: reference_optimized.py[/#4eba65]")
        else:
            self._console.print(f"  [#999999]No improvement found. Current kernel unchanged.[/#999999]")
        self._console.print()

    def _handle_natural_language(self, text: str) -> None:
        """Handle natural language input through the agentic loop.

        The AgentLoop sends the user's question plus session context and
        available tools to the LLM.  The LLM may call tools (up to 3 per
        turn) to gather data before producing a final text answer.

        Uses streaming to display the final response token-by-token.
        Falls back to non-streaming if streaming fails.
        """
        # Lazy-initialise the agent loop so we only create the LLM
        # provider when actually needed.  Build ModelConfig from current
        # settings so the user's /models selection is respected.
        if self._agent_loop is None:
            # Ensure API keys from settings are in env before creating provider
            inject_api_keys(self._settings)
            from openkernel.config import ModelConfig
            model_config = ModelConfig(
                provider=self._settings.default_provider,
                model_id=self._settings.default_model,
            )
            self._agent_loop = AgentLoop(
                session_context=self._session_data,
                model_config=model_config,
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
        except Exception as stream_exc:
            # Fall back to non-streaming if streaming fails
            try:
                with self._agent_progress.thinking():
                    answer = asyncio.run(self._agent_loop.run(text))
                self._console.print(escape(answer))
            except Exception as exc:
                format_error(exc, context="LLM call", console=self._console)

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
        import time as _time

        from kernel_code.mock_data import generate_mock_session

        start_time = _time.time()

        # Fire pre_optimize hooks (silently -- we print our own header)
        self._console.quiet = True
        self._hooks.fire(
            HookRegistry.PRE_OPTIMIZE,
            config={"backend": "mock", "hardware": "mock"},
            iterations=iterations,
        )
        self._console.quiet = False

        # Print compact header
        estimated_cost = estimate_cost(iterations)
        self._console.print(format_optimization_header(
            iterations, backend="mock", hardware="mock",
            estimated_cost=estimated_cost,
        ))

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

        # --- Compact iteration loop ---
        # One line per iteration; hooks fire silently for state tracking.
        best = None
        best_speedup = 0.0
        best_iter_num = 0
        last_bottleneck = ""
        last_profile: dict | None = None
        suggestions: list[dict] | None = None

        for it in iters:
            speedup = it.get("speedup", 0)
            status = it.get("decision", it.get("status", ""))
            intent = it.get("intent", "")
            iter_num = it.get("iteration", 0)

            is_new_best = status == "keep" and (
                best is None or speedup > best_speedup
            )
            if is_new_best:
                best = it
                best_speedup = speedup
                best_iter_num = iter_num

            # Print compact one-liner
            self._console.print(format_iteration_line(
                iter_num, speedup, status, intent, is_new_best,
            ))

            # Track conversation events (silent)
            if status == "keep":
                self._conversation.add_optimization_event(
                    "kept", speedup=speedup, iteration=iter_num,
                )
            elif status == "discard":
                self._conversation.add_optimization_event(
                    "discarded", speedup=speedup, iteration=iter_num,
                )
            elif status in ("compile_error", "error", "incorrect"):
                self._conversation.add_optimization_event(
                    "error",
                    error=it.get("error", status),
                    iteration=iter_num,
                )

            # Track bottleneck / profile for summary card
            profile = it.get("profile", {})
            bn = profile.get("bottleneck_type", "")
            if bn and bn != "unknown":
                last_bottleneck = bn
                last_profile = profile

            # Fire hooks silently (state tracking: advisor, evolution, etc.)
            self._console.quiet = True
            if status == "keep":
                self._hooks.fire(
                    HookRegistry.POST_KEEP,
                    speedup=speedup,
                    iteration=iter_num,
                    intent=intent,
                )
            elif status == "discard":
                self._hooks.fire(
                    HookRegistry.POST_DISCARD,
                    speedup=speedup,
                    best_speedup=best_speedup,
                    intent=intent,
                )
            self._hooks.fire(
                HookRegistry.POST_ITERATE,
                iteration=iter_num,
                speedup=speedup,
                status=status,
                intent=intent,
            )
            self._console.quiet = False

        self._best_run = best
        kept = sum(1 for it in iters if it["status"] == "keep")

        # Compute final skill suggestions (once, for summary card)
        if best:
            final_profile = best.get("profile", {})
            final_bn = final_profile.get("bottleneck_type", "")
            if final_bn and final_bn != "unknown":
                last_bottleneck = final_bn
                last_profile = final_profile
                suggestions = suggest_skills(
                    bottleneck_type=final_bn,
                    problem_description=best.get("intent", ""),
                    skill_library=self._skill_library,
                    top_k=3,
                )

        # Launch TUI for visualization
        self._console.print()
        self._console.print("[white]Launching TUI...[/white]")

        # Flush stdin to prevent stale newline from triggering TUI input
        import sys, select
        while select.select([sys.stdin], [], [], 0.0)[0]:
            sys.stdin.read(1)

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

        # Fire post_optimize hooks silently (state tracking only)
        self._console.quiet = True
        self._hooks.fire(
            HookRegistry.POST_OPTIMIZE,
            best_speedup=self._session_data.get("best_speedup", 0.0),
            iterations_kept=kept,
            iterations_total=len(iters),
            cost=self._total_cost,
            cache_session_id=self._session_id,
        )
        self._console.quiet = False

        # Generate next-step suggestions for the recursive loop
        next_steps = generate_next_steps_rule_based(
            self._session_data,
            {"bottleneck": last_bottleneck, "tried": []},
        )
        # Try LLM-powered suggestions (async, falls back to rule-based)
        try:
            next_steps = asyncio.run(generate_next_steps_llm(
                self._session_data,
            ))
        except Exception:
            pass  # keep rule-based fallback

        # Store for "1"/"2"/"3" shortcut input
        self._pending_next_steps = next_steps

        # Single summary card — replaces all scattered post-optimization output
        elapsed = _time.time() - start_time
        render_optimization_summary(
            iterations=iters,
            best_speedup=best_speedup,
            best_iteration=best_iter_num,
            kept_count=kept,
            total_count=len(iters),
            cost_summary=self._cost_tracker.format_summary(),
            elapsed_seconds=elapsed,
            bottleneck_type=last_bottleneck,
            bottleneck_metrics=last_profile,
            skill_suggestions=suggestions,
            next_steps=next_steps,
            dashboard_url=f"http://localhost:8050/session/{self._session_id}",
            saved_path="",
            console=self._console,
        )

    def _run_kernel_agent_optimization(
        self, reference: str, iterations: int
    ) -> None:
        """Run optimization using Meta's KernelAgent engine."""
        import logging as _logging
        from pathlib import Path as _Path
        from kernel_code.integration.kernel_agent_bridge import KernelAgentBridge
        from kernel_code.live_display import LiveOptimizationDisplay
        from kernel_code.run_log import RunLogger

        # Suppress all noisy loggers before anything runs
        for _name in ["LiteLLM", "litellm", "httpx", "openai", "kernel_code.ke_profile"]:
            _lg = _logging.getLogger(_name)
            _lg.setLevel(_logging.CRITICAL)
            _lg.propagate = False

        ref_path = _Path(reference)
        if not ref_path.exists():
            self._console.print(f"[red]Error:[/red] file not found: {escape(reference)}")
            return

        reference_source = ref_path.read_text()

        # Detect problem name
        ref_problem = ""
        for line in reference_source.split("\n", 5):
            if "Problem" in line or "KernelBench" in line:
                ref_problem = line.strip().strip('"').strip("'").strip()
                break

        live_display = LiveOptimizationDisplay(
            console=self._console,
            problem=ref_problem or ref_path.name,
            hardware=self._settings.default_gpu,
            backend=self._settings.default_backend,
        )

        run_logger = RunLogger()
        run_logger.start_run(
            command="/optimize --engine kernel-agent",
            config={
                "engine": "kernel-agent",
                "model": self._settings.default_model,
                "hardware": self._settings.default_gpu,
                "backend": self._settings.default_backend,
                "reference": reference,
            },
        )

        model = self._settings.default_model
        workers = self._settings.num_workers
        self._console.print(f"[bold]Running optimization (KernelAgent engine)...[/bold]")
        self._console.print(f"  Reference: {escape(reference)}")
        self._console.print(f"  Model:     {model}")
        self._console.print(f"  Workers:   {workers} parallel")
        self._console.print()

        bridge = KernelAgentBridge(
            reference_source=reference_source,
            model_name=model,
            num_workers=workers,
            max_rounds=iterations,
            hardware=self._settings.default_gpu,
            live_display=live_display,
            run_logger=run_logger,
        )

        result = {}
        live_display.start()
        try:
            result = bridge.run()
        finally:
            stop_reason = "completed" if result.get("success") else "no correct kernel found"
            live_display.finish(stop_reason=stop_reason)

        # Show results
        speedup = result.get("speedup", 0.0)
        kernel_code = result.get("kernel_code", "")

        run_logger.end_run(
            best_speedup=speedup,
            best_kernel=kernel_code,
            stop_reason=stop_reason,
            total_cost=0.0,
        )

        # Render kernel profile report
        from kernel_code.kernel_profile import render_kernel_profile
        render_kernel_profile(
            speedup=speedup,
            ref_runtime_us=result.get("ref_runtime_us", 0.0),
            kernel_runtime_us=result.get("kernel_runtime_us", 0.0),
            profile=result.get("profile", {}),
            hardware=self._settings.default_gpu,
            console=self._console,
        )

        if result.get("success"):
            self._console.print(
                f"\n  [bold #4ade80]Correct kernel found: {speedup:.2f}x[/bold #4ade80]"
            )
            self._console.print(
                f"  [white]Worker {result.get('worker_id', '?')}, "
                f"{result.get('rounds', '?')} rounds, "
                f"{result.get('elapsed', 0):.0f}s[/white]"
            )
            # Save
            if kernel_code:
                from kernel_agent.model_wrapper import wrap_in_model_new
                wrapped = wrap_in_model_new(kernel_code, reference_source)
                out_path = _Path.cwd() / "reference_optimized.py"
                out_path.write_text(wrapped)
                self._console.print(f"  [#4ade80]Saved: reference_optimized.py[/#4ade80]")
        else:
            self._console.print(
                f"\n  [white]No correct kernel found after {result.get('rounds', '?')} rounds[/white]"
            )

        # Record to KE profile
        from kernel_code.ke_profile import KEProfile, classify_problem
        ke_profile = KEProfile()
        problem_type = classify_problem(ref_problem or "custom")
        ke_profile.record_run(
            problem_type=problem_type,
            total_iterations=result.get("rounds", 0),
            best_at_iteration=result.get("rounds", 0) if result.get("success") else 0,
            final_speedup=speedup,
        )

        # Generate next-step suggestions
        self._session_data = {"iterations": [], "best_speedup": speedup}
        next_steps = generate_next_steps_rule_based(self._session_data)
        try:
            from openkernel.config import ModelConfig as _MC
            _model_cfg = _MC(
                provider=self._settings.default_provider,
                model_id=self._settings.default_model,
            )
            next_steps = asyncio.run(generate_next_steps_llm(
                self._session_data, model_config=_model_cfg,
            ))
        except Exception:
            pass
        self._pending_next_steps = next_steps
        if next_steps:
            self._console.print(format_next_steps(next_steps))

        self._console.print()
        self._console.print(
            f"  [white]/dashboard[/white] [#888888]— open full analysis in browser[/#888888]"
        )
        self._console.print(f"  [white]Run log:[/white] [#888888]{run_logger.log_path}[/#888888]")
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
            config={"backend": backend, "hardware": self._settings.default_gpu},
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
        from kernel_code.live_display import LiveOptimizationDisplay

        # Detect problem name from reference.py docstring
        ref_problem = ""
        ref_lines = ref_path.read_text().split("\n", 5)
        for line in ref_lines:
            if "Problem" in line or "KernelBench" in line:
                ref_problem = line.strip().strip('"').strip("'").strip()
                break
        if not ref_problem:
            ref_problem = ref_path.name

        live_display = LiveOptimizationDisplay(
            console=self._console,
            problem=ref_problem,
            hardware=self._settings.default_gpu,
            backend=backend,
            max_iterations=iterations,
        )

        # Run logger
        from kernel_code.run_log import RunLogger
        run_logger = RunLogger()
        run_logger.start_run(
            command=f"/optimize --reference {reference}",
            config={
                "model": self._settings.default_model,
                "hardware": self._settings.default_gpu,
                "backend": backend,
                "reference": reference,
                "iterations": iterations,
            },
        )

        problem_label = f"L{level}#{problem}"
        bridge = OpenKernelBridge(
            config=config,
            session_id=self._session_id,
            problem_label=problem_label,
            hardware=self._settings.default_gpu,
            backend=backend,
            hooks=self._hooks,
            progress=self._opt_progress,
            live_display=live_display,
            run_logger=run_logger,
            file_cache=self._file_cache,
        )

        # Start inline live display
        live_display.start()
        try:
            result = bridge.run_optimization(reference_source)
        finally:
            # Get stop reason from bridge if available
            stop_reason = getattr(bridge, '_stop_reason', '')
            live_display.finish(stop_reason=stop_reason)

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

        # Generate next-step suggestions (use user's selected model, not default)
        next_steps = generate_next_steps_rule_based(self._session_data)
        try:
            from openkernel.config import ModelConfig as _MC
            _model_cfg = _MC(
                provider=self._settings.default_provider,
                model_id=self._settings.default_model,
            )
            next_steps = asyncio.run(generate_next_steps_llm(
                self._session_data, model_config=_model_cfg,
            ))
        except Exception:
            pass  # keep rule-based fallback
        self._pending_next_steps = next_steps
        if next_steps:
            self._console.print(format_next_steps(next_steps))

        # Post-optimization hints
        self._console.print()
        self._console.print(
            f"  [white]/dashboard[/white] [#888888]— open full analysis in browser[/#888888]"
        )

        # Save best kernel to file
        if result.final_kernel:
            out_name = f"{ref_path.stem}_optimized.py"
            out_path = _Path.cwd() / out_name
            out_path.write_text(result.final_kernel)
            self._console.print(
                f"[green]Best kernel saved:[/green] {out_name}"
            )

        # Finalize run log
        run_logger.end_run(
            best_speedup=result.final_speedup,
            best_kernel=result.final_kernel,
            stop_reason=stop_reason,
            total_cost=self._total_cost,
        )
        self._console.print(
            f"  [white]/log[/white] [#888888]— {run_logger.log_path}[/#888888]"
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
        """Print the welcome banner — Claude Code style."""
        self._console.print()
        self._console.print(
            "  [bold #d77757]openkernel[/bold #d77757] [#999999]v0.1[/#999999]"
        )
        self._console.print()

        # Problem
        ref_path = _PROJECT_ROOT / "reference.py"
        problem_label = ""
        if ref_path.is_file():
            first_lines = ref_path.read_text().split("\n", 5)
            for line in first_lines:
                if "KernelBench" in line or "Problem" in line:
                    problem_label = line.strip().strip('"').strip("'").strip()
                    break
        if problem_label:
            self._console.print(f"  [bold white]{problem_label}[/bold white]")
        else:
            self._console.print("  [#ffc107]No problem loaded[/#ffc107] \u2014 run [bold]/problem 1.5[/bold]")

        # Sub-info via ⎿
        hw = self._settings.default_gpu
        be = self._settings.default_backend
        model_name = self._settings.default_model
        from kernel_code.settings import _API_KEY_ENV_MAP
        env_key = _API_KEY_ENV_MAP.get(f"{self._settings.default_provider}_api_key", "")
        has_key = self._has_provider_key(self._settings.default_provider, env_key)
        key_sym = "\u2713" if has_key else "\u2717"
        key_color = "#4eba65" if has_key else "#ff6b80"

        self._console.print(
            f"  \u23bf  [{key_color}]{key_sym}[/{key_color}] "
            f"[white]{model_name}[/white]  [#999999]\u00b7[/#999999]  "
            f"[white]{hw}[/white]  [#999999]\u00b7[/#999999]  "
            f"[white]{be}[/white]"
        )

        # Context + skills
        context_parts = []
        if self._kernel_config and self._kernel_config.source_path:
            from kernel_code.kernel_config import load_hardware_context, load_backend_context, load_pitfalls
            ctx_count = 1  # KERNEL.md
            if load_hardware_context(hw): ctx_count += 1
            if load_backend_context(be): ctx_count += 1
            if load_pitfalls(): ctx_count += 1
            context_parts.append(f"{ctx_count} context files")

        n_skills = len(self._skill_library)
        if n_skills:
            context_parts.append(f"{n_skills} skills")

        if self._settings.max_budget is not None:
            context_parts.append(f"${self._settings.max_budget:.2f} budget")

        if context_parts:
            self._console.print(f"  \u23bf  [#999999]{' \u00b7 '.join(context_parts)}[/#999999]")

        self._console.print()
        self._console.print(
            "  [#999999]Type [bold white]/help[/bold white] for commands, or ask a question.[/#999999]"
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
