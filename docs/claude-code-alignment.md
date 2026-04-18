# Claude Code Alignment: Tiered Extensions for kernel code

*Based on thorough review of Claude Code source at /Users/juliansaks/Downloads/src (1,884 TypeScript files)*

---

## Architecture Comparison

| Aspect | Claude Code | kernel code (current) | Gap |
|--------|------------|----------------------|-----|
| **Language** | TypeScript/Bun | Python | Different but OK |
| **UI Framework** | Custom Ink (React for terminal) | Textual | Different but OK |
| **Tool Count** | 43 registered tools | 6 agent tools + 14 slash commands | Need formal tool system |
| **Command Count** | 50+ slash commands | 14 slash commands | Need more domain commands |
| **Skill System** | Bundled skills + disk-based skills + MCP skills | 10 JSON skills with /skill:NAME | Need auto-triggering |
| **Query Engine** | Dedicated QueryEngine.ts (streaming + tool dispatch + compaction) | Simple agent_loop.py | Need proper query engine |
| **Context Management** | 5+ compaction strategies, auto-compact, reactive compact | Basic compaction.py | Need smarter compaction |
| **Cost Tracking** | Per-model, per-session, cache read/write, line changes | Basic cost estimation | Need detailed tracking |
| **History** | 100-item history with paste support, up/down arrow | None | Need command history |
| **Permissions** | 6 modes, per-tool rules, compound command analysis | Basic cost confirmation | Need per-tool permissions |
| **Error Handling** | Formatted errors with actionable messages | Raw tracebacks | Need error formatting |
| **Onboarding** | Project onboarding state with steps | Basic onboarding wizard | Already close |

---

## Tier 1: Highest ROI (Build Next)

### 1.1 Formal Tool Registry System
**What CC does**: `Tool.ts` defines a `ToolUseContext` with 43 tools, each having: name, description, input schema (JSON Schema), execute function, permission requirements, result formatting, progress reporting, spinner mode.

**What kernel code should do**: Create `kernel_code/tools/` directory with formal tool definitions:
```python
@dataclass
class KernelTool:
    name: str
    description: str
    parameters: dict  # JSON Schema
    execute: Callable
    permission: str  # "auto" | "ask" | "deny"
    category: str  # "optimization" | "analysis" | "management"
```

Kernel-specific tools to add:
- `profile_kernel` — run profiler on a kernel file
- `evaluate_kernel` — benchmark against reference on Modal
- `generate_kernel` — generate a kernel variant for a given intent
- `compare_kernels` — side-by-side two kernels with profiling diff
- `search_kernelbench` — search KernelBench problems by type
- `load_skill` — load an optimization skill into context
- `show_roofline` — display roofline analysis
- `suggest_optimization` — suggest next optimization based on profiler data

**ROI**: High. Makes the agentic loop much more capable — LLM can call specific kernel tools, not just generic Q&A functions.

### 1.2 Command History (Up/Down Arrow)
**What CC does**: `history.ts` — 100-item command history persisted to disk, up/down arrow navigation, paste content handling with hash-based dedup.

**What kernel code should do**: Use `prompt_toolkit` instead of `input()` for the REPL. This gives: up/down history, line editing, multi-line input, tab completion for commands, syntax highlighting.

```python
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory

session = PromptSession(
    history=FileHistory('.kernel-code/history.txt'),
    completer=KernelCodeCompleter(),  # tab-complete /commands and skill names
)
user_input = session.prompt('kernel-code > ')
```

**ROI**: Very high. Single biggest UX improvement for daily use. Every KE uses up-arrow constantly.

### 1.3 Proper Error Formatting
**What CC does**: Errors are caught, formatted with actionable messages, and displayed in styled panels. Never shows raw tracebacks to users.

**What kernel code should do**: Create `kernel_code/errors.py`:
```python
def format_error(exc: Exception, context: str = "") -> str:
    """Format an error for user display."""
    if isinstance(exc, ConfigurationError):
        return f"Configuration error: {exc}\n  Fix: check .kernel-code/settings.yaml"
    if isinstance(exc, RateLimitError):
        return f"Rate limited. Wait {extract_wait_time(exc)}s and retry."
    if isinstance(exc, ModalError):
        return f"GPU eval failed: {exc}\n  Fix: run 'modal deploy modal_infra/app.py'"
    # ... etc
```

Wrap all shell command handlers with error formatting. Never show raw Python tracebacks.

**ROI**: High. Professional feel. KEs shouldn't see `litellm.RateLimitError: RateLimitError: GroqException - {"error":...}`.

### 1.4 Query Engine (Proper Agentic Loop)
**What CC does**: `QueryEngine.ts` + `query.ts` — dedicated engine that handles: streaming API calls, tool dispatch from streamed chunks, auto-compaction when context fills, token budget enforcement, turn-level budgets, fallback on errors.

**What kernel code should do**: Upgrade `agent_loop.py` into a proper `QueryEngine`:
- Stream tokens AND parse tool calls from the stream (not wait for full response)
- Auto-compact conversation when approaching token limit
- Enforce per-turn budget (max tokens per response)
- Handle API errors with retry + fallback
- Track conversation turns with proper message types (user, assistant, tool_result)

**ROI**: High. Makes the shell feel responsive and handles long sessions properly.

### 1.5 Cost Tracking Dashboard
**What CC does**: `cost-tracker.ts` tracks per-model usage, cache read/write tokens, total cost, API duration, tool duration, lines changed. Displays with `/cost` command.

**What kernel code should do**: Upgrade `permissions.py` BudgetTracker into a full cost tracker:
- Track per-model token usage (input/output/cached)
- Track Modal GPU time separately
- Track per-run cost breakdown
- Add `/cost` command showing session cost summary
- Show cost in status bar (already done partially)

**ROI**: Medium-high. KEs need to know what they're spending.

---

## Tier 2: High ROI (Build After Tier 1)

### 2.1 Auto-Triggering Skills
**What CC does**: `bundledSkills.ts` — skills have `whenToUse` field that triggers automatically when the LLM determines the skill is relevant. Not just slash commands.

**What kernel code should do**: When the LLM generates a kernel and the Critic identifies a bottleneck, automatically search skills and suggest relevant ones:
```
Critic: memory_bound, L2 hit rate 45%
→ Auto-suggest: /skill:triton_reduction (online softmax, 2.4x on similar problems)
```

Add `auto_trigger` field to skill JSON: condition under which the skill is automatically loaded.

### 2.2 Progress Reporting Per Tool
**What CC does**: Each tool has a `ToolProgressData` type — spinners show what the tool is doing (e.g., "Reading file...", "Running command..."). Different spinner modes per tool.

**What kernel code should do**: During optimization, show per-iteration progress:
```
[●] Generating kernel (Triton, intent: vectorize loads)...
[●] Evaluating on L40S (fast mode, ~5s)...
[●] Profiling: memory_bound, BW 73%...
[✓] Kept: 2.84x (new best!)
```

Not just "Thinking..." — show what's happening at each stage.

### 2.3 Tab Completion
**What CC does**: Has typeahead/autocomplete for commands and file paths.

**What kernel code should do**: With `prompt_toolkit`, add completers:
- `/` → list all slash commands
- `/skill:` → list all skill names
- `/show ` → `best`, `results`, `run`
- `/optimize --` → `--reference`, `--backend`, `--config`, `--parallel`
- File paths for `--reference`

### 2.4 Conversation Message Types
**What CC does**: 7+ message types: UserMessage, AssistantMessage, SystemMessage, ToolUseSummaryMessage, AttachmentMessage, TombstoneMessage, ProgressMessage. Each renders differently.

**What kernel code should do**: Type the conversation history properly:
```python
@dataclass
class KernelMessage:
    role: str  # "user" | "assistant" | "system" | "tool_result" | "optimization_event"
    content: str
    timestamp: str
    metadata: dict  # tool name, speedup, cost, etc.
```

This enables: proper context assembly, smart compaction (summarize tool_results, keep user messages), replay/resume.

### 2.5 File State Cache
**What CC does**: `fileStateCache.ts` tracks which files have been read, their line counts, and modification times. Prevents re-reading unchanged files.

**What kernel code should do**: Cache kernel files and profiler outputs:
- Don't re-profile a kernel variant that hasn't changed
- Cache reference file parsing (only re-parse if modified)
- Track which skills have been loaded in this session

---

## Tier 3: Medium ROI (Polish)

### 3.1 Multi-Line Input
**What CC does**: Shift+Enter for multi-line input, paste detection for code blocks.

**What kernel code should do**: With `prompt_toolkit`, support multi-line input for pasting kernel code directly into the shell:
```
kernel-code > paste this kernel:
... @triton.jit
... def my_kernel(...):
...     ...
```

### 3.2 /compact Command
**What CC does**: Manual `/compact` command to force conversation compaction.

**What kernel code should do**: Add `/compact` that summarizes the current session:
```
kernel-code > /compact
Compacted session: 45 iterations → summary (kept 8 results, 3 strategies, current best 2.84x)
Context reduced from ~12K to ~3K tokens.
```

### 3.3 /context Command
**What CC does**: Shows what's in the current context window — files, tools, system prompt size, conversation size.

**What kernel code should do**: `/context` showing:
```
Context breakdown:
  System prompt:     800 tokens
  KERNEL.md:         200 tokens
  Skills loaded:     150 tokens (triton_reduction)
  Session history:   2,400 tokens (14 iterations, compacted)
  Best kernel:       300 tokens
  Total:             3,850 / 4,096 tokens (94%)
```

### 3.4 /diff Command
**What CC does**: Shows git diff inline.

**What kernel code should do**: `/diff` showing what changed between the best kernel and the original reference:
```
kernel-code > /diff
--- reference.py (torch.softmax)
+++ softmax_optimized.py (2.84x)

+ @triton.jit
+ def softmax_kernel(...)
+     # Online softmax: single-pass
- return torch.softmax(x, dim=-1)
+ return triton_softmax(x)
```

### 3.5 Theme Command
**What CC does**: `/color` and `/theme` to switch themes.

**What kernel code should do**: Already have the warm theme. Add `/theme` to switch between light/dark or adjust accent colors.

### 3.6 Doctor Command
**What CC does**: `/doctor` diagnoses installation issues.

**What kernel code should do**: `/doctor` checks:
```
kernel-code > /doctor
✓ Python 3.11+
✓ Modal authenticated (L40S available)
✓ Groq API key valid
✗ MiniMax API key missing (optional)
✓ KernelBench importable (via Modal)
✓ 10 skills loaded
✓ KERNEL.md found
✓ Settings loaded from 2 files
```

---

## Tier 4: Nice to Have (Future)

### 4.1 MCP Server Integration
Expose kernel code tools as MCP servers so other AI tools can call them.

### 4.2 Vim Mode
`/vim` for vi keybindings in the REPL (Claude Code has this).

### 4.3 Voice Mode
Voice input for kernel optimization commands (Claude Code has experimental voice).

### 4.4 Remote Sessions
Run kernel code as a remote server accessible from browser (like Claude Code's bridge mode).

### 4.5 Plugin System
Allow third-party kernel optimization plugins (new backends, new profilers, new skills).

---

## Implementation Priority

| Priority | Item | Estimated Effort | Impact |
|----------|------|-----------------|--------|
| **1** | Command history (prompt_toolkit) | 2 hours | Huge UX improvement |
| **2** | Error formatting | 2 hours | Professional feel |
| **3** | Formal tool registry | 4 hours | Better agentic loop |
| **4** | Progress reporting | 3 hours | Better feedback during optimization |
| **5** | Cost tracking dashboard + /cost | 2 hours | Budget visibility |
| **6** | Tab completion | 2 hours | Faster command entry |
| **7** | Auto-triggering skills | 3 hours | Smarter optimization |
| **8** | Query engine upgrade | 6 hours | Handles long sessions |
| **9** | /context, /compact, /diff, /doctor | 3 hours | Power user commands |
| **10** | Conversation message types | 3 hours | Better context management |
