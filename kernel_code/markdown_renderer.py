"""Render LLM responses as markdown in the terminal using Rich."""
import re
from rich.console import Console
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.panel import Panel

# Fast check: does the text contain markdown syntax?
# Matches: headers (#), code blocks (```), lists (- or *), bold (**), links ([)
_MD_SYNTAX_RE = re.compile(r'[#*`|[>\-_~]|\n\n|^\d+\. |\n\d+\. ', re.MULTILINE)

def has_markdown(text: str) -> bool:
    """Quick check if text contains markdown syntax.
    Samples first 500 chars (like Claude Code's hasMarkdownSyntax)."""
    sample = text[:500] if len(text) > 500 else text
    return bool(_MD_SYNTAX_RE.search(sample))

def render_response(text: str, console: Console | None = None) -> None:
    """Render an LLM response with markdown formatting.

    If markdown is detected: renders headers, code blocks, lists, bold/italic.
    If no markdown: prints as plain text (fast path).
    """
    c = console or Console()

    if not has_markdown(text):
        # Fast path — no markdown, just print
        c.print(text)
        return

    # Render as Rich Markdown
    try:
        md = Markdown(text, code_theme="monokai")
        c.print(md)
    except Exception:
        # Fallback to plain text if markdown parsing fails
        c.print(text)

def render_code_block(code: str, language: str = "python",
                      console: Console | None = None) -> None:
    """Render a code block with syntax highlighting."""
    c = console or Console()
    syntax = Syntax(code, language, theme="monokai", line_numbers=True,
                    word_wrap=True)
    c.print(syntax)

def render_kernel_diff(before: str, after: str, speedup: float = 0.0,
                       console: Console | None = None) -> None:
    """Render a before/after kernel comparison."""
    import difflib
    c = console or Console()

    diff = difflib.unified_diff(
        before.splitlines(keepends=True),
        after.splitlines(keepends=True),
        fromfile="reference (baseline)",
        tofile=f"optimized ({speedup:.2f}x)" if speedup > 0 else "optimized",
    )
    diff_text = "".join(diff)

    if diff_text:
        syntax = Syntax(diff_text, "diff", theme="monokai")
        c.print(syntax)
    else:
        c.print("[dim]No differences[/dim]")
