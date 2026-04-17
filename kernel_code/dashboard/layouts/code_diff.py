"""Code diff layout for the Dash dashboard.

Panel 5: Side-by-side diff of the current best kernel vs. the previous
iteration, with syntax highlighting via Pygments (falls back to plain <pre>).
"""

from __future__ import annotations

import difflib
import html as html_mod

import pandas as pd
from dash import dcc, html


def _highlight_line(source: str) -> str:
    """Syntax-highlight a single line of Python/Triton code to HTML.

    Uses Pygments when available; otherwise returns HTML-escaped plain text.
    """
    try:
        from pygments import highlight
        from pygments.formatters import HtmlFormatter
        from pygments.lexers import PythonLexer

        formatter = HtmlFormatter(nowrap=True, style="monokai")
        return highlight(source, PythonLexer(), formatter)
    except ImportError:
        return html_mod.escape(source)


def _highlight_block(source: str) -> str:
    """Syntax-highlight a full code block to HTML."""
    try:
        from pygments import highlight
        from pygments.formatters import HtmlFormatter
        from pygments.lexers import PythonLexer

        formatter = HtmlFormatter(nowrap=True, style="monokai")
        return highlight(source, PythonLexer(), formatter)
    except ImportError:
        return html_mod.escape(source)


def _diff_lines(old_code: str, new_code: str) -> tuple[list[dict], list[dict]]:
    """Compute a side-by-side diff with added/removed/unchanged markers.

    Returns two parallel lists of {text, status} dicts where status is one of
    "added", "removed", "unchanged", or "blank".
    """
    old_lines = old_code.splitlines()
    new_lines = new_code.splitlines()

    matcher = difflib.SequenceMatcher(None, old_lines, new_lines)

    left: list[dict] = []
    right: list[dict] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                left.append({"text": old_lines[i1 + k], "status": "unchanged"})
                right.append({"text": new_lines[j1 + k], "status": "unchanged"})
        elif tag == "replace":
            max_len = max(i2 - i1, j2 - j1)
            for k in range(max_len):
                if i1 + k < i2:
                    left.append({"text": old_lines[i1 + k], "status": "removed"})
                else:
                    left.append({"text": "", "status": "blank"})
                if j1 + k < j2:
                    right.append({"text": new_lines[j1 + k], "status": "added"})
                else:
                    right.append({"text": "", "status": "blank"})
        elif tag == "delete":
            for k in range(i2 - i1):
                left.append({"text": old_lines[i1 + k], "status": "removed"})
                right.append({"text": "", "status": "blank"})
        elif tag == "insert":
            for k in range(j2 - j1):
                left.append({"text": "", "status": "blank"})
                right.append({"text": new_lines[j1 + k], "status": "added"})

    return left, right


_STATUS_BG = {
    "added": "rgba(34,197,94,0.15)",
    "removed": "rgba(239,68,68,0.15)",
    "unchanged": "transparent",
    "blank": "transparent",
}

_STATUS_BORDER = {
    "added": "2px solid rgba(34,197,94,0.4)",
    "removed": "2px solid rgba(239,68,68,0.4)",
    "unchanged": "none",
    "blank": "none",
}


def _render_side(lines: list[dict], header: str) -> html.Div:
    """Render one side of the diff as a Dash html.Div.

    Each line is rendered as plain text inside monospace-styled Dash components.
    Pygments highlighting is applied only when wrapping in dcc.Markdown; for
    the line-by-line diff we keep plain text to avoid invalid Dash props.
    """
    code_lines = []
    for idx, line in enumerate(lines, 1):
        display_text = line["text"] if line["text"] else " "
        code_lines.append(
            html.Div(
                style={
                    "display": "flex",
                    "backgroundColor": _STATUS_BG[line["status"]],
                    "borderLeft": _STATUS_BORDER[line["status"]],
                    "padding": "1px 8px",
                    "minHeight": "20px",
                    "lineHeight": "20px",
                },
                children=[
                    html.Span(
                        f"{idx}",
                        style={
                            "color": "#475569",
                            "width": "35px",
                            "textAlign": "right",
                            "paddingRight": "12px",
                            "userSelect": "none",
                            "flexShrink": "0",
                            "fontFamily": "monospace",
                            "fontSize": "12px",
                        },
                    ),
                    html.Code(
                        display_text,
                        style={
                            "fontFamily": "monospace",
                            "fontSize": "12px",
                            "whiteSpace": "pre",
                            "color": "#e2e8f0",
                            "background": "none",
                        },
                    ),
                ],
            )
        )

    return html.Div(
        style={"flex": "1", "overflow": "auto", "minWidth": "0"},
        children=[
            html.Div(
                header,
                style={
                    "padding": "8px 12px",
                    "backgroundColor": "#1e293b",
                    "color": "#94a3b8",
                    "fontWeight": "bold",
                    "fontSize": "12px",
                    "borderBottom": "1px solid #334155",
                },
            ),
            html.Div(
                style={
                    "fontFamily": "monospace",
                    "fontSize": "12px",
                    "backgroundColor": "#0f172a",
                    "maxHeight": "500px",
                    "overflowY": "auto",
                },
                children=code_lines,
            ),
        ],
    )


def create_code_diff_component(df: pd.DataFrame) -> html.Div:
    """Create a side-by-side code diff component.

    Compares the current best kernel against the previous iteration's kernel.
    Requires a 'kernel_code_snippet' column in the DataFrame.

    Args:
        df: DataFrame with column kernel_code_snippet (and decision for filtering).

    Returns:
        A Dash html.Div containing the side-by-side diff.
    """
    if df.empty or "kernel_code_snippet" not in df.columns:
        return html.Div(
            "No kernel code available for diff.",
            style={"color": "#64748b", "padding": "20px"},
        )

    # Find current best and previous iteration
    keeps = df[df["decision"] == "keep"]
    if len(keeps) < 2:
        # Not enough kept iterations for a diff; show latest code via Markdown
        latest = df.iloc[-1]
        code = str(latest.get("kernel_code_snippet", ""))
        return html.Div(
            [
                html.H4(
                    "Current Kernel (no previous best to diff against)",
                    style={"color": "#e2e8f0"},
                ),
                dcc.Markdown(
                    f"```python\n{code}\n```",
                    style={
                        "backgroundColor": "#0f172a",
                        "padding": "16px",
                        "borderRadius": "6px",
                        "border": "1px solid #1e293b",
                        "fontFamily": "monospace",
                        "fontSize": "12px",
                        "overflowX": "auto",
                        "maxHeight": "500px",
                    },
                ),
            ]
        )

    old_code = str(keeps.iloc[-2].get("kernel_code_snippet", ""))
    new_code = str(keeps.iloc[-1].get("kernel_code_snippet", ""))

    old_iter = int(keeps.iloc[-2].get("iteration", 0))
    new_iter = int(keeps.iloc[-1].get("iteration", 0))

    left_lines, right_lines = _diff_lines(old_code, new_code)

    return html.Div(
        [
            html.Div(
                style={
                    "display": "flex",
                    "gap": "4px",
                    "border": "1px solid #1e293b",
                    "borderRadius": "6px",
                    "overflow": "hidden",
                },
                children=[
                    _render_side(left_lines, f"Previous Best (Iter {old_iter})"),
                    _render_side(right_lines, f"Current Best (Iter {new_iter})"),
                ],
            )
        ]
    )
