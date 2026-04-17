"""Shared design tokens from the humanoid-terminal design system.

All dashboard layout files should import from this module instead of
hardcoding colors, fonts, or styles.
"""

from __future__ import annotations

from dash import html

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------
COLORS = {
    "bg": "#f5f2ed",              # warm off-white background
    "bg_card": "#ffffff",          # white cards
    "text": "#1a1a1a",             # primary text
    "text_secondary": "#6b6b6b",
    "text_dim": "#a0a0a0",
    "bg_muted": "#ebe7e1",         # muted backgrounds
    "border": "#e0ddd8",           # subtle borders
    "border_hover": "#c5c0b8",
    "accent": "#1a1a1a",           # black accent
    "red": "#c53030",              # error / discard
    "green": "#276749",            # success / keep
    "baseline": "#1a1a1a",         # baseline reference line
    "gridline": "#e0ddd8",
    # Diff backgrounds
    "diff_added_bg": "#f0fff4",
    "diff_removed_bg": "#fff5f5",
}

# Status colors for data points
STATUS_COLORS = {
    "keep": COLORS["green"],
    "discard": COLORS["red"],
    "compile_error": COLORS["red"],
    "incorrect": "#b7791f",        # warm amber, not neon
    "error": COLORS["red"],
    "pending": COLORS["text_dim"],
    "active": COLORS["accent"],
}

# ---------------------------------------------------------------------------
# Fonts
# ---------------------------------------------------------------------------
FONTS = {
    "mono": "Share Tech Mono, monospace",
    "body": "Inter, sans-serif",
}

# Google Fonts link element (include in app.layout)
GOOGLE_FONTS_LINK = html.Link(
    rel="stylesheet",
    href=(
        "https://fonts.googleapis.com/css2?"
        "family=Inter:wght@400;500;600&family=Share+Tech+Mono&display=swap"
    ),
)

# ---------------------------------------------------------------------------
# Plotly figure theme
# ---------------------------------------------------------------------------
PLOTLY_THEME = dict(
    template="plotly_white",
    paper_bgcolor=COLORS["bg_card"],
    plot_bgcolor=COLORS["bg"],
    font=dict(family=FONTS["mono"], color=COLORS["text"], size=11),
    title_font=dict(family=FONTS["mono"], size=13, color=COLORS["text"]),
)

PLOTLY_AXIS_THEME = dict(
    gridcolor=COLORS["gridline"],
    linecolor=COLORS["border"],
    tickfont=dict(color=COLORS["text_secondary"]),
    title_font=dict(color=COLORS["text_secondary"]),
)


def apply_theme(fig, **extra_layout):
    """Apply the humanoid-terminal Plotly theme to a figure.

    Merges PLOTLY_THEME with any extra layout kwargs, then applies axis
    styling to xaxis / yaxis.
    """
    layout = {**PLOTLY_THEME, **extra_layout}
    fig.update_layout(**layout)
    fig.update_xaxes(**PLOTLY_AXIS_THEME)
    fig.update_yaxes(**PLOTLY_AXIS_THEME)
    return fig


# ---------------------------------------------------------------------------
# Dash component helpers
# ---------------------------------------------------------------------------

def card_style(**overrides) -> dict:
    """Return a CSS style dict for a card container."""
    base = {
        "backgroundColor": COLORS["bg_card"],
        "border": f"1px solid {COLORS['border']}",
        "borderRadius": "4px",
        "padding": "16px",
        "marginBottom": "16px",
    }
    base.update(overrides)
    return base


def section_header(title: str) -> html.Div:
    """Return an uppercase, dim section header matching humanoid-terminal."""
    return html.Div(
        title.upper(),
        style={
            "fontFamily": FONTS["mono"],
            "fontSize": "10px",
            "color": COLORS["text_dim"],
            "textTransform": "uppercase",
            "letterSpacing": "1.5px",
            "marginBottom": "8px",
        },
    )
