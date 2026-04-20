"""TUI design tokens — dark-mode complement to the dashboard's light theme.

Inverts the humanoid-terminal palette (docs/kernel-code-design.md) for
dark terminals while preserving the same visual identity:
  - Same green (#276749 light → #4ade80 dark for readability)
  - Same red (#c53030 light → #ef4444 dark for readability)
  - Same warm undertones (brown-tinted neutrals, not blue-gray)
  - Same font intent (monospace labels, clean data)

Dashboard (light):  bg #f5f2ed → text #1a1a1a → border #e0ddd8
TUI (dark):         bg #1a1a18 → text #e0ddd8 → border #3d3a36
"""

# Background layers (warm brown-blacks, NOT blue-black or pure black)
BG = "#1a1a18"           # app background — darkest
SURFACE = "#24231f"      # panel/card background
SURFACE_ALT = "#2e2c28"  # slightly lighter (headers, active rows)
BORDER = "#3d3a36"       # panel borders — warm, subtle

# Text hierarchy (warm off-whites, matching dashboard's warm text)
TEXT = "#e0ddd8"         # primary — same as dashboard bg inverted
TEXT_SECONDARY = "#a09890"  # secondary labels
TEXT_DIM = "#6b6360"     # dim info, separators

# Semantic colors (brighter than dashboard for dark-bg readability)
GREEN = "#4ade80"        # keep / success (dashboard uses #276749)
RED = "#ef4444"          # discard / error (dashboard uses #c53030)
AMBER = "#fbbf24"        # warnings, compute-bound
CYAN = "#22d3ee"         # accents, links, active problem
PURPLE = "#c084fc"       # latency-bound

# Status color map (matches dashboard STATUS_COLORS keys)
STATUS = {
    "keep": GREEN,
    "discard": RED,
    "compile_error": RED,
    "incorrect": AMBER,
    "error": RED,
    "pending": TEXT_DIM,
    "active": CYAN,
}

# Best-row highlight
BEST_ROW_BG = "#1a2e1a"  # dark green tint for best row background
BEST_ROW_FG = GREEN
"""

All colors are chosen to:
1. Match the dashboard's visual identity (same hue family)
2. Be readable on dark terminal backgrounds
3. Use warm undertones (brown not blue) for consistency with humanoid-terminal
"""
