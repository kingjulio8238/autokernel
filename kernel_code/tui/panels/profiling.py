"""Profiling panel -- 4 gauge widgets + bottleneck banner + headroom/roofline info.

Displays: bandwidth utilization, compute utilization, L2 cache efficiency,
occupancy, bottleneck type banner, estimated headroom, and roofline position.
"""

from __future__ import annotations

from rich.text import Text

from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Static

from kernel_code.tui.widgets.gauge import GaugeWidget

# Target thresholds for each metric (reasonable GPU optimization targets)
_TARGETS = {
    "bandwidth_util": 0.85,
    "compute_util": 0.80,
    "cache_efficiency": 0.85,
    "occupancy": 0.90,
}

# Bottleneck banner colors
_BOTTLENECK_COLORS = {
    "memory_bound": "#f87171",
    "compute_bound": "#fbbf24",
    "latency_bound": "#c084fc",
    "unknown": "dim",
}


class ProfilingPanel(Widget):
    """Profiling summary panel with utilization gauges and diagnostics."""

    DEFAULT_CSS = """
    ProfilingPanel {
        height: auto;
        min-height: 10;
        background: #24231f;
        border: solid #3d3a36;
        padding: 0;
    }
    ProfilingPanel > Static {
        height: 1;
        padding: 0 1;
    }
    ProfilingPanel > .panel-title {
        background: #2e2c28;
        color: #a09890;
        text-style: bold;
    }
    ProfilingPanel > .bottleneck-banner {
        height: 1;
        text-style: bold;
        padding: 0 1;
    }
    ProfilingPanel > .diagnostics-line {
        height: 1;
        padding: 0 1;
    }
    """

    def __init__(self, profile: dict | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._profile = profile or {}

    def compose(self) -> ComposeResult:
        title = Text("PROFILING SUMMARY", style="bold")
        yield Static(title, classes="panel-title")
        yield GaugeWidget(
            label="BW",
            value=self._profile.get("bandwidth_util", 0.0),
            target=_TARGETS["bandwidth_util"],
            id="gauge-bw",
        )
        yield GaugeWidget(
            label="Comp",
            value=self._profile.get("compute_util", 0.0),
            target=_TARGETS["compute_util"],
            id="gauge-compute",
        )
        yield GaugeWidget(
            label="L2",
            value=self._profile.get("cache_efficiency", 0.0),
            target=_TARGETS["cache_efficiency"],
            id="gauge-cache",
        )
        yield GaugeWidget(
            label="Occ",
            value=self._profile.get("occupancy", 0.0),
            target=_TARGETS["occupancy"],
            id="gauge-occ",
        )
        yield Static(
            self._render_bottleneck_banner(),
            classes="bottleneck-banner",
            id="bottleneck-text",
        )
        yield Static(
            self._render_diagnostics(),
            classes="diagnostics-line",
            id="diagnostics-text",
        )

    def update_profile(self, profile: dict) -> None:
        """Update profiling gauges with new profile data."""
        self._profile = profile
        try:
            self.query_one("#gauge-bw", GaugeWidget).update_value(
                profile.get("bandwidth_util", 0.0)
            )
            self.query_one("#gauge-compute", GaugeWidget).update_value(
                profile.get("compute_util", 0.0)
            )
            self.query_one("#gauge-cache", GaugeWidget).update_value(
                profile.get("cache_efficiency", 0.0)
            )
            self.query_one("#gauge-occ", GaugeWidget).update_value(
                profile.get("occupancy", 0.0)
            )
            self.query_one("#bottleneck-text", Static).update(
                self._render_bottleneck_banner()
            )
            self.query_one("#diagnostics-text", Static).update(
                self._render_diagnostics()
            )
        except Exception:
            pass

    def _render_bottleneck_banner(self) -> Text:
        """Render a full-width colored bottleneck banner."""
        bottleneck = self._profile.get("bottleneck_type", "unknown")
        color = _BOTTLENECK_COLORS.get(bottleneck, "dim")
        label = bottleneck.upper().replace("_", " ")

        banner = Text()
        banner.append("\u2588 ", style=color)
        banner.append(f"Bottleneck: {label}", style=f"bold {color}")
        return banner

    def _render_diagnostics(self) -> Text:
        """Render headroom and roofline diagnostics line."""
        parts = Text()
        has_content = False

        headroom = self._profile.get("estimated_headroom", 0.0)
        if headroom and headroom > 0:
            if has_content:
                parts.append(" \u2502 ", style="dim")
            parts.append("Headroom: ", style="dim")
            parts.append(f"~{headroom:.1f}x remaining", style="bold cyan")
            has_content = True

        roofline = self._profile.get("roofline_position", 0.0)
        if roofline and roofline > 0:
            if has_content:
                parts.append(" \u2502 ", style="dim")
            roofline_pct = int(roofline * 100)
            parts.append("Roofline: ", style="dim")
            roofline_color = "#4ade80" if roofline >= 0.7 else "#fbbf24" if roofline >= 0.4 else "#f87171"
            parts.append(f"{roofline_pct}% of peak", style=f"bold {roofline_color}")
            has_content = True

        if not has_content:
            parts.append(" ", style="dim")

        return parts
