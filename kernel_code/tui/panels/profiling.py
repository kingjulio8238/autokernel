"""Profiling panel — 4 gauge widgets + bottleneck label.

Displays: bandwidth utilization, compute utilization, L2 cache efficiency,
occupancy, and the current bottleneck type.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Static

from kernel_code.tui.widgets.gauge import GaugeWidget


class ProfilingPanel(Widget):
    """Profiling summary panel with utilization gauges."""

    DEFAULT_CSS = """
    ProfilingPanel {
        height: auto;
        min-height: 8;
        border: solid $accent;
        padding: 0;
    }
    ProfilingPanel > Static {
        height: 1;
        padding: 0 1;
    }
    ProfilingPanel > .panel-title {
        background: $accent;
        color: $text;
        text-style: bold;
    }
    ProfilingPanel > .bottleneck-label {
        color: $warning;
        text-style: bold;
    }
    """

    def __init__(self, profile: dict | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._profile = profile or {}

    def compose(self) -> ComposeResult:
        yield Static("Profiling Summary", classes="panel-title")
        yield GaugeWidget(
            label="BW",
            value=self._profile.get("bandwidth_util", 0.0),
            id="gauge-bw",
        )
        yield GaugeWidget(
            label="Comp",
            value=self._profile.get("compute_util", 0.0),
            id="gauge-compute",
        )
        yield GaugeWidget(
            label="L2",
            value=self._profile.get("cache_efficiency", 0.0),
            id="gauge-cache",
        )
        yield GaugeWidget(
            label="Occ",
            value=self._profile.get("occupancy", 0.0),
            id="gauge-occ",
        )
        bottleneck = self._profile.get("bottleneck_type", "unknown")
        yield Static(
            f" Bottleneck: [bold]{bottleneck.upper().replace('_', ' ')}[/bold]",
            classes="bottleneck-label",
            id="bottleneck-text",
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
            bottleneck = profile.get("bottleneck_type", "unknown")
            self.query_one("#bottleneck-text", Static).update(
                f" Bottleneck: [bold]{bottleneck.upper().replace('_', ' ')}[/bold]"
            )
        except Exception:
            pass
