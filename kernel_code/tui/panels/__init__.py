"""TUI panels for kernel code."""

from kernel_code.tui.panels.experiment_log import ExperimentLogPanel
from kernel_code.tui.panels.optimization_feed import OptimizationFeedPanel
from kernel_code.tui.panels.profiling import ProfilingPanel
from kernel_code.tui.panels.status_bar import StatusBar
from kernel_code.tui.panels.trajectory import TrajectoryPanel

__all__ = [
    "ExperimentLogPanel",
    "OptimizationFeedPanel",
    "ProfilingPanel",
    "StatusBar",
    "TrajectoryPanel",
]
