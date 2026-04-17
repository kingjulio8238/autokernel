"""Dashboard layout components."""

from kernel_code.dashboard.layouts.convergence import create_convergence_figure
from kernel_code.dashboard.layouts.cost_efficiency import create_cost_efficiency_figure
from kernel_code.dashboard.layouts.strategy_stats import create_strategy_stats_figure
from kernel_code.dashboard.layouts.trajectory import create_trajectory_figure

__all__ = [
    "create_convergence_figure",
    "create_cost_efficiency_figure",
    "create_strategy_stats_figure",
    "create_trajectory_figure",
]
