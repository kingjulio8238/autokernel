"""Formalized optimization log for structured round tracking.

Replaces ad-hoc dict-based round history with typed dataclasses.
Enables richer context injection into generator prompts and
structured A/B testing analysis.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum


class RoundStatus(Enum):
    """Status of an optimization round."""
    SUCCESS = "success"
    COMPILE_ERROR = "compile_error"
    RUNTIME_ERROR = "runtime_error"
    INCORRECT = "incorrect"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"


@dataclass
class ProfileMetrics:
    """Profiling metrics from NCU or Modal eval."""
    bandwidth_utilization_pct: float = 0.0
    compute_utilization_pct: float = 0.0
    occupancy: float = 0.0
    operational_intensity: float = 0.0
    total_flops: int = 0
    total_bytes: int = 0
    runtime_us: float = 0.0
    ref_runtime_us: float = 0.0
    l1_hit_rate_pct: float = 0.0
    l2_hit_rate_pct: float = 0.0
    top_stalls: list[str] = field(default_factory=list)
    sol_score: float = 0.0

    @classmethod
    def from_modal_profile(cls, profile: dict) -> "ProfileMetrics":
        """Create from a Modal eval profile dict."""
        return cls(
            bandwidth_utilization_pct=profile.get("bandwidth_utilization", 0.0) * 100,
            compute_utilization_pct=profile.get("compute_utilization", 0.0) * 100,
            occupancy=profile.get("occupancy", 0.0),
            operational_intensity=profile.get("operational_intensity", 0.0),
            total_flops=profile.get("total_flops", 0),
            total_bytes=profile.get("total_bytes", 0),
            runtime_us=profile.get("runtime_us", 0.0),
            ref_runtime_us=profile.get("ref_runtime_us", 0.0),
        )


@dataclass
class OptimizationRound:
    """A single round in the optimization trajectory.

    Designed for injection into LLM generator context and
    structured analysis in A/B testing.
    """
    round: int
    kernel_code: str
    is_correct: bool
    speedup: float
    status: RoundStatus
    strategy: str = ""
    bottleneck: str = ""
    method_applied: str = ""
    profile: ProfileMetrics = field(default_factory=ProfileMetrics)
    worker_results: list[dict] = field(default_factory=list)
    cost_usd: float = 0.0
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization."""
        d = asdict(self)
        d["status"] = self.status.value
        return d

    def to_context_string(self) -> str:
        """Format for LLM context injection."""
        parts = [
            f"Round {self.round}: {self.status.value}",
            f"  Speedup: {self.speedup:.2f}x",
            f"  Strategy: {self.strategy}",
        ]
        if self.bottleneck:
            parts.append(f"  Bottleneck: {self.bottleneck}")
        if self.method_applied:
            parts.append(f"  Method: {self.method_applied}")
        if self.profile.bandwidth_utilization_pct > 0:
            parts.append(
                f"  Profile: BW={self.profile.bandwidth_utilization_pct:.1f}%, "
                f"Compute={self.profile.compute_utilization_pct:.1f}%, "
                f"Occupancy={self.profile.occupancy:.2f}"
            )
        if self.profile.sol_score > 0:
            parts.append(f"  SOL Score: {self.profile.sol_score:.2f}")
        return "\n".join(parts)


@dataclass
class OptimizationLog:
    """Complete optimization trajectory across rounds.

    Accumulates OptimizationRound entries and provides
    context formatting for generator prompts.
    """
    rounds: list[OptimizationRound] = field(default_factory=list)
    best_speedup: float = 0.0
    best_round: int = 0
    target_speedup: float = 0.0

    def add_round(self, round_entry: OptimizationRound) -> None:
        """Add a round and update best tracking."""
        self.rounds.append(round_entry)
        if round_entry.speedup > self.best_speedup:
            self.best_speedup = round_entry.speedup
            self.best_round = round_entry.round

    def to_context_string(self, max_rounds: int = 10) -> str:
        """Format the trajectory for LLM context injection.

        Shows the most recent rounds (up to max_rounds) with
        a summary header.
        """
        if not self.rounds:
            return "No optimization history yet."

        header = (
            f"## Optimization Trajectory\n"
            f"Rounds completed: {len(self.rounds)} | "
            f"Best speedup: {self.best_speedup:.2f}x (round {self.best_round}) | "
            f"Target: {self.target_speedup:.1f}x\n"
        )

        # Show most recent rounds
        recent = self.rounds[-max_rounds:]
        round_strs = [r.to_context_string() for r in recent]

        return header + "\n".join(round_strs)

    def get_method_history(self) -> list[dict]:
        """Return list of method dicts for deduplication checks."""
        return [
            {"method_name": r.method_applied, "speedup": r.speedup, "round": r.round}
            for r in self.rounds
            if r.method_applied
        ]

    def to_dict_list(self) -> list[dict]:
        """Serialize all rounds for JSON output."""
        return [r.to_dict() for r in self.rounds]

    @classmethod
    def from_dict_list(
        cls, dicts: list[dict], target_speedup: float = 0.0
    ) -> "OptimizationLog":
        """Reconstruct an OptimizationLog from serialized round dicts.

        Used to restore trajectory context from checkpoints.
        """
        log = cls(target_speedup=target_speedup)
        for d in dicts:
            profile_data = d.get("profile", {})
            profile = ProfileMetrics(**{
                k: v for k, v in profile_data.items()
                if k in {f.name for f in ProfileMetrics.__dataclass_fields__.values()}
            }) if isinstance(profile_data, dict) else ProfileMetrics()

            status_str = d.get("status", "success")
            try:
                status = RoundStatus(status_str)
            except ValueError:
                status = RoundStatus.SUCCESS

            round_entry = OptimizationRound(
                round=d.get("round", 0),
                kernel_code=d.get("kernel_code", ""),
                is_correct=d.get("is_correct", False),
                speedup=d.get("speedup", 0.0),
                status=status,
                strategy=d.get("strategy", ""),
                bottleneck=d.get("bottleneck", ""),
                method_applied=d.get("method_applied", ""),
                profile=profile,
                worker_results=d.get("worker_results", []),
                cost_usd=d.get("cost_usd", 0.0),
                elapsed_seconds=d.get("elapsed_seconds", 0.0),
            )
            log.add_round(round_entry)
        return log
