"""Centralized mock implementations for testing.

All mock classes used in tests live here — production code (openkernel/,
kernel_code/, modal_infra/) contains zero mock implementations.

Classes:
    MockEvalFn         — mock async evaluation function for inner loop tests
    MockInnerLoop      — mock inner loop for orchestrator tests
    MockLLMCaller      — mock LLM for orchestrator tree operations
    MockProblemLoader  — synthetic KernelBench problems

Functions:
    create_mock_eval_fn() — factory that returns a MockEvalFn instance
"""

from __future__ import annotations

import json
import random
from typing import Any

from openkernel.engine.orchestrator import RefinementResult
from openkernel.engine.world_model import IntentNode
from openkernel.eval.types import (
    BottleneckType,
    EvalResult,
    EvalStatus,
    ProfileData,
)


# ---------------------------------------------------------------------------
# MockEvalFn  (was in openkernel/engine/inner_loop.py)
# ---------------------------------------------------------------------------


class MockEvalFn:
    """A mock evaluation function for testing the inner loop without Modal.

    Produces random but trending results: starts with some errors, then
    increasingly correct results with improving speedup.
    """

    def __init__(
        self,
        error_rate: float = 0.2,
        incorrect_rate: float = 0.1,
        base_speedup: float = 0.8,
        speedup_improvement: float = 0.15,
        seed: int | None = None,
    ) -> None:
        self._error_rate = error_rate
        self._incorrect_rate = incorrect_rate
        self._base_speedup = base_speedup
        self._speedup_improvement = speedup_improvement
        self._rng = random.Random(seed)
        self._call_count = 0

    async def __call__(self, kernel_code: str, reference: str) -> EvalResult:
        self._call_count += 1
        roll = self._rng.random()

        # Decrease error probability over attempts (simulate LLM learning from feedback)
        adjusted_error_rate = self._error_rate * max(0.2, 1.0 - 0.15 * self._call_count)
        adjusted_incorrect_rate = self._incorrect_rate * max(0.2, 1.0 - 0.1 * self._call_count)

        if roll < adjusted_error_rate:
            return EvalResult(
                status=EvalStatus.COMPILE_ERROR,
                error="mock compile error: undefined symbol 'tl.store'",
            )

        if roll < adjusted_error_rate + adjusted_incorrect_rate:
            return EvalResult(
                status=EvalStatus.INCORRECT,
                error="torch.allclose failed: max diff = 0.0123",
            )

        # Correct — compute a trending speedup
        noise = self._rng.gauss(0, 0.1)
        speedup = (
            self._base_speedup
            + self._speedup_improvement * self._call_count
            + noise
        )
        speedup = max(0.5, speedup)  # floor at 0.5x

        ref_runtime = 100.0  # microseconds
        kernel_runtime = ref_runtime / speedup

        return EvalResult(
            status=EvalStatus.CORRECT,
            correct=True,
            speedup=speedup,
            runtime_us=kernel_runtime,
            ref_runtime_us=ref_runtime,
            profile=ProfileData(
                bottleneck_type=self._rng.choice(list(BottleneckType)),
                roofline_position=self._rng.uniform(0.3, 0.9),
                cache_efficiency=self._rng.uniform(0.4, 0.95),
                occupancy=self._rng.uniform(0.3, 0.95),
                bandwidth_utilization=self._rng.uniform(0.2, 0.9),
                compute_utilization=self._rng.uniform(0.2, 0.9),
                top_stalls=self._rng.sample(
                    [
                        "memory_dependency",
                        "execution_dependency",
                        "instruction_fetch",
                        "synchronization",
                        "pipe_busy",
                    ],
                    k=2,
                ),
            ),
            eval_seconds=self._rng.uniform(1.0, 5.0),
        )


# ---------------------------------------------------------------------------
# MockInnerLoop  (was in openkernel/engine/orchestrator.py)
# ---------------------------------------------------------------------------


class MockInnerLoop:
    """Mock inner loop that returns randomized results for testing.

    Produces a mix of successes and failures with plausible speedups.
    The seed can be set for reproducibility.
    """

    def __init__(self, seed: int | None = None, success_rate: float = 0.6) -> None:
        self._rng = random.Random(seed)
        self._success_rate = success_rate

    def refine(
        self,
        intent: IntentNode,
        reference_code: str,
        backend: str,
        config: dict,
    ) -> RefinementResult:
        iterations = self._rng.randint(1, config.get("max_retries_per_intent", 5))
        succeeded = self._rng.random() < self._success_rate

        if succeeded:
            # Speedup between 0.8x and 3.5x — intentionally allows regression
            # (< 1.0) to test non-monotonic path handling.
            speedup = round(self._rng.uniform(0.8, 3.5), 3)
            return RefinementResult(
                status="succeeded",
                best_kernel=f"# mock kernel for intent: {intent.description}\n"
                f"# speedup: {speedup}x\n"
                f"import triton\n",
                best_speedup=speedup,
                iterations=iterations,
                critic_feedback=f"Mock diagnosis: intent '{intent.description}' achieved {speedup}x. "
                f"Bottleneck: memory_bound. Recommendation: try tiling.",
            )
        else:
            return RefinementResult(
                status="failed",
                best_kernel="",
                best_speedup=0.0,
                iterations=iterations,
                critic_feedback=f"Mock diagnosis: intent '{intent.description}' failed after "
                f"{iterations} iterations. Compile errors on all attempts.",
            )


# ---------------------------------------------------------------------------
# MockLLMCaller  (was in openkernel/engine/orchestrator.py)
# ---------------------------------------------------------------------------


class MockLLMCaller:
    """Mock LLM that returns plausible structured JSON for tree operations.

    Used for testing the orchestrator without a real LLM.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def call(self, prompt: str) -> str:
        """Return mock JSON based on what type of prompt this is.

        Detection order matters: check the most specific patterns first to
        avoid false matches (e.g., a priority prompt mentioning "propose").
        """
        lower = prompt.lower()
        # Prune prompts contain "you must decide whether to prune"
        if "decide whether to prune" in lower:
            return self._mock_prune()
        # Priority prompts contain "re-estimate the priority"
        if "re-estimate" in lower and "priority" in lower:
            return self._mock_priorities(prompt)
        # Propose prompts contain "propose new optimization intents"
        if "propose" in lower:
            return self._mock_propose(prompt)
        # Default: propose intents
        return self._mock_propose(prompt)

    def _mock_propose(self, prompt: str) -> str:
        """Generate mock intent proposals."""
        intent_descriptions = [
            "Vectorize memory loads with float4 to improve bandwidth utilization",
            "Apply shared memory tiling to reduce global memory accesses",
            "Fuse adjacent operations to eliminate intermediate memory writes",
            "Optimize thread block dimensions for better occupancy",
            "Use warp-level reduction to minimize synchronization overhead",
        ]
        chosen = self._rng.sample(
            intent_descriptions, k=min(3, len(intent_descriptions))
        )
        intents = []
        for desc in chosen:
            intents.append({
                "parent_id": "__ROOT__",
                "description": desc,
                "priority": round(self._rng.uniform(0.3, 0.9), 2),
                "rationale": f"Mock rationale for: {desc}",
            })
        return json.dumps({"intents": intents})

    def _mock_priorities(self, prompt: str) -> str:
        """Generate mock priority updates (empty — no pending IDs available)."""
        return json.dumps({
            "priority_updates": {},
            "reasoning": "Mock reasoning: priorities unchanged.",
        })

    def _mock_prune(self) -> str:
        """Generate mock prune decision (don't prune by default)."""
        return json.dumps({
            "prune": False,
            "node_ids_to_prune": [],
            "reasoning": "Mock reasoning: not enough evidence to prune yet.",
            "alternative_suggestion": "",
        })


# ---------------------------------------------------------------------------
# MockProblemLoader  (was in openkernel/kernelbench/problems.py)
# ---------------------------------------------------------------------------

# Synthetic reference sources covering common kernel patterns.
_MOCK_PROBLEMS: dict[str, str] = {
    "matmul": '''\
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return torch.matmul(A, B)

M = 4096
K = 4096
N = 4096

def get_inputs():
    A = torch.randn(M, K)
    B = torch.randn(K, N)
    return [A, B]

def get_init_inputs():
    return []
''',
    "elementwise_add": '''\
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return A + B

def get_inputs():
    A = torch.randn(1024, 1024)
    B = torch.randn(1024, 1024)
    return [A, B]

def get_init_inputs():
    return []
''',
    "softmax": '''\
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(x, dim=-1)

def get_inputs():
    x = torch.randn(32, 16384)
    return [x]

def get_init_inputs():
    return []
''',
    "layernorm": '''\
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.ln = nn.LayerNorm(1024)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(x)

def get_inputs():
    x = torch.randn(32, 128, 1024)
    return [x]

def get_init_inputs():
    return []
''',
    "relu": '''\
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x)

def get_inputs():
    x = torch.randn(1024, 1024)
    return [x]

def get_init_inputs():
    return []
''',
    "reduction_sum": '''\
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sum(x, dim=-1)

def get_inputs():
    x = torch.randn(32, 16384)
    return [x]

def get_init_inputs():
    return []
''',
    "conv2d": '''\
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(64, 128, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

def get_inputs():
    x = torch.randn(8, 64, 56, 56)
    return [x]

def get_init_inputs():
    return []
''',
    "transpose": '''\
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(1, 2).contiguous()

def get_inputs():
    x = torch.randn(32, 128, 256)
    return [x]

def get_init_inputs():
    return []
''',
    "gelu": '''\
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.gelu(x)

def get_inputs():
    x = torch.randn(1024, 1024)
    return [x]

def get_init_inputs():
    return []
''',
    "bmm": '''\
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return torch.bmm(A, B)

def get_inputs():
    A = torch.randn(16, 512, 256)
    B = torch.randn(16, 256, 512)
    return [A, B]

def get_init_inputs():
    return []
''',
}

_MOCK_PROBLEM_NAMES = list(_MOCK_PROBLEMS.keys())


class MockProblemLoader:
    """Provides synthetic KernelBench-style problems for testing.

    Each problem follows the standard format: a ``Model`` class with a
    ``forward`` method, plus ``get_inputs()`` and ``get_init_inputs()``.
    """

    @staticmethod
    def load(level: int, problem_id: int) -> dict[str, Any]:
        """Load a synthetic problem by cycling through the mock set."""
        idx = problem_id % len(_MOCK_PROBLEM_NAMES)
        name = _MOCK_PROBLEM_NAMES[idx]
        source = _MOCK_PROBLEMS[name]

        return {
            "reference_source": source,
            "problem_name": f"L{level}_mock_{name}_{problem_id}",
            "level": level,
            "problem_id": problem_id,
        }

    @staticmethod
    def get_all(level: int) -> list[dict[str, Any]]:
        """Return all mock problems for a level."""
        from openkernel.kernelbench.problems import get_problem_count

        count = get_problem_count(level)
        return [MockProblemLoader.load(level, pid) for pid in range(count)]


# ---------------------------------------------------------------------------
# create_mock_eval_fn helper  (was in openkernel/eval/__init__.py)
# ---------------------------------------------------------------------------


def create_mock_eval_fn(
    *,
    error_rate: float = 0.2,
    incorrect_rate: float = 0.1,
    base_speedup: float = 0.8,
    speedup_improvement: float = 0.15,
    seed: int | None = None,
) -> MockEvalFn:
    """Create a mock async eval function for testing.

    Returns a :class:`MockEvalFn` instance configured with the given
    parameters.
    """
    return MockEvalFn(
        error_rate=error_rate,
        incorrect_rate=incorrect_rate,
        base_speedup=base_speedup,
        speedup_improvement=speedup_improvement,
        seed=seed,
    )
