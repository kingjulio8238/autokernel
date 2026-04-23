"""openkernel configuration.

Central configuration for eval modes, backends, models, Modal, and HF Hub.
Uses pydantic for validation.
"""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, Field

from openkernel.exceptions import ConfigurationError


class EvalMode(str, Enum):
    FAST = "fast"  # 5 correctness trials, 10 perf trials (~5s)
    THOROUGH = "thorough"  # 5 correctness trials, 100 perf trials (~15s)


class Backend(str, Enum):
    TRITON = "triton"
    CUDA = "cuda"


class GpuType(str, Enum):
    H100 = "H100"
    A100_80GB = "A100-80GB"
    A100_40GB = "A100-40GB"
    L40S = "L40S"


class ModelConfig(BaseModel):
    """LLM provider configuration (BYOM).

    Default: MiniMax M2.5 — best cost/quality/integration ratio for kernel optimization.
    80.2% SWE-Bench Verified at $0.30/$1.20 per M tokens with native OpenAI-compatible API.

    MiniMax uses OpenAI-compatible API, so litellm model_id uses the ``openai/`` prefix
    with a custom api_base pointing to ``api.minimax.io``.
    """

    provider: str = "nvidia"  # nvidia, minimax, groq, openai, anthropic, google, local, etc.
    model_id: str = "deepseek-ai/deepseek-v3.2"  # NVIDIA NIM (OpenAI-compatible, free 40 RPM)
    api_key: str | None = None  # loaded from env if not set
    api_base: str | None = None  # set per-provider; None = litellm uses provider default
    temperature: float = 0.7
    max_tokens: int = 8192


class ModalConfig(BaseModel):
    """Modal cloud GPU configuration.

    Default GPU is L40S — matches KernelBench's official evaluation hardware.
    All published baselines (CudaForge, Kernel-Smith, CUDA Agent, KernelSkill)
    report on L40S, so we target L40S for comparable results.
    Use H100 for production optimization (faster but not benchmark-comparable).
    """

    gpu_type: GpuType = GpuType.L40S
    timeout_seconds: int = 300
    max_concurrency: int = 10
    keep_warm: int = 1  # number of warm containers


class HubConfig(BaseModel):
    """Hugging Face Hub configuration."""

    org: str = "openkernel"
    token: str | None = None  # loaded from env if not set
    traces_repo: str = "openkernel/optimization-traces"
    results_repo: str = "openkernel/kernelbench-results"
    kernels_repo: str = "openkernel/optimized-kernels"
    skills_repo: str = "openkernel/skill-library"


class OpenKernelConfig(BaseModel):
    """Top-level openkernel configuration."""

    # Eval
    eval_mode: EvalMode = EvalMode.FAST
    backend: Backend = Backend.TRITON
    correctness_trials: int = 5
    perf_trials_fast: int = 10
    perf_trials_thorough: int = 100

    # Search
    max_retries_per_intent: int = 5
    stagnation_threshold: int = 7  # consecutive failures before escalating
    max_iterations: int = 100

    # Agents
    model: ModelConfig = Field(default_factory=ModelConfig)

    # Infrastructure
    modal: ModalConfig = Field(default_factory=ModalConfig)
    hub: HubConfig = Field(default_factory=HubConfig)

    # Traces
    capture_traces: bool = True
    traces_dir: str = "traces"

    # Budget
    max_budget_usd: float | None = None  # hard cost cap; None = unlimited

    # Profiling
    enable_deep_profiling: bool = False  # RunPod NCU tier (more expensive)
    analytical_prescreen: bool = True  # roofline pre-screening before GPU eval

    # -- Validation -----------------------------------------------------------

    _PROVIDER_ENV_VARS: ClassVar[dict[str, str]] = {
        "minimax": "MINIMAX_API_KEY",
        "groq": "GROQ_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "google": "GOOGLE_API_KEY",
    }

    def validate_config(self) -> None:
        """Validate configuration. Raises ConfigurationError if invalid."""
        errors: list[str] = []

        # -- LLM API key ------------------------------------------------------
        provider = self.model.provider
        if provider != "local":
            env_var = self._PROVIDER_ENV_VARS.get(provider)
            if self.model.api_key is None and (
                env_var is None or not os.environ.get(env_var)
            ):
                hint = (
                    f"Set {env_var} environment variable or pass api_key in ModelConfig."
                    if env_var
                    else f"Unknown provider '{provider}'. Set api_key in ModelConfig explicitly."
                )
                errors.append(f"No LLM API key configured for provider '{provider}'. {hint}")

        # -- Enum sanity (already enforced by pydantic, but guard against
        #    programmatic misuse with raw strings) ----------------------------
        if not isinstance(self.backend, Backend):
            try:
                Backend(self.backend)
            except ValueError:
                errors.append(
                    f"Invalid backend '{self.backend}'. Must be one of: "
                    f"{', '.join(b.value for b in Backend)}."
                )

        if not isinstance(self.eval_mode, EvalMode):
            try:
                EvalMode(self.eval_mode)
            except ValueError:
                errors.append(
                    f"Invalid eval_mode '{self.eval_mode}'. Must be one of: "
                    f"{', '.join(e.value for e in EvalMode)}."
                )

        # -- Numeric bounds ---------------------------------------------------
        if self.max_iterations <= 0:
            errors.append(
                f"max_iterations must be > 0, got {self.max_iterations}."
            )

        if self.max_retries_per_intent <= 0:
            errors.append(
                f"max_retries_per_intent must be > 0, got {self.max_retries_per_intent}."
            )

        if errors:
            raise ConfigurationError(
                "OpenKernel configuration is invalid:\n  - " + "\n  - ".join(errors)
            )

    def check_modal(self) -> bool:
        """Check if Modal is likely configured (token exists).

        Returns ``True`` when any of these are present:
        - ``MODAL_TOKEN_ID`` environment variable
        - ``~/.modal.toml`` configuration file (created by ``modal setup``)
        - ``~/.modal/`` configuration directory (legacy)
        """
        if os.environ.get("MODAL_TOKEN_ID"):
            return True
        if Path.home().joinpath(".modal.toml").is_file():
            return True
        if Path.home().joinpath(".modal").is_dir():
            return True
        return False

    # -- YAML config -----------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> "OpenKernelConfig":
        """Load configuration from a YAML file.

        YAML fields map directly to OpenKernelConfig fields. Nested objects
        (model, modal, hub) use nested dicts. Missing fields use defaults.

        Example YAML::

            backend: triton
            eval_mode: fast
            max_iterations: 50
            model:
              provider: groq
              model_id: groq/llama-3.3-70b-versatile
            modal:
              gpu_type: L40S
        """
        import yaml  # pip install pyyaml

        path = Path(path)
        if not path.exists():
            raise ConfigurationError(f"Config file not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f) or {}

        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file."""
        import yaml

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = self.model_dump(mode="json")
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
