"""Hierarchical settings for kernel-code (global -> project -> local).

Provides a Claude Code-style settings hierarchy where engineers configure
their defaults once and every ``/optimize`` uses them automatically.

Merge order (later wins):
    1. ``~/.kernel-code/settings.yaml``  -- global user defaults
    2. ``.kernel-code/settings.yaml``    -- project defaults (committed)
    3. ``.kernel-code/settings.local.yaml`` -- personal overrides (gitignored)

CLI flags override everything.

Usage::

    from kernel_code.settings import load_settings, settings_to_config

    settings = load_settings()
    config_kwargs = settings_to_config(settings)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Sentinel to distinguish "not set" from an explicit value.
_UNSET = object()

# Settings file names and search locations.
_GLOBAL_DIR = Path.home() / ".kernel-code"
_PROJECT_DIR_NAME = ".kernel-code"
_SETTINGS_FILE = "settings.yaml"
_LOCAL_SETTINGS_FILE = "settings.local.yaml"

# Known setting keys and their types (for /config set validation).
_FIELD_TYPES: dict[str, type] = {
    "default_model": str,
    "default_provider": str,
    "default_backend": str,
    "default_gpu": str,
    "max_budget": float,
    "auto_confirm_under": float,
    "auto_save": bool,
    "capture_traces": bool,
    "dashboard_port": int,
    "show_profiling": bool,
    "show_trajectory": bool,
}


@dataclass
class KernelCodeSettings:
    """Merged settings from global -> project -> local."""

    # Model
    default_model: str = "openai/MiniMax-M2.5"
    default_provider: str = "minimax"

    # Backend
    default_backend: str = "triton"
    default_gpu: str = "L40S"

    # Budget
    max_budget: float | None = None  # per-session spending cap
    auto_confirm_under: float = 0.10  # auto-approve costs below this

    # Behavior
    auto_save: bool = True  # auto-save best kernel after optimization
    capture_traces: bool = True
    dashboard_port: int = 8050

    # Display
    show_profiling: bool = True
    show_trajectory: bool = True

    # Internal: tracks which files were merged (in order)
    source_files: list[str] = field(default_factory=list)

    # Internal: tracks where each setting's value came from
    _origins: dict[str, str] = field(default_factory=dict, repr=False)


# ------------------------------------------------------------------
# Loading
# ------------------------------------------------------------------


def _find_project_dir(start: Path | None = None) -> Path | None:
    """Walk up from *start* looking for a ``.kernel-code/`` directory."""
    directory = (start or Path.cwd()).resolve()
    for _ in range(20):
        candidate = directory / _PROJECT_DIR_NAME
        if candidate.is_dir():
            return candidate
        parent = directory.parent
        if parent == directory:
            break
        directory = parent
    return None


def _load_yaml(path: Path) -> dict:
    """Load a YAML file, returning an empty dict on any error."""
    if not path.is_file():
        return {}
    try:
        import yaml  # type: ignore[import-untyped]

        with open(path) as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        logger.warning("Failed to load %s: %s", path, exc)
        return {}


def _coerce(key: str, raw: object) -> object:
    """Coerce *raw* to the expected type for *key*."""
    expected = _FIELD_TYPES.get(key)
    if expected is None:
        return raw
    if expected is bool:
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, str):
            return raw.lower() in ("true", "yes", "1")
        return bool(raw)
    if expected is float:
        if raw is None:
            return None
        return float(raw)
    return expected(raw)


def _merge(settings: KernelCodeSettings, data: dict, source_label: str) -> None:
    """Merge *data* into *settings*, recording the source of each key."""
    for key, raw_value in data.items():
        if key.startswith("_") or key == "source_files":
            continue
        if not hasattr(settings, key):
            logger.debug("Ignoring unknown setting key: %s", key)
            continue
        value = _coerce(key, raw_value)
        setattr(settings, key, value)
        settings._origins[key] = source_label


def load_settings(start_dir: Path | None = None) -> KernelCodeSettings:
    """Load and merge settings from 3 levels:

    1. ``~/.kernel-code/settings.yaml``  (global defaults)
    2. ``.kernel-code/settings.yaml``    (project)
    3. ``.kernel-code/settings.local.yaml`` (personal, gitignored)

    Later files override earlier ones.  CLI flags override all (handled by
    the caller after this function returns).
    """
    settings = KernelCodeSettings()

    # Mark built-in defaults
    for key in _FIELD_TYPES:
        settings._origins[key] = "default"

    # 1. Global
    global_path = _GLOBAL_DIR / _SETTINGS_FILE
    global_data = _load_yaml(global_path)
    if global_data:
        _merge(settings, global_data, f"global ({global_path})")
        settings.source_files.append(str(global_path))

    # 2. Project
    project_dir = _find_project_dir(start_dir)
    if project_dir is not None:
        project_path = project_dir / _SETTINGS_FILE
        project_data = _load_yaml(project_path)
        if project_data:
            _merge(settings, project_data, f"project ({project_path})")
            settings.source_files.append(str(project_path))

        # 3. Local (gitignored)
        local_path = project_dir / _LOCAL_SETTINGS_FILE
        local_data = _load_yaml(local_path)
        if local_data:
            _merge(settings, local_data, f"local ({local_path})")
            settings.source_files.append(str(local_path))

    return settings


# ------------------------------------------------------------------
# Saving
# ------------------------------------------------------------------


def save_settings(settings: KernelCodeSettings, path: Path) -> None:
    """Save settings to a YAML file.

    Only writes the setting fields (not internal metadata like
    ``source_files`` or ``_origins``).
    """
    import yaml  # type: ignore[import-untyped]

    data: dict = {}
    for key in _FIELD_TYPES:
        value = getattr(settings, key, _UNSET)
        if value is not _UNSET:
            data[key] = value

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def save_project_setting(key: str, value: object, start_dir: Path | None = None) -> Path:
    """Update a single key in the project settings file.

    Creates ``.kernel-code/settings.yaml`` if it doesn't exist.
    Returns the path that was written.
    """
    import yaml  # type: ignore[import-untyped]

    project_dir = _find_project_dir(start_dir)
    if project_dir is None:
        project_dir = (start_dir or Path.cwd()).resolve() / _PROJECT_DIR_NAME
        project_dir.mkdir(parents=True, exist_ok=True)

    path = project_dir / _SETTINGS_FILE
    data = _load_yaml(path)
    data[key] = _coerce(key, value)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    return path


# ------------------------------------------------------------------
# Conversion to OpenKernelConfig kwargs
# ------------------------------------------------------------------


def settings_to_config(settings: KernelCodeSettings) -> dict:
    """Convert settings to kwargs for ``OpenKernelConfig()``.

    Returns a dict that can be unpacked into the OpenKernelConfig constructor,
    mapping kernel-code setting names to the openkernel config field names.
    """
    from openkernel.config import Backend, GpuType

    backend_map = {"triton": Backend.TRITON, "cuda": Backend.CUDA}
    gpu_map = {g.value: g for g in GpuType}

    kwargs: dict = {}

    # Backend
    if settings.default_backend in backend_map:
        kwargs["backend"] = backend_map[settings.default_backend]

    # GPU
    if settings.default_gpu in gpu_map:
        kwargs["modal"] = {"gpu_type": gpu_map[settings.default_gpu]}

    # Model
    kwargs["model"] = {
        "provider": settings.default_provider,
        "model_id": settings.default_model,
    }

    # Traces
    kwargs["capture_traces"] = settings.capture_traces

    return kwargs
