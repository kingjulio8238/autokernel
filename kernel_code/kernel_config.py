"""KERNEL.md / kernel.toml discovery and parsing.

Provides project-level kernel optimization configuration, analogous to how
Claude Code uses CLAUDE.md for persistent project instructions.  A KERNEL.md
(Markdown with YAML frontmatter) or kernel.toml file at the repo root (or any
parent directory) is automatically discovered and loaded on shell startup.

The parsed config is available as a :class:`KernelConfig` dataclass whose
``raw_text`` field preserves the full file content for direct injection into
LLM prompts.

Usage::

    from kernel_code.kernel_config import discover_kernel_config, inject_config_context

    config = discover_kernel_config()
    if config:
        context = inject_config_context(config)
        # prepend *context* to every LLM prompt
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# File names we search for, in priority order.
_CONFIG_FILENAMES = ("KERNEL.md", "kernel.toml")

# Maximum number of parent directories to traverse.
_MAX_DEPTH = 20


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class KernelConfig:
    """Parsed kernel optimization configuration."""

    backend: str = "triton"
    hardware: str = "L40S"
    max_shared_memory_kb: int | None = None
    target_occupancy: float | None = None
    must_beat_baseline: bool = True
    preferred_skills: list[str] = field(default_factory=list)
    optimization_hints: list[str] = field(default_factory=list)
    constraints: dict = field(default_factory=dict)
    raw_text: str = ""  # full KERNEL.md content for LLM injection
    source_path: Path | None = None  # where the config was loaded from


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def discover_kernel_config(start_dir: Path | None = None) -> KernelConfig | None:
    """Search *start_dir* and its parents for ``KERNEL.md`` or ``kernel.toml``.

    Walks upward at most :data:`_MAX_DEPTH` levels.  Returns ``None`` if no
    config file is found.
    """
    directory = (start_dir or Path.cwd()).resolve()

    for _ in range(_MAX_DEPTH):
        for name in _CONFIG_FILENAMES:
            candidate = directory / name
            if candidate.is_file():
                logger.info("Discovered kernel config: %s", candidate)
                if name.endswith(".md"):
                    return load_kernel_md(candidate)
                else:
                    return load_kernel_toml(candidate)

        parent = directory.parent
        if parent == directory:
            break  # reached filesystem root
        directory = parent

    return None


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_kernel_md(path: Path) -> KernelConfig:
    """Parse a ``KERNEL.md`` file (Markdown with optional YAML frontmatter).

    Frontmatter is delimited by ``---`` lines at the top of the file.  Keys in
    the frontmatter map directly to :class:`KernelConfig` fields.  The body
    text (everything after the frontmatter) is scanned for list items under
    ``## Constraints`` and ``## Hints`` headings.
    """
    raw = path.read_text(encoding="utf-8")
    fm, body = _split_frontmatter(raw)

    config = KernelConfig(raw_text=raw, source_path=path)

    # -- Apply frontmatter fields ------------------------------------------
    if fm:
        _apply_frontmatter(config, fm)

    # -- Extract constraints / hints from body -----------------------------
    _extract_body_sections(config, body)

    return config


def load_kernel_toml(path: Path) -> KernelConfig:
    """Parse a ``kernel.toml`` file.

    Expects a flat table or ``[kernel]`` section with the same keys as
    :class:`KernelConfig`.
    """
    import tomllib

    raw = path.read_text(encoding="utf-8")
    data = tomllib.loads(raw)

    # Allow a top-level [kernel] table or flat keys.
    section = data.get("kernel", data)

    config = KernelConfig(raw_text=raw, source_path=path)
    _apply_dict(config, section)
    return config


# ---------------------------------------------------------------------------
# LLM prompt injection
# ---------------------------------------------------------------------------


def inject_config_context(config: KernelConfig) -> str:
    """Format *config* as an LLM-ready context string.

    The output is designed to be prepended to the system prompt so that every
    optimization decision is shaped by the project-level constraints.
    """
    parts: list[str] = []

    parts.append("=== Project Kernel Config (KERNEL.md) ===")
    parts.append(f"Backend: {config.backend}")
    parts.append(f"Hardware: {config.hardware}")

    if config.target_occupancy is not None:
        parts.append(f"Target occupancy: {config.target_occupancy}")
    if config.max_shared_memory_kb is not None:
        parts.append(f"Max shared memory: {config.max_shared_memory_kb} KB")
    if config.must_beat_baseline:
        parts.append("MUST beat baseline performance")

    if config.preferred_skills:
        parts.append(f"Preferred skills: {', '.join(config.preferred_skills)}")

    if config.constraints:
        parts.append("\nConstraints:")
        for key, val in config.constraints.items():
            parts.append(f"  {key}: {val}")

    if config.optimization_hints:
        parts.append("\nOptimization hints:")
        for hint in config.optimization_hints:
            parts.append(f"  - {hint}")

    # Always include the raw text so the LLM can read free-form notes.
    if config.raw_text:
        parts.append(f"\n--- Full KERNEL.md ---\n{config.raw_text.strip()}")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Regex for YAML frontmatter delimiters.
_FM_RE = re.compile(r"\A---\s*\n(.*?\n)---\s*\n", re.DOTALL)

# Regex for Markdown section headings.
_HEADING_RE = re.compile(r"^##\s+(.+)", re.MULTILINE)


def _split_frontmatter(text: str) -> tuple[dict | None, str]:
    """Split YAML frontmatter from the body of a Markdown file.

    Returns ``(frontmatter_dict | None, body_text)``.
    """
    match = _FM_RE.match(text)
    if not match:
        return None, text

    fm_text = match.group(1)
    body = text[match.end():]

    try:
        import yaml  # type: ignore[import-untyped]

        fm = yaml.safe_load(fm_text)
        if not isinstance(fm, dict):
            fm = None
    except Exception:
        # Fallback: simple key: value parsing for common scalars.
        fm = _parse_simple_yaml(fm_text)

    return fm, body


def _parse_simple_yaml(text: str) -> dict:
    """Minimalist key-value parser for YAML frontmatter (no dependency)."""
    result: dict = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip()
        val = val.strip()
        # Coerce common types.
        if val.lower() in ("true", "yes"):
            result[key] = True
        elif val.lower() in ("false", "no"):
            result[key] = False
        else:
            try:
                result[key] = int(val)
            except ValueError:
                try:
                    result[key] = float(val)
                except ValueError:
                    result[key] = val
    return result


def _apply_frontmatter(config: KernelConfig, fm: dict) -> None:
    """Apply frontmatter dict to a :class:`KernelConfig`."""
    _apply_dict(config, fm)


def _apply_dict(config: KernelConfig, data: dict) -> None:
    """Set :class:`KernelConfig` fields from a dict, ignoring unknown keys."""
    if "backend" in data:
        config.backend = str(data["backend"])
    if "hardware" in data:
        config.hardware = str(data["hardware"])
    if "max_shared_memory_kb" in data:
        config.max_shared_memory_kb = int(data["max_shared_memory_kb"])
    if "target_occupancy" in data:
        config.target_occupancy = float(data["target_occupancy"])
    if "must_beat_baseline" in data:
        config.must_beat_baseline = bool(data["must_beat_baseline"])
    if "preferred_skills" in data:
        val = data["preferred_skills"]
        config.preferred_skills = list(val) if isinstance(val, (list, tuple)) else [str(val)]
    if "optimization_hints" in data:
        val = data["optimization_hints"]
        config.optimization_hints = list(val) if isinstance(val, (list, tuple)) else [str(val)]
    if "constraints" in data and isinstance(data["constraints"], dict):
        config.constraints.update(data["constraints"])


def _extract_body_sections(config: KernelConfig, body: str) -> None:
    """Extract list items from ``## Constraints`` and ``## Hints`` sections."""
    sections = _split_sections(body)

    for heading, content in sections.items():
        items = _extract_list_items(content)
        heading_lower = heading.lower()
        if "constraint" in heading_lower:
            for item in items:
                # Store as key-value if a colon is present, else as a note.
                if ":" in item:
                    key, _, val = item.partition(":")
                    config.constraints[key.strip()] = val.strip()
                else:
                    config.constraints[item] = True
        elif "hint" in heading_lower:
            config.optimization_hints.extend(items)
        elif "skill" in heading_lower:
            config.preferred_skills.extend(items)


def _split_sections(body: str) -> dict[str, str]:
    """Split a Markdown body into ``{heading: content}`` pairs by ``##`` headings."""
    headings = list(_HEADING_RE.finditer(body))
    sections: dict[str, str] = {}
    for i, m in enumerate(headings):
        heading = m.group(1).strip()
        start = m.end()
        end = headings[i + 1].start() if i + 1 < len(headings) else len(body)
        sections[heading] = body[start:end]
    return sections


def _extract_list_items(text: str) -> list[str]:
    """Extract Markdown list items (``- ...``) from *text*."""
    items: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            items.append(stripped[2:].strip())
    return items
