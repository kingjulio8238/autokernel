"""Configurable output verbosity for kernel code."""


class Verbosity:
    QUIET = "quiet"    # only results, no progress, no suggestions
    NORMAL = "normal"  # current behavior
    VERBOSE = "verbose"  # full profiler dumps, raw LLM prompts


def should_show(level: str, current: str) -> bool:
    """Check if a message at 'level' should be shown given current verbosity.
    quiet shows: results only
    normal shows: results + progress + suggestions
    verbose shows: everything including raw data
    """
    order = {"quiet": 0, "normal": 1, "verbose": 2}
    return order.get(level, 1) <= order.get(current, 1)


def format_for_verbosity(content: str, verbosity: str, level: str = "normal") -> str | None:
    """Return content if it should be shown, None otherwise."""
    if should_show(level, verbosity):
        return content
    return None
