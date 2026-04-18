"""Tool: edit_kernel -- apply a find/replace edit to the current best kernel code."""

from __future__ import annotations

from typing import Any


def execute(session_context: dict, **kwargs: Any) -> str:
    """Apply a find/replace edit to the current best kernel code.

    Useful for making targeted modifications to the optimization.

    Required kwargs:
        find (str): the exact substring to find in the kernel code.
        replace (str): the replacement string.
    """
    find = kwargs.get("find", "")
    replace = kwargs.get("replace", "")

    if not find:
        return "Error: 'find' parameter is required."
    # 'replace' can legitimately be an empty string (deletion), so we don't
    # reject it when empty.

    # Locate the best kernel code in the session context
    best_code = session_context.get("best_kernel_code", "")
    if not best_code:
        # Fall back: look through iterations for the best kept kernel
        iterations = session_context.get("iterations", [])
        best_speedup = -1.0
        for it in iterations:
            if it.get("status") == "keep":
                sp = it.get("speedup", 0.0)
                if sp > best_speedup:
                    best_speedup = sp
                    best_code = it.get("code", "")

    if not best_code:
        return "Error: no kernel code found in session context. Run /optimize first."

    if find not in best_code:
        # Provide context to help the user fix the find string
        preview = best_code[:500]
        if len(best_code) > 500:
            preview += "\n... [truncated]"
        return (
            f"Error: find string not found in the current best kernel.\n\n"
            f"Current kernel preview:\n{preview}"
        )

    edited_code = best_code.replace(find, replace, 1)

    # Store the edited code back into the session context so subsequent tools
    # see the update.
    session_context["best_kernel_code"] = edited_code

    return edited_code
