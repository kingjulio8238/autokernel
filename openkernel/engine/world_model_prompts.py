"""LLM prompts for world-model tree operations.

Each function returns a prompt string that instructs an LLM to return
structured JSON. The orchestrator is responsible for calling the LLM and
parsing the response.

Three core operations:
- propose_intents: Insert new child intents into the tree
- update_priorities: Re-estimate node priorities after observing results
- prune_decision: Decide whether to prune a failing subtree
"""

from __future__ import annotations

import json


def prompt_propose_intents(
    reference_code: str,
    hardware: str,
    current_tree_json: dict,
    current_best_speedup: float,
    backend: str,
) -> str:
    """Build a prompt that asks the LLM to propose new optimization intents.

    The LLM should analyze the reference code, the current state of the search
    tree, and the hardware target, then propose 2-5 new child intents attached
    to existing nodes.

    Returns a prompt string expecting a JSON response.
    """
    tree_str = json.dumps(current_tree_json, indent=2)

    return f"""\
You are an expert GPU kernel optimization strategist. Your job is to propose
new optimization intents for the search tree.

## Context

**Backend**: {backend}
**Hardware**: {hardware}
**Current best speedup**: {current_best_speedup:.3f}x

### Reference Code (PyTorch)
```python
{reference_code}
```

### Current Search Tree
```json
{tree_str}
```

## Instructions

Analyze the reference code and the current search tree state. Propose 2-5 new
optimization intents as children of existing nodes. Each intent should describe
WHAT to optimize, not HOW to implement it (that is the inner loop's job).

Consider:
1. What bottlenecks likely exist in the reference code?
2. Which existing tree nodes could benefit from more specific sub-intents?
3. Are there unexplored optimization directions?
4. Non-monotonic paths are OK — a temporary regression may unlock a better path.

For {backend} backend, consider backend-specific opportunities:
- Triton: autotune configs, tiling strategies, vectorized loads, shared memory
- CUDA: warp-level primitives, tensor cores, coalesced access, launch configs

## Response Format

Return ONLY valid JSON with this structure:
```json
{{
  "intents": [
    {{
      "parent_id": "<id of existing node to attach to>",
      "description": "<what to optimize — strategy-level, not implementation>",
      "priority": <float 0-1, estimated value of this direction>,
      "rationale": "<why this intent is worth exploring>"
    }}
  ]
}}
```

Respond with ONLY the JSON object, no other text."""


def prompt_update_priorities(
    tree_json: dict,
    latest_eval_result_summary: str,
) -> str:
    """Build a prompt that asks the LLM to re-estimate priorities.

    After observing new results, the LLM should update priority scores
    for all pending nodes in the tree based on what was learned.

    Returns a prompt string expecting a JSON response.
    """
    tree_str = json.dumps(tree_json, indent=2)

    return f"""\
You are an expert GPU kernel optimization strategist. Your job is to re-estimate
the priority of pending optimization intents based on the latest results.

## Current Search Tree
```json
{tree_str}
```

## Latest Result
{latest_eval_result_summary}

## Instructions

Based on the latest evaluation result and the overall tree state, re-estimate
priorities for all PENDING nodes. Consider:

1. Did the latest result reveal new information about which directions are promising?
2. Should sibling intents of a successful node get higher priority (similar approach)?
3. Should sibling intents of a failed node get lower priority (similar approach unlikely to work)?
4. Non-monotonic insight: even if a parent regressed, its children might still be valuable
   if the regression was due to implementation quality rather than strategy quality.
5. Has a particular branch been over-explored relative to others?

## Response Format

Return ONLY valid JSON with this structure:
```json
{{
  "priority_updates": {{
    "<node_id>": <new_priority_float_0_to_1>,
    "<node_id>": <new_priority_float_0_to_1>
  }},
  "reasoning": "<brief explanation of the priority changes>"
}}
```

Include ALL pending nodes in the response, even if their priority stays the same.
Respond with ONLY the JSON object, no other text."""


def prompt_prune_decision(
    tree_json: dict,
    failed_node_summary: str,
) -> str:
    """Build a prompt that asks the LLM whether to prune a failing subtree.

    When a node has exhausted its attempts or a branch is stagnating, the LLM
    decides whether to prune it or keep exploring children.

    Returns a prompt string expecting a JSON response.
    """
    tree_str = json.dumps(tree_json, indent=2)

    return f"""\
You are an expert GPU kernel optimization strategist. A branch of the search
tree is failing. You must decide whether to prune it or keep exploring.

## Current Search Tree
```json
{tree_str}
```

## Failed Node Details
{failed_node_summary}

## Instructions

Analyze whether the failing node/branch should be pruned. Consider:

1. Is the strategy fundamentally flawed, or was it just poorly implemented?
2. Do the node's children (if any) show promise despite the parent's failure?
   (Non-monotonic paths: a child can succeed even if its parent failed.)
3. Has enough evidence accumulated to confidently declare this direction dead?
4. What is the opportunity cost of continuing this branch vs exploring others?
5. Is the profiler summary indicative of a structural limitation or a fixable issue?

## Response Format

Return ONLY valid JSON with this structure:
```json
{{
  "prune": <true or false>,
  "node_ids_to_prune": ["<node_id>", ...],
  "reasoning": "<detailed explanation of the decision>",
  "alternative_suggestion": "<if pruning, suggest what to try instead (optional)>"
}}
```

If `prune` is false, `node_ids_to_prune` should be an empty list.
Respond with ONLY the JSON object, no other text."""
