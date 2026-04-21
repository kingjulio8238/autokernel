"""KernelBook few-shot example loading from HuggingFace.

Loads PyTorch → Triton kernel pairs from GPUMODE/KernelBook dataset
(25K pairs) for richer few-shot context during optimization.

Usage::

    from kernel_code.kernelbook import find_similar_example

    example = find_similar_example(reference_code)
    if example:
        # Use as few-shot in prompt
        ...

The dataset is cached locally after first download to avoid repeated
HuggingFace API calls. Requires the `datasets` library.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_CACHE_DIR = Path.home() / ".kernel-code" / "cache" / "kernelbook"
_DATASET_NAME = "GPUMODE/KernelBook"
_CACHE_FILE = _CACHE_DIR / "examples.json"

# Lazily loaded examples
_examples: list[dict] | None = None


def _extract_keywords(code: str) -> set[str]:
    """Extract relevant keywords from kernel/reference code."""
    code_lower = code.lower()
    keywords = set()

    # Operation keywords
    ops = [
        "matmul", "gemm", "conv", "conv2d", "conv1d",
        "softmax", "relu", "gelu", "sigmoid", "tanh",
        "layernorm", "batchnorm", "batch_norm", "layer_norm",
        "attention", "flash", "dropout", "linear",
        "reduce", "sum", "mean", "max", "min",
        "transpose", "permute", "reshape", "view",
        "elementwise", "add", "mul", "fused",
        "pooling", "max_pool", "avg_pool",
        "embedding", "scatter", "gather",
        "cross_entropy", "loss", "backward",
        "fft", "sort", "topk", "cumsum",
        "vectoradd", "vector_add", "saxpy",
    ]
    for op in ops:
        if op in code_lower:
            keywords.add(op)

    # Triton-specific patterns
    if "triton" in code_lower:
        keywords.add("triton")
    if "tl.load" in code_lower or "tl.store" in code_lower:
        keywords.add("triton_memory")
    if "@triton.autotune" in code_lower:
        keywords.add("autotune")
    if "tl.dot" in code_lower:
        keywords.add("triton_dot")

    return keywords


def load_kernelbook_examples(limit: int = 500) -> list[dict]:
    """Load KernelBook examples from HuggingFace, with local caching.

    Returns a list of dicts with keys:
    - 'pytorch': str — PyTorch reference code
    - 'triton': str — Triton kernel code
    - 'keywords': list[str] — extracted keywords for matching

    Returns empty list if datasets library is unavailable or download fails.
    """
    global _examples
    if _examples is not None:
        return _examples

    # Try loading from cache first
    if _CACHE_FILE.is_file():
        try:
            cached = json.loads(_CACHE_FILE.read_text())
            if isinstance(cached, list) and len(cached) > 0:
                _examples = cached
                logger.debug("Loaded %d KernelBook examples from cache", len(_examples))
                return _examples
        except Exception:
            pass

    # Download from HuggingFace
    try:
        from datasets import load_dataset
    except ImportError:
        logger.debug("datasets library not installed — KernelBook unavailable")
        _examples = []
        return _examples

    try:
        logger.info("Downloading KernelBook from HuggingFace...")
        ds = load_dataset(_DATASET_NAME, split="train")

        examples = []
        for i, row in enumerate(ds):
            if i >= limit:
                break

            pytorch_code = row.get("python_code", row.get("pytorch_code", row.get("pytorch", "")))
            triton_code = row.get("triton_code", row.get("original_triton_code", row.get("triton", "")))

            if not pytorch_code or not triton_code:
                continue

            keywords = list(_extract_keywords(pytorch_code) | _extract_keywords(triton_code))
            examples.append({
                "pytorch": pytorch_code,
                "triton": triton_code,
                "keywords": keywords,
            })

        # Warn if most examples were skipped (likely column name mismatch)
        scanned = min(limit, len(ds))
        if scanned > 0 and len(examples) / scanned < 0.2:
            logger.warning(
                "Only %d/%d KernelBook rows had valid code — "
                "dataset column names may have changed. "
                "Check GPUMODE/KernelBook schema.",
                len(examples), scanned,
            )

        # Cache locally (compact JSON to save disk)
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _CACHE_FILE.write_text(json.dumps(examples))

        _examples = examples
        return _examples

    except Exception as exc:
        logger.warning("Failed to load KernelBook: %s", exc)
        _examples = []
        return _examples


def find_similar_example(reference_code: str, top_k: int = 1) -> str | None:
    """Find the most relevant KernelBook Triton example for a reference.

    Uses keyword matching (same approach as the skills library matcher).

    Args:
        reference_code: The PyTorch reference code to match against
        top_k: Number of top matches to consider (returns best)

    Returns:
        Triton code string from the best matching example, or None.
    """
    examples = load_kernelbook_examples()
    if not examples:
        return None

    ref_keywords = _extract_keywords(reference_code)
    if not ref_keywords:
        return None

    # Score each example by keyword overlap
    scored = []
    for ex in examples:
        ex_keywords = set(ex.get("keywords", []))
        overlap = len(ref_keywords & ex_keywords)
        if overlap > 0:
            scored.append((overlap, ex))

    if not scored:
        return None

    # Return best match's triton code
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1].get("triton", "")
