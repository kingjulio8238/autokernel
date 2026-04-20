"""Unified Problem interface — format-agnostic kernel optimization problems.

Supports multiple problem formats behind a single interface:
- KernelBench: Model/ModelNew + get_inputs()
- GPU Mode: ref_kernel/kernel_function + generate_input()
- Custom: auto-detected from file contents

The engine and eval don't care about format — they work with
the Problem interface. Adapters handle conversion.

Usage::

    from kernel_code.problem import load_problem

    problem = load_problem("reference.py")
    print(problem.format)      # "kernelbench" or "gpumode" or "custom"
    print(problem.name)        # "Square matrix multiplication"
    print(problem.submit_fn)   # "ModelNew" or "kernel_function"
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Problem formats
# ---------------------------------------------------------------------------

FORMAT_KERNELBENCH = "kernelbench"
FORMAT_GPUMODE = "gpumode"
FORMAT_CUSTOM = "custom"


@dataclass
class Problem:
    """Format-agnostic kernel optimization problem."""

    name: str = ""
    reference_code: str = ""
    reference_path: str = ""

    # Format info
    format: str = FORMAT_CUSTOM  # "kernelbench", "gpumode", "custom"

    # Function/class names
    ref_fn: str = ""       # "Model" or "ref_kernel"
    submit_fn: str = ""    # "ModelNew" or "kernel_function"

    # Input generation
    input_code: str = ""   # code that generates test inputs

    # Metadata
    dtype: str = "float32"
    source: str = ""       # "kernelbench L1.5", "gpumode/pmpp_v2/vectoradd", etc.
    difficulty: str = ""   # "easy", "medium", "hard"

    # For GPU Mode: additional files
    task_code: str = ""    # task.py contents
    utils_code: str = ""   # utils.py contents


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------


def detect_format(code: str) -> str:
    """Detect the problem format from reference source code."""
    has_model_class = "class Model" in code and "def forward" in code
    has_get_inputs = "def get_inputs" in code
    has_ref_kernel = "def ref_kernel" in code
    has_generate_input = "def generate_input" in code
    has_kernel_function = "kernel_function" in code

    if has_model_class and has_get_inputs:
        return FORMAT_KERNELBENCH
    if has_ref_kernel or has_generate_input:
        return FORMAT_GPUMODE
    if has_model_class:
        return FORMAT_KERNELBENCH
    return FORMAT_CUSTOM


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_problem(path: str | Path) -> Problem:
    """Load a problem from a file, auto-detecting format."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Problem file not found: {path}")

    code = path.read_text()
    fmt = detect_format(code)

    if fmt == FORMAT_KERNELBENCH:
        return _load_kernelbench(code, path)
    elif fmt == FORMAT_GPUMODE:
        return _load_gpumode(code, path)
    else:
        return _load_custom(code, path)


def _load_kernelbench(code: str, path: Path) -> Problem:
    """Load a KernelBench-format problem."""
    name = _extract_name(code) or path.stem

    return Problem(
        name=name,
        reference_code=code,
        reference_path=str(path),
        format=FORMAT_KERNELBENCH,
        ref_fn="Model",
        submit_fn="ModelNew",
        input_code="get_inputs()",
        dtype=_detect_dtype(code),
        source=f"kernelbench ({path.name})",
    )


def _load_gpumode(code: str, path: Path) -> Problem:
    """Load a GPU Mode-format problem."""
    name = path.stem or "gpu_mode_problem"

    # Check for task.py and utils.py in same directory or parent
    task_code = ""
    utils_code = ""
    parent = path.parent
    for search_dir in [parent, parent.parent]:
        task_path = search_dir / "task.py"
        if task_path.is_file() and not task_code:
            task_code = task_path.read_text()
        utils_path = search_dir / "utils.py"
        if utils_path.is_file() and not utils_code:
            utils_code = utils_path.read_text()

    return Problem(
        name=name,
        reference_code=code,
        reference_path=str(path),
        format=FORMAT_GPUMODE,
        ref_fn="ref_kernel",
        submit_fn="kernel_function",
        input_code="generate_input(...)",
        dtype=_detect_dtype(code),
        source=f"gpumode ({path.name})",
        task_code=task_code,
        utils_code=utils_code,
    )


def _load_custom(code: str, path: Path) -> Problem:
    """Load a custom-format problem."""
    name = _extract_name(code) or path.stem

    # Auto-detect what functions exist
    has_model = "class Model" in code
    has_ref_kernel = "def ref_kernel" in code

    return Problem(
        name=name,
        reference_code=code,
        reference_path=str(path),
        format=FORMAT_CUSTOM,
        ref_fn="Model" if has_model else "ref_kernel" if has_ref_kernel else "",
        submit_fn="ModelNew" if has_model else "kernel_function",
        dtype=_detect_dtype(code),
        source=f"custom ({path.name})",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_self_contained(problem: Problem) -> str:
    """Make a GPU Mode reference self-contained for Modal eval.

    GPU Mode references import from task.py and utils.py — these aren't
    available in the Modal container. This function strips those
    dependencies and creates a clean, standalone reference.

    The key insight: our Modal eval does its own correctness checking,
    so we don't need GPU Mode's check_implementation or DeterministicContext.
    We just need ref_kernel() and generate_input().
    """
    if problem.format != FORMAT_GPUMODE:
        return problem.reference_code

    code = problem.reference_code

    # Remove utils imports and check_implementation (our eval handles correctness)
    code = re.sub(r"^from utils import.*$", "", code, flags=re.MULTILINE)
    code = re.sub(r"^check_implementation\s*=.*$", "", code, flags=re.MULTILINE)

    # Remove task imports and type annotations that reference task types
    code = re.sub(r"^from task import.*$", "", code, flags=re.MULTILINE)
    # Replace type hints with generic ones
    code = code.replace(": input_t", "")
    code = code.replace(": output_t", "")
    code = code.replace("-> output_t", "")
    code = code.replace("-> input_t", "")

    # Replace DeterministicContext with a no-op (preserves indentation)
    code = code.replace("with DeterministicContext():", "if True:  # deterministic")

    # Add minimal imports
    if "import torch" not in code:
        code = "import torch\n" + code

    # Clean up blank lines
    code = re.sub(r"\n{3,}", "\n\n", code)

    return code.strip() + "\n"


def _extract_name(code: str) -> str:
    """Extract problem name from docstring or comments."""
    # Check for KernelBench-style docstring
    for line in code.split("\n", 10):
        if "Problem" in line or "KernelBench" in line:
            return line.strip().strip('"').strip("'").strip()
        if "reference" in line.lower() and ":" in line:
            return line.split(":", 1)[1].strip().strip('"').strip("'").strip()
    return ""


def _detect_dtype(code: str) -> str:
    """Detect the primary dtype from the code."""
    if "float16" in code or "fp16" in code:
        return "float16"
    if "bfloat16" in code or "bf16" in code:
        return "bfloat16"
    if "float64" in code:
        return "float64"
    return "float32"


def build_test_code(problem: Problem) -> str:
    """Build test code appropriate for the problem format.

    This is the test code passed to KernelAgent so workers validate
    against the correct reference with the correct dtypes.
    """
    if problem.format == FORMAT_KERNELBENCH:
        return _build_kernelbench_test(problem)
    elif problem.format == FORMAT_GPUMODE:
        return _build_gpumode_test(problem)
    else:
        return _build_custom_test(problem)


def _build_kernelbench_test(problem: Problem) -> str:
    return f'''"""Test kernel_function against KernelBench reference."""
import torch, sys

{problem.reference_code}

def test_kernel():
    from kernel import kernel_function
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = get_inputs()
    inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in inputs]
    model = Model().to(device).eval()
    with torch.no_grad():
        expected = model(*inputs)
        result = kernel_function(*inputs)
    if result.dtype != expected.dtype:
        print(f"DTYPE MISMATCH: expected {{expected.dtype}}, got {{result.dtype}}")
        return False
    if not torch.allclose(result, expected, rtol=1e-2, atol=1e-2):
        diff = (result - expected).abs().max().item()
        print(f"NUMERICAL MISMATCH: max diff = {{diff}}")
        return False
    print("PASS")
    return True

if __name__ == "__main__":
    sys.exit(0 if test_kernel() else 1)
'''


def _build_gpumode_test(problem: Problem) -> str:
    return f'''"""Test kernel_function against GPU Mode reference."""
import torch, sys

{problem.reference_code}

def test_kernel():
    from kernel import kernel_function
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Use generate_input from the reference
    data = generate_input(1024, seed=42) if 'seed' in generate_input.__code__.co_varnames else generate_input(1024)
    if isinstance(data, tuple):
        data = tuple(x.to(device) if isinstance(x, torch.Tensor) else x for x in data)
    elif isinstance(data, torch.Tensor):
        data = data.to(device)
    expected = ref_kernel(data)
    # Try tuple call first, fall back to unpacked args
    try:
        result = kernel_function(data)
    except TypeError:
        if isinstance(data, tuple):
            result = kernel_function(*data)
        else:
            raise
    if isinstance(expected, torch.Tensor) and isinstance(result, torch.Tensor):
        if not torch.allclose(result, expected, rtol=1e-2, atol=1e-2):
            diff = (result - expected).abs().max().item()
            print(f"MISMATCH: max diff = {{diff}}")
            return False
    print("PASS")
    return True

if __name__ == "__main__":
    sys.exit(0 if test_kernel() else 1)
'''


def _build_custom_test(problem: Problem) -> str:
    return _build_kernelbench_test(problem)
