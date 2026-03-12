"""
Load a KernelBench problem into reference.py and reset kernel.py to a naive baseline.

Usage:
    uv run python scripts/setup_problem.py --level 1 --problem 1
    uv run python scripts/setup_problem.py --level 2 --problem 12
    uv run python scripts/setup_problem.py --level 1 --problem 1 --source huggingface
"""

import argparse
import re
import sys
import textwrap

from kernelbench.dataset import construct_kernelbench_dataset


def extract_forward_signature(source: str) -> str:
    """Extract the forward() method signature from the Model class to generate a matching ModelNew."""
    match = re.search(
        r"def forward\(self,(.*?)\).*?:", source, re.DOTALL
    )
    if match:
        params = match.group(1).strip()
        return params
    return "*args"


def make_naive_kernel(forward_params: str) -> str:
    """Generate a naive kernel.py that just calls the reference implementation."""
    # Extract just parameter names (strip type annotations)
    param_names = []
    for param in forward_params.split(","):
        param = param.strip()
        name = param.split(":")[0].strip()
        if name:
            param_names.append(name)

    args_str = ", ".join(param_names)

    return textwrap.dedent(f'''\
        """
        Autokernel — agent-modifiable kernel file.
        Must define ModelNew with the same forward() signature as Model in reference.py.
        Evaluated by prepare.py via KernelBench's eval harness.

        This starter version just wraps the reference implementation (functionally correct, no speedup).
        The agent will iteratively replace this with optimized Triton/CUDA kernels.
        """

        import torch
        import torch.nn as nn


        class ModelNew(nn.Module):
            def __init__(self):
                super(ModelNew, self).__init__()

            def forward(self, {forward_params}) -> torch.Tensor:
                return torch.matmul({args_str})
    ''')


def make_passthrough_kernel(source: str, forward_params: str) -> str:
    """Generate a kernel.py that delegates to the reference Model.
    Used when the forward() body is too complex to naively replicate."""
    return textwrap.dedent(f'''\
        """
        Autokernel — agent-modifiable kernel file.
        Must define ModelNew with the same forward() signature as Model in reference.py.
        Evaluated by prepare.py via KernelBench's eval harness.

        This starter version delegates to the reference Model (functionally correct, ~1.0x speedup).
        The agent will iteratively replace this with optimized Triton/CUDA kernels.
        """

        import torch
        import torch.nn as nn

        # Import the reference model to use as a starting point
        from reference import Model


        class ModelNew(nn.Module):
            def __init__(self):
                super(ModelNew, self).__init__()
                self._ref = Model()

            def forward(self, {forward_params}) -> torch.Tensor:
                return self._ref({", ".join(p.split(":")[0].strip() for p in forward_params.split(",") if p.strip())})
    ''')


def main():
    parser = argparse.ArgumentParser(description="Load a KernelBench problem")
    parser.add_argument("--level", type=int, required=True, help="Problem level (1-4)")
    parser.add_argument("--problem", type=int, required=True, help="Problem ID within level")
    parser.add_argument("--source", default="local", choices=["local", "huggingface"],
                        help="Dataset source (default: local)")
    args = parser.parse_args()

    print(f"Loading KernelBench Level {args.level}, Problem {args.problem}...")

    try:
        dataset = construct_kernelbench_dataset(
            level=args.level,
            source=args.source,
        )
    except Exception as e:
        print(f"Error loading dataset: {e}", file=sys.stderr)
        print("Try --source huggingface if local dataset is not installed.", file=sys.stderr)
        sys.exit(1)

    try:
        problem = dataset.get_problem_by_id(args.problem)
    except Exception as e:
        print(f"Error loading problem {args.problem}: {e}", file=sys.stderr)
        sys.exit(1)

    # Write reference.py
    header = f'"""\nKernelBench Level {args.level}, Problem {args.problem}: {problem.name}\nREAD-ONLY — do not modify this file. The agent modifies kernel.py instead.\nLoaded via scripts/setup_problem.py from KernelBench dataset.\n"""\n\n'
    with open("reference.py", "w") as f:
        f.write(header + problem.code)

    print(f"Wrote reference.py: {problem.name}")

    # Generate a naive kernel.py
    forward_params = extract_forward_signature(problem.code)

    # For simple problems (L1), try a passthrough kernel
    kernel_code = make_passthrough_kernel(problem.code, forward_params)
    with open("kernel.py", "w") as f:
        f.write(kernel_code)

    print(f"Wrote kernel.py: passthrough baseline for {problem.name}")
    print(f"\nReady to run: uv run python prepare.py")


if __name__ == "__main__":
    main()
