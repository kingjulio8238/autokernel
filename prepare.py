"""
Fixed evaluation harness for autokernel.
Wraps KernelBench's eval_kernel_against_ref().
Agent CANNOT modify this file.
"""

import sys
import time

from kernelbench.eval import eval_kernel_against_ref, get_torch_dtype_from_string

# Constants — agent cannot change these.
# KernelBench defaults are num_correct_trials=1, num_perf_trials=10.
# We override to match the paper's evaluation protocol.
CORRECTNESS_TRIALS = 5
PERF_TRIALS = 100
PRECISION = "fp32"
TIMING_METHOD = "cuda_event"


def load_source(path):
    with open(path) as f:
        return f.read()


def evaluate():
    reference_src = load_source("reference.py")
    kernel_src = load_source("kernel.py")

    start = time.time()
    result = eval_kernel_against_ref(
        original_model_src=reference_src,
        custom_model_src=kernel_src,
        measure_performance=True,
        num_correct_trials=CORRECTNESS_TRIALS,
        num_perf_trials=PERF_TRIALS,
        precision=get_torch_dtype_from_string(PRECISION),
        timing_method=TIMING_METHOD,
        verbose=True,
        # check_for_excessive_speedup=True is the default — populates ref_runtime
    )
    elapsed = time.time() - start

    # eval_kernel_against_ref can return None on lock file errors
    if result is None:
        print("status:error")
        print("speedup:0.00")
        sys.exit(1)

    if not result.compiled:
        print("status:compile_error")
        print("speedup:0.00")
        error = result.metadata.get("compilation_error", "unknown")
        print(f"error:{error}")
        sys.exit(1)

    if not result.correctness:
        print("status:incorrect")
        print("speedup:0.00")
        issue = result.metadata.get("correctness_issue", "unknown")
        print(f"error:{issue}")
        if "max_difference" in result.metadata:
            print(f"max_diff:{result.metadata['max_difference']}")
        sys.exit(1)

    # Correct — report performance
    speedup = result.ref_runtime / result.runtime
    print(f"status:correct")
    print(f"speedup:{speedup:.4f}")
    print(f"runtime_us:{result.runtime:.2f}")
    print(f"ref_runtime_us:{result.ref_runtime:.2f}")
    print(f"eval_seconds:{elapsed:.1f}")

    if speedup > 10:
        print(f"WARNING:excessive_speedup_{speedup:.1f}x")


if __name__ == "__main__":
    evaluate()
