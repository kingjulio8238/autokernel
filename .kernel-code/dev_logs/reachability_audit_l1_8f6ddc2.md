# L1 Reachability Audit (KernelBench Level 1, 100 problems)

| Field | Value |
|---|---|
| Commit SHA | `8f6ddc2` |
| Hardware | L40S (864 GB/s HBM, 733 TFLOPs FP16 TC, 48 GB) |
| Target speedup | 1.5× |
| Date | 2026-04-23 |
| Oracle | `kernel_code/shell.py:check_reachability` (HBM + compute + memory-capacity) |
| Mode | Pure local analysis. No Modal. No measured baselines — heuristic. |

## TL;DR

**At our default baseline-efficiency assumption (PyTorch eager ≈ 30% of peak HBM
bandwidth), 100 / 100 L1 problems are physically reachable at target = 1.5× on
L40S. Zero unreachable. Zero tight.**

That headline is sensitive to the baseline assumption, though. If PyTorch is
actually running at ≥ 67% of peak HBM (i.e., already near speed-of-light), 48
of the 100 problems become unreachable at target = 1.5× — and 16 GEMM problems
remain reachable thanks to a higher compute ceiling.

So: **the engine's previous 40% target-rate on L1 is *nowhere near* the
physical ceiling under reasonable PyTorch baseline assumptions.** Even in the
worst-case (eff = 0.67), the ceiling is ~52 reachable, vs. 40 hit — i.e., the
engine is at most at ~77% of its physical ceiling and has at least ~12
percentage points of room to grow before any reachability constraint binds.
At realistic PyTorch baseline efficiency (~30–50%), the ceiling is 100, so the
engine is at ~40% of ceiling and has 60 pp of room.

## Headline numbers

| Bucket | Count | Definition |
|---|---:|---|
| Reachable | **100** | Oracle silent at default heuristic baseline |
| Tight | 0 | Target within 5–15% of HBM/compute ceiling |
| Unreachable | 0 | Oracle warning fires (target ≥ 95% of ceiling) |

## Per-bucket × bottleneck

| Bucket | memory | compute | launch | unclassified |
|---|---:|---:|---:|---:|
| Reachable | 20 | 52 | 11 | 17 |
| Tight | 0 | 0 | 0 | 0 |
| Unreachable | 0 | 0 | 0 | 0 |

`bottleneck` is the classifier's verdict (`is_memory_bound_likely` /
`is_compute_bound_likely` / `is_launch_bound_likely`, in that order).
"unclassified" = the classifier returned `op_type=custom` because no pattern
fired strongly.

## Per-bucket × op_type

| op_type | Reachable |
|---|---:|
| conv        | 35 |
| custom      | 17 |
| gemm        | 16 |
| reduction   | 16 |
| pooling     |  6 |
| elementwise |  5 |
| norm        |  4 |
| attention   |  1 |
| **Total**   | **100** |

## Sample problems per bucket

### Reachable (showing 6 of 100)

| # | name | op_type | bottleneck | elements | HBM ceiling × |
|---:|---|---|---|---:|---:|
|   1 | 1_Square_matrix_multiplication_      | gemm        | compute      |    16,777,216 | 2.41 |
|   2 | 2_Standard_matrix_multiplication_    | gemm        | compute      |    33,554,432 | 3.41 |
|   3 | 3_Batched_matrix_multiplication_     | gemm        | compute      |   268,435,456 | 9.66 |
|   4 | 4_Matrix_vector_multiplication_      | gemm        | compute      | 2,147,483,648 | 27.31 |
|  19 | 19_ReLU                              | elementwise | memory       | 1,610,612,736 | 3.33 |
|  56 | 56_conv_standard_2D__asymmetric_…    | conv        | compute      |     <…> | (no fire — fp16 GEMM ceiling not computed) |

### Tight (0 of 100)
*(empty)*

### Unreachable (0 of 100)
*(empty)*

## The big question

**Of the 100 L1 problems at target = 1.5× on L40S, the theoretical reachable
ceiling under our default baseline assumption (PyTorch ≈ 30% of peak HBM) is
100% (= 100/100).**

The current engine hits **40% target-rate on L1** (from the prior 10-problem
sweep, extrapolated). Therefore the engine is at **≤ 40 / 100 = 40% of its
physical ceiling** and has **≥ 60 percentage points of room to grow** before
any reachability constraint binds at target = 1.5×.

Even in the worst plausible case (PyTorch eager already at 67% of peak HBM,
which would be unusual for unfused eager kernels), 52 of 100 problems remain
reachable — the engine still has at least 12 pp of room.

**Implication for kernel+ Step 2 (full L1 sweep) and Step 3 (no-stop engine):**
The 40% target-rate is bandwidth-limited by *engine quality*, not by
*physics*. We can pursue full-budget runs aggressively without expecting to
hit a ceiling cliff — the cliff is far above us.

## Sensitivity to PyTorch baseline efficiency

The dominant uncertainty is "how fast is the PyTorch eager baseline already?"
We have no measured baselines, so the audit assumes a fixed PyTorch HBM
efficiency `e_hbm` and treats the implied baseline = floor / `e_hbm`. The HBM
ceiling speedup at that assumption is `1 / e_hbm`.

| `e_hbm` | implied HBM ceiling × | reachable | tight | unreachable |
|---:|---:|---:|---:|---:|
| 0.20 | 5.00× |  100 | 0 |  0 |
| 0.30 | 3.33× |  100 | 0 |  0 |
| 0.40 | 2.50× |  100 | 0 |  0 |
| 0.50 | 2.00× |  100 | 0 |  0 |
| 0.60 | 1.67× |   52 | 48 |  0 |
| 0.67 | 1.49× |   52 |  0 | 48 |
| 0.75 | 1.33× |   52 |  0 | 48 |
| 0.80 | 1.25× |   52 |  0 | 48 |
| 0.90 | 1.11× |   52 |  0 | 48 |

The 52 that survive at every efficiency are GEMM (16) + conv (35) + the 1
attention problem — these are compute-bound and use a separate compute
ceiling (peak FP16 TC at 50% efficiency → ceiling ≈ 2.0×).

The 48 that flip from reachable → tight → unreachable as `e_hbm` rises are
the memory-bound + launch-bound + memory-flavored "unclassified" ops:

| op_type | count flipped at high eff |
|---|---:|
| custom (mostly tiny activation kernels) | 17 |
| reduction      | 16 |
| pooling        |  6 |
| elementwise    |  5 |
| norm           |  4 |

Sample of the 48 sensitive problems: `19_ReLU`, `20_LeakyReLU`, `21_Sigmoid`,
`22_Tanh`, `23_Softmax`, `33_BatchNorm`, `34_InstanceNorm`, `40_LayerNorm`,
`41_Max_Pooling_1D`, `47_Sum_reduction_over_a_dimension`,
`51_Argmax_over_a_dimension`. These are exactly the small-elementwise /
reduction / norm shapes where PyTorch's eager runtime *might* already be
near-optimal.

## Caveats (read these)

1. **No measured baselines.** We use a heuristic: memory-bound problems
   assume PyTorch ≈ 30% of peak HBM; compute-bound matmul assumes cuBLAS
   ≈ 50% of peak FP16 TC. These are typical-but-rough rules of thumb. The
   sensitivity sweep above shows how much the verdict moves.

2. **Classifier shape extraction was incomplete on the unmodified code.**
   `kernel_code/problem_classifier.py:_estimate_tensor_elements` only matches
   `UPPERCASE = <int>` module-level assignments, but KernelBench uses
   `N = 2048 * 2`, lowercase `batch_size = 4096`, tuple `input_shape = (8192,)`,
   and starred-expansion `torch.rand(batch_size, *input_shape)`.
   Running with the unmodified classifier returned **`elements = 0` for 27/100
   problems**, which silenced both the HBM and compute ceiling checks
   (they require `elements >= 1024`). The audit augmented shape extraction
   *in the audit script only* (no changes to the oracle) using a small
   AST-based const-folder; this recovered all 27 missing shapes.
   **Action item**: porting the audit's `extract_largest_tensor` into
   `_estimate_tensor_elements` would make the live oracle materially more
   accurate. Out of scope for this audit, but flagged.

3. **Compute-bound non-GEMM has no FLOP-based ceiling.** `_compute_ceiling_check`
   only fires for `op_type == "gemm"`. The 35 conv problems and the 1 attention
   problem fall back to memory-floor logic, which is a loose upper bound on
   their reachability. A conv-specific ceiling (im2col GEMM-equivalent FLOPs)
   would tighten this.

4. **Memory-capacity check (`_memory_capacity_check`)** uses a tensor-count
   heuristic (3 for norms, 4 for attention, 3 for gemm, 2 for elementwise).
   None of the 100 L1 problems came close to L40S's 48 GB capacity at the
   shapes recovered, so this check stayed silent throughout — but at the
   2 GB-element scale on some L1 reductions/norms, fp32 working sets are
   ~17 GB and a fatter tensor-count assumption could trip it.

5. **L1-only.** This audit does not say anything about L2 (fused multi-op)
   or L3 (fully synthetic). Reachability for those is presumably *harder*
   because more tensors round-trip, but the oracle's per-op heuristics
   weren't designed for them.

6. **`is_launch_bound_likely` is a tiny-shape heuristic** that predates the
   augmented shape extraction. Once the real KernelBench shapes are
   recovered, several "launch-bound" classifications (e.g., 33_BatchNorm at
   1B elements) are clearly not launch-bound in reality. The audit treats
   `bottleneck=launch` as memory-bound for baseline estimation purposes;
   this matches what the oracle's HBM check actually does.

## Files

- Audit script (transient, not committed): `/tmp/reachability_audit_v2.py`
- Raw audit JSON (transient): `/tmp/reachability_audit_v2_out.json`
- Summarizer (transient): `/tmp/reachability_summary_v2.py`

If we want to re-run on a different commit / target / hardware, the script
takes ~3 seconds wallclock and zero $.
