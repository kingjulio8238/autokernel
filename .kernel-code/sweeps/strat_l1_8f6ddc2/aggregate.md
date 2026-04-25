# Stratified L1 Mini-Sweep — `strat_l1_8f6ddc2`

**Sweep dir:** `/Users/juliansaks/Desktop/code/autokernel/.kernel-code/sweeps/strat_l1_8f6ddc2`
**Git SHA:** `8f6ddc2` (master)
**Date:** 2026-04-25
**Hardware:** L40S | **Backend:** triton | **Target:** 1.5x | **Budget cap per problem:** $0.50
**Wallclock total:** 67.4 min  |  **Total cost:** $0.442 (cap was $5.00)

## Headline numbers

- **Target rate (≥1.5x):** 2/10 (20%)
- **Beat-baseline rate (>1.02x):** 4/10 (40%)
- **Median speedup:** 0.73x  |  **Max speedup:** 2.28x
- **Crashed (Python error):** 0/10
- **No-correct-kernel (5/5 attempts wrong):** 4/10

## Per-problem table

| ID  | Problem                                    | Op family             | best   | target | stop bucket          | cost     | elapsed |
|-----|--------------------------------------------|-----------------------|--------|--------|----------------------|----------|---------|
|   1 | 1_Square_matrix_multiplication_            | matmul                |  2.28x |      ✓ | target_reached       | $0.023 |   329s |
|  11 | 11_4D_tensor_matrix_multiplication         | matmul                |  0.73x |      — | sub_baseline         | $0.057 |   500s |
|  21 | 21_Sigmoid                                 | pointwise (activation) |  1.06x |      — | completed_all_rounds | $0.023 |   300s |
|  31 | 31_ELU                                     | pointwise (activation) |  1.05x |      — | completed_all_rounds | $0.023 |   317s |
|  41 | 41_Max_Pooling_1D                          | pooling               |  2.01x |      ✓ | target_reached       | $0.011 |   222s |
|  51 | 51_Argmax_over_a_dimension                 | reduction             |  0.14x |      — | sub_baseline         | $0.079 |   733s |
|  61 | 61_conv_transposed_3D__square_input__squar | conv-transposed       |  0.00x |      — | no_correct_kernel    | $0.057 |   382s |
|  71 | 71_conv_transposed_2D__asymmetric_input__s | conv-transposed       |  0.00x |      — | no_correct_kernel    | $0.057 |   421s |
|  81 | 81_conv_transposed_2D_asymmetric_input_squ | conv-transposed       |  0.00x |      — | no_correct_kernel    | $0.057 |   429s |
|  91 | 91_cumsum_reverse                          | scan                  |  0.00x |      — | no_correct_kernel    | $0.057 |   413s |

## Op-family breakdown

| Family               | n | target hit | beat baseline | best speedup | no_correct | sub_baseline |
|----------------------|---|-----------:|--------------:|-------------:|-----------:|-------------:|
| matmul               | 2 | 1/2 | 1/2 | 2.28x | 0 | 1 |
| pooling              | 1 | 1/1 | 1/1 | 2.01x | 0 | 0 |
| pointwise (activation) | 2 | 0/2 | 2/2 | 1.06x | 0 | 0 |
| reduction            | 1 | 0/1 | 0/1 | 0.14x | 0 | 1 |
| conv-transposed      | 3 | 0/3 | 0/3 | 0.00x | 3 | 0 |
| scan                 | 1 | 0/1 | 0/1 | 0.00x | 1 | 0 |

## Stop-reason taxonomy

| Bucket               | count | description |
|----------------------|------:|-------------|
| target_reached       |     2 | best speedup ≥ target (or SOL gate hit) |
| sub_baseline         |     2 | best speedup < 1.0x after all rounds — refused to keep spending |
| completed_all_rounds |     2 | ran every round; result is final |
| no_correct_kernel    |     4 | round 1 failed all 5 attempts (no correct kernel produced) |

## Crashes

**None.** No Python-level engine errors during this sweep. The four `no_correct_kernel` cases produced incorrect kernels but the engine handled them gracefully and stopped after round 1.

## Notes & observations

- **Wins (target ≥1.5x):** matmul-1 (2.28x, ✓SOL gate after 1 iter), maxpool-1D-41 (2.01x, ✓ on round 1).
- **Pointwise activations (sigmoid-21, ELU-31)** ceiling at ~1.05x — torch is already near-optimal for these on L40S; engine ran all rounds, accumulated 2-4 evidence pieces each, found no headroom.
- **Sub-baseline (matmul-11, argmax-51)** — engine correctly cut its losses after round 2 instead of burning budget. Argmax-51 went very deep (0.14x), worth a dedicated look at the prompt+kernel for that op.
- **Conv-transposed family (61/71/81): 0/3 produced any correct kernel** in round 1 (5 attempts each, all incorrect). Stop-reason hint says "check reference dtypes, atomics, or backend choice" — conv-T's weight-initialization-baked-into-Module behavior likely confuses the test scaffolding. This is the loudest measurement signal of the sweep.
- **Scan family (cumsum_reverse-91)** — also 5/5 incorrect in round 1. Reverse-cumsum + flip is non-trivial to express as a parallel Triton scan; likely mis-handled the dim/flip semantics.
- **Engine cost discipline is excellent.** Total $0.45 across 10 problems (avg $0.045) — none came close to the $0.50/problem cap. The two `target_reached` runs spent only $0.011 and $0.023 each. The deepest sub-baseline (argmax-51, 7 iterations) only cost $0.079.

## Cost & time per problem (sorted by cost)

| ID  | cost     | iterations | rounds |
|-----|---------:|-----------:|-------:|
|  51 | $0.079 |          7 |      2 |
|  11 | $0.057 |          5 |      2 |
|  61 | $0.057 |          5 |      1 |
|  71 | $0.057 |          5 |      1 |
|  81 | $0.057 |          5 |      1 |
|  91 | $0.057 |          5 |      1 |
|   1 | $0.023 |          2 |      2 |
|  21 | $0.023 |          2 |      2 |
|  31 | $0.023 |          2 |      2 |
|  41 | $0.011 |          1 |      1 |
