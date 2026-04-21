# Leaderboard Record Schema

**Version:** 1.0
**Status:** Stable contract for Phase 1 (writer 1.2a, reader 1.2b, batch-runner 1.3, metrics 1.4a, reporting 1.4b).

A leaderboard record is a single JSON object representing the **winning kernel** produced by one optimization run on one problem, on one hardware target, on one date, under one configuration. Records are append-only historical artifacts — they should never be edited in place after a run completes.

---

## Storage layout

- One JSON object per record.
- Records live under `results/leaderboard/` (exact sharding — by date, hardware, etc. — is owned by the writer 1.2a, not this schema).
- `_example.json` in this directory is a canonical schema-valid sample; filenames beginning with `_` are reserved for schema artifacts and MUST NOT be treated as real records by readers.

## Uniqueness key

The tuple **`(problem_id, hardware, date, config_hash)`** uniquely identifies a record. The writer (1.2a) MUST atomic-write against this key — a second run with the same tuple on the same day overwrites the earlier record. A run with a different `config_hash` (e.g. different model, budget, or seed) produces a new row alongside the existing one, not an overwrite. This is how we keep "claude vs o3 on the same problem" both visible on the leaderboard.

---

## Fields

All fields are **required** (non-nullable). If a value is genuinely unknown, use the documented default rather than omitting the field or using `null`. Rigid schema → simpler readers.

### `schema_version`

- **Type:** `string`
- **Nullable:** no
- **Description:** Explicit version string so readers can route old records through migration logic. Convention: semver-like `"major.minor"` — bump `major` on breaking changes (field removal, type change, semantic reinterpretation) and `minor` on additive changes (new fields only). Writers MUST emit this field on every record. See the reader contract below for how missing / unknown versions are handled.
- **Constraints:** `^\d+\.\d+$`.
- **Current value:** `"1.0"`.
- **Example:** `"1.0"`

### `problem_id`

- **Type:** `string`
- **Nullable:** no
- **Description:** Stable unique identifier for the problem. Follows the namespace of its source suite (KernelBench L1/L2 use `kb_l1_NNNN` / `kb_l2_NNNN`; GPU Mode problems use `gpumode_<slug>`).
- **Constraints:** `^[a-z0-9_]+$`, max 64 chars.
- **Examples:** `"kb_l1_0042"`, `"gpumode_histogram"`

### `problem_name`

- **Type:** `string`
- **Nullable:** no
- **Description:** Human-readable problem name, used for reporting and display. Not a key.
- **Constraints:** non-empty, max 128 chars.
- **Example:** `"Softmax"`

### `tier`

- **Type:** `string` (enum)
- **Nullable:** no
- **Description:** Problem suite / difficulty bucket.
- **Allowed values:** `"L1"`, `"L2"`, `"GPU_MODE"`
- **Example:** `"L1"`

### `hardware`

- **Type:** `string` (enum)
- **Nullable:** no
- **Description:** GPU the kernel was benchmarked on. Part of the uniqueness key — the same problem on two hardwares produces two records.
- **Allowed values:** `"L40S"`, `"H100"`, `"A100-80GB"`, `"B200"`
- **Example:** `"L40S"`

### `date`

- **Type:** `string` (ISO-8601 date, `YYYY-MM-DD`)
- **Nullable:** no
- **Description:** Calendar date the run was executed, in UTC. Part of the uniqueness key — re-running the same problem on a later date produces a fresh record (so we can track progress over time).
- **Example:** `"2026-04-22"`

### `timestamp`

- **Type:** `string` (ISO-8601 datetime, RFC 3339, UTC, `Z`-suffixed)
- **Nullable:** no
- **Description:** Precise start time of the run. Redundant with `date` but allows sub-day ordering for analytics.
- **Example:** `"2026-04-22T14:17:03Z"`

### `kernel_hash`

- **Type:** `string`
- **Nullable:** no
- **Description:** SHA-256 content hash of the winning kernel source, truncated to the first 16 lowercase hex chars. Used for de-duplication across problems and to detect when a "new record" is actually the same kernel as before.
- **Constraints:** `^[a-f0-9]{16}$`
- **Example:** `"a3f2b91c4d7e6081"`

### `kernel_source_path`

- **Type:** `string` (POSIX path, relative to the leaderboard root `results/leaderboard/`)
- **Nullable:** no
- **Description:** Location of the full kernel source for this record. Required for audit-ability: `kernel_hash` alone cannot be verified without the source, and cheat detection (Sakana lesson — sandbox exploits, cached outputs, absurd speedups) requires a human to read the actual kernel. Convention: `kernels/{kernel_hash}.py` so the filename contains the hash. Writers MUST persist the source alongside the record; readers MAY assume the file exists when `correct == true`.
- **Constraints:** relative path, no `..`, ends in `.py`, `.cu`, or `.triton`.
- **Example:** `"kernels/a3f2b91c4d7e6081.py"`

### `model`

- **Type:** `string`
- **Nullable:** no
- **Description:** The LLM (or routing tag) that produced the winning kernel. For a single-model run, this is the provider's canonical model id. For Phase 3's per-op-type routing, this MAY hold a routing tag like `"sonnet-for-gemm"` or `"router:v2"` — the schema does not change, only the convention. Downstream readers should treat unknown prefixes as opaque labels.
- **Examples:** `"o3-mini"`, `"claude-sonnet-4-6"`, `"sonnet-for-gemm"`

### `speedup`

- **Type:** `number` (float)
- **Nullable:** no
- **Description:** `reference_runtime / kernel_runtime`, measured on `hardware`. A value `>= 1.0` means the kernel is at least as fast as the reference; `< 1.0` means slower. Unbounded above.
- **Constraints:** `> 0`.
- **Example:** `1.85`

### `sol_score`

- **Type:** `number` (float)
- **Nullable:** no
- **Description:** Speed-of-Light score as computed by `sol_metrics.py`. A normalized efficiency metric in `[0.0, 1.0]` where `1.0` = saturating the hardware's theoretical peak for the dominant resource.
- **Constraints:** `0.0 <= sol_score <= 1.0`.
- **Example:** `0.42`

### `compute_util`

- **Type:** `number` (float)
- **Nullable:** no
- **Description:** Measured compute utilization as a percentage of the GPU's peak TFLOPS for the relevant dtype.
- **Constraints:** `0.0 <= compute_util <= 100.0`.
- **Example:** `18.4`

### `bandwidth_util`

- **Type:** `number` (float)
- **Nullable:** no
- **Description:** Measured memory bandwidth utilization as a percentage of the GPU's peak HBM bandwidth.
- **Constraints:** `0.0 <= bandwidth_util <= 100.0`.
- **Example:** `45.2`

### `bottleneck_type`

- **Type:** `string` (enum)
- **Nullable:** no
- **Description:** Roofline classification of the winning kernel. `"unknown"` is the default when profiling was inconclusive — do not emit `null`.
- **Allowed values:** `"compute-bound"`, `"memory-bound"`, `"balanced"`, `"unknown"`
- **Example:** `"memory-bound"`

### `correct`

- **Type:** `boolean`
- **Nullable:** no
- **Description:** Whether the kernel passed its correctness check against the reference. Incorrect kernels MAY still be recorded (useful for progress reports) but consumers that rank leaderboards MUST filter to `correct == true`.
- **Example:** `true`

### `cost_usd`

- **Type:** `number` (float)
- **Nullable:** no
- **Description:** Total LLM cost for the optimization run in USD (input + output tokens across all rounds/iterations). `0.0` is valid for free/local models — do not emit `null`.
- **Constraints:** `>= 0.0`.
- **Example:** `0.12`

### `elapsed_s`

- **Type:** `integer`
- **Nullable:** no
- **Description:** Wall-clock seconds from run start to winning kernel being saved. Includes LLM latency, compile time, and benchmark time.
- **Constraints:** `>= 0`.
- **Example:** `142`

### `stop_reason`

- **Type:** `string`
- **Nullable:** no
- **Description:** Free-form human-readable reason the run ended. Writer convention: start with a short category tag (`"Target reached:"`, `"Budget exhausted:"`, `"Error:"`, `"Manual stop:"`) followed by details.
- **Constraints:** non-empty, max 256 chars.
- **Example:** `"Target reached: 0.42 >= 0.30"`

### `config_hash`

- **Type:** `string`
- **Nullable:** no
- **Description:** Short deterministic hash of the run configuration tuple `(model, backend, budget, target, seed)`. Distinguishes runs on the same problem under different configs so they appear as separate leaderboard rows. Part of the uniqueness key.
- **Constraints:** `^cfg_[a-f0-9]{6,16}$` — the `cfg_` prefix is required so logs visibly distinguish this from `kernel_hash`.
- **Example:** `"cfg_f7d12e"`

### `rounds`

- **Type:** `integer`
- **Nullable:** no
- **Description:** Number of autopilot rounds executed. A "round" is one cycle of plan → generate → evaluate → reflect in the optimizer.
- **Constraints:** `>= 1` when `correct == true`; `>= 0` otherwise (a run can fail before completing a round).
- **Example:** `3`

### `iterations`

- **Type:** `integer`
- **Nullable:** no
- **Description:** Total kernel attempts across all rounds. Always `>= rounds` (each round produces at least one attempt, usually several).
- **Constraints:** `>= 0`.
- **Example:** `12`

---

## Reader contract for `schema_version`

Readers MUST tolerate cross-version records using the following rules:

- **Missing `schema_version`** → treat the record as `"0.0"` (pre-migration). Readers MAY skip or route through a legacy-migration path; they MUST NOT hard-fail.
- **`major` matches the reader's current major** → process normally. Unknown `minor` bumps are guaranteed additive-only, so unknown fields MUST be ignored rather than rejected.
- **`major` is higher than the reader knows about** → log a warning and skip the record. Do not attempt partial parsing; breaking changes are opaque to older readers by definition.
- **`major` is lower than the reader's current major** → the reader SHOULD run its migration path for that version, or skip with a warning if no migration is defined.

Writers MUST NOT emit records with a `major` version they did not themselves write — i.e. never fabricate a future version to "reserve" it.

---

## Design notes

### Why `config_hash` is separate from `kernel_hash`

`kernel_hash` identifies the *artifact* (the source of the winning kernel). `config_hash` identifies the *run configuration* that produced it (model, backend, token/time budget, SOL target, RNG seed). Two different models producing the same kernel source would share a `kernel_hash` but have different `config_hash`es — we want both rows on the leaderboard so we can answer "which model is best on this problem". Conversely, the same config reproducing the same kernel on a later date is still a fresh row (different `date`) so we can track drift / reproducibility.

### Uniqueness key, revisited

The tuple `(problem_id, hardware, date, config_hash)` is the full primary key. In plain terms:

- Same problem, same hardware, same day, same config → **overwrite** (latest run wins).
- Same problem, same hardware, same day, different config (e.g. different model) → **two rows**.
- Same problem, same hardware, different day → **two rows** (historical tracking).
- Same problem, different hardware → **two rows**.

Writer 1.2a is responsible for enforcing this atomically.

### Correctness filtering

`correct: false` records are intentionally kept on disk so Phase 1.4b reports can show attempt counts and failure modes. Any ranking / "best per problem" selection MUST filter `correct == true` first. Readers (1.2b) should expose a convenience flag for this.

### Future-proofing the `model` field

Phase 3 will introduce per-op-type model routing. Rather than add a `router` / `routing_policy` field now (and force every record to carry an almost-always-null value), the convention is to put a routing tag directly in `model` (e.g. `"sonnet-for-gemm"`, `"router:v2:gemm-sonnet-reduce-o3"`). The schema is unchanged; tooling just learns to parse the tag. Revisit in Phase 3 if this overloading becomes painful.

### No `null` values

Every field is required and non-nullable. When a value is unknown, use the documented default (`"unknown"` for `bottleneck_type`, `0.0` for `cost_usd` on free models, etc.). This keeps readers trivial — no `if x is None` branches — and makes bad data loud rather than silent.

---

## Changelog

Future `1.x` bumps are **additive-only** (new optional-by-omission-only fields with documented defaults). Any field removal, type change, or semantic reinterpretation requires a `2.x` bump.

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-04-21 | Initial schema. 21 required fields. Uniqueness key `(problem_id, hardware, date, config_hash)`. Added `schema_version` and `kernel_source_path` post-review. |
