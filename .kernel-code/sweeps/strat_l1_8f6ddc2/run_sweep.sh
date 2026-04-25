#!/bin/bash
# Stratified 10-problem L1 sweep launcher.
# Detached from the harness shell via setsid so it survives shell rotation.
set -u

REPO=/Users/juliansaks/Desktop/code/autokernel
OUT=/Users/juliansaks/Desktop/code/autokernel/.kernel-code/sweeps/strat_l1_8f6ddc2
PY=$REPO/.venv/bin/python

cd "$REPO"
export MODAL_PROFILE=kernel+

LOG=$OUT/sweep.log
echo "=== Stratified L1 sweep (run_sweep.sh) ===" | tee "$LOG"
echo "pid=$$  parent=$PPID" | tee -a "$LOG"
date -u +"start_utc=%Y-%m-%dT%H:%M:%SZ" | tee -a "$LOG"

for ID in 1 11 21 31 41 51 61 71 81 91; do
    echo "" | tee -a "$LOG"
    echo "========== problem $ID ==========" | tee -a "$LOG"
    date -u +"problem_${ID}_start=%Y-%m-%dT%H:%M:%SZ" | tee -a "$LOG"
    "$PY" scripts/kb_l1_sweep.py \
        --count 1 \
        --start-id "$ID" \
        --target 1.5 \
        --budget 0.5 \
        --hardware L40S \
        --resume \
        --out-dir "$OUT" 2>&1 | tee -a "$LOG"
    rc=${PIPESTATUS[0]}
    echo "problem_${ID}_rc=$rc" | tee -a "$LOG"
done

date -u +"end_utc=%Y-%m-%dT%H:%M:%SZ" | tee -a "$LOG"
echo "SWEEP_DONE" | tee -a "$LOG"
