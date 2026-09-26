#!/usr/bin/env bash
# Round 4: wide selection + v1.2 features + 300k training businesses
# (+ stage 2 if it measurably helps) -> test run v3.
# Every step must exit 0 AND write its expected file freshly, or the queue stops.
set -u
cd "$(dirname "$0")/../../.."
RUN=code/business_entity_resolution/experiments/run_capped.sh
DATA=amazon_ml_dataset/student_resource/dataset
LOG=artifacts/logs
SEL="--k-wide 150 --k-formula 10"

step() {   # step <name> <expected file> <command...>; skipped if the file already exists
    local name=$1 want=$2; shift 2
    if [ -f "$want" ]; then echo "[q4] $(date +%T) $name: already done ($want), skipping"; return 0; fi
    local t0; t0=$(date +%s)
    echo "[q4] $(date +%T) $name: start"
    "$@" > "$LOG/q4_$name.log" 2>&1; local rc=$?
    if [ $rc -ne 0 ]; then echo "[q4] $(date +%T) $name FAILED (exit $rc) -- stopping"; exit 1; fi
    if [ ! -f "$want" ] || [ "$(stat -c %Y "$want")" -lt "$t0" ]; then
        echo "[q4] $(date +%T) $name produced no fresh $want -- stopping"; exit 1; fi
    echo "[q4] $(date +%T) $name: ok"
}
wait_idle() { while systemctl --user list-units 'run-*.service' --no-legend | grep -q running; do sleep 30; done; }

wait_idle
step build-pairs artifacts/pairs_sel/build_report.json \
  $RUN -m business_er build-pairs --data-dir $DATA --artifacts artifacts --out-name pairs_sel \
       --train-anchors 300000 --easy-rate 0.3 --workers 4 $SEL
step train artifacts/report_sel_300k.json \
  $RUN -m business_er train --artifacts artifacts --pairs-name pairs_sel --tag sel_300k
T1=$(python3 -c "import json; r=json.load(open('artifacts/report_sel_300k.json'))['threshold_sweep']; print(max(r, key=r.get))")
echo "[q4] stage-1 threshold $T1"

# stage 2 only if the measurement on the v1.2 world showed a real held-out gain
wait_idle
GAIN=$(python3 -c "import json; print(json.load(open('artifacts/models/stage2_v12.json'))['gain'])" 2>/dev/null || echo 0)
EXTRA=""
if python3 -c "import sys; sys.exit(0 if float('$GAIN') > 0.001 else 1)"; then
    step world artifacts/world_sel/build_report.json \
      $RUN -m business_er build-pairs --data-dir $DATA --artifacts artifacts --out-name world_sel \
           --train-anchors 0 --val-world --workers 4 $SEL
    step stage2 artifacts/models/stage2_sel.json \
      $RUN -m business_er train-stage2 --world artifacts/world_sel \
           --stage1-model artifacts/models/matcher_sel_300k.lgb --out artifacts/models/stage2_sel
    read G2 T2 < <(python3 -c "import json; s=json.load(open('artifacts/models/stage2_sel.json')); print(s['gain'], s['threshold'])")
    echo "[q4] stage-2 gain on the selection world: $G2 (t2=$T2)"
    if python3 -c "import sys; sys.exit(0 if float('$G2') > 0.001 else 1)"; then
        EXTRA="--stage2-model artifacts/models/stage2_sel.lgb --stage2-threshold $T2"
    fi
else
    echo "[q4] stage 2 skipped (gain on v1.2 world: $GAIN)"
fi
echo "[q4] predict extra: ${EXTRA:-none}"
step predict output_v3/matching_results.tsv \
  $RUN -m business_er predict --data-dir $DATA --split test --model artifacts/models/matcher_sel_300k.lgb \
       --freq artifacts/token_freq_test.npz --output-dir output_v3 --work-dir artifacts/test_work_v3b \
       --workers 4 --threshold $T1 $SEL $EXTRA
$RUN --max-gb 3 code/business_entity_resolution/experiments/check_submission.py output_v3 $DATA/test \
  > $LOG/check_v3.log 2>&1
echo "[q4] $(date +%T) check: $(tail -1 $LOG/check_v3.log)"
