#!/usr/bin/env bash
# Round 3 queue: after round 2 finishes, build the complete validation world
# and run the stage-2 (competing-claims) probe with the better v1.2 model.
set -u
cd "$(dirname "$0")/../../.."
RUN=code/business_entity_resolution/experiments/run_capped.sh
DATA=amazon_ml_dataset/student_resource/dataset
LOG=artifacts/logs
while pgrep -f queue_round2.sh >/dev/null; do sleep 30; done
while systemctl --user list-units 'run-*.service' --no-legend | grep -q running; do sleep 30; done
echo "[queue3] $(date +%T) start"
$RUN -m business_er build-pairs --data-dir $DATA --artifacts artifacts --out-name world_v12 \
     --train-anchors 0 --val-world --workers 4 > $LOG/build_world_v12.log 2>&1
echo "[queue3] $(date +%T) world exit=$?"
BEST=$(python3 -c "
import json
r={t: json.load(open(f'artifacts/report_{t}.json'))['macro_f05'] for t in ('v12_100k','v12_300k')}
print(max(r, key=r.get))" 2>/dev/null || echo v12_300k)
echo "[queue3] best stage-1 model: $BEST"
$RUN code/business_entity_resolution/experiments/stage2_probe.py --world artifacts/world_v12 \
     --model artifacts/models/matcher_$BEST.lgb > $LOG/stage2_probe.log 2>&1
echo "[queue3] $(date +%T) stage2 exit=$?"
