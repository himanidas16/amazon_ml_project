#!/usr/bin/env bash
# Round 2 queue: waits until no capped job is running, then runs the
# experiments one after another (one heavy job at a time on a 15 GB laptop).
set -u
cd "$(dirname "$0")/../../.."          # repo root
RUN=code/business_entity_resolution/experiments/run_capped.sh
DATA=amazon_ml_dataset/student_resource/dataset
LOG=artifacts/logs
while systemctl --user list-units 'run-*.service' --no-legend | grep -q running; do sleep 30; done
echo "[queue] $(date +%T) start"
$RUN -m business_er build-pairs --data-dir $DATA --artifacts artifacts --out-name pairs_v12 \
     --train-anchors 300000 --easy-rate 0.3 --workers 4 > $LOG/build_pairs_v12.log 2>&1
echo "[queue] $(date +%T) build-pairs exit=$?"
$RUN -m business_er train --artifacts artifacts --pairs-name pairs_v12 --tag v12_100k \
     --max-train-anchors 100000 > $LOG/train_v12_100k.log 2>&1
echo "[queue] $(date +%T) train 100k exit=$?"
$RUN -m business_er train --artifacts artifacts --pairs-name pairs_v12 --tag v12_300k \
     > $LOG/train_v12_300k.log 2>&1
echo "[queue] $(date +%T) train 300k exit=$?"
$RUN code/business_entity_resolution/experiments/selection_probe.py > $LOG/selection_probe.log 2>&1
echo "[queue] $(date +%T) selection probe exit=$?"
