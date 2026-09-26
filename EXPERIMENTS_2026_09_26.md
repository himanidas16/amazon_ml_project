# Improvement experiments — 26 September 2026

Existing submission and v1 model are preserved. These experiments use the supplied
records only. No leaderboard submission has been made by this work.

## Correctness fix

`experiments/build_pairs_v2.py` mixed global anchor rows with country-local
candidate indices when constructing labels. It now uses the tested
`business_er.labels.label_candidates` helper. Rebuild any pair tables produced
with the previous label construction before training on them.

## Baseline

Saved v1 predictions, threshold 0.80, one-owner decision, all 20,000 validation
businesses and complete truth: macro F0.5 **0.95334894**. India **0.94071987**;
US **0.96175079**. Candidate oracle **0.97895657**.

## Completed experiments

- **Additional tie-aware context model:** 100k training businesses, all positive
  and hard-negative pairs plus 10% random remaining negatives with inverse
  sampling weights; 1,086,840 retained pairs and 68 features. Best dev iteration
  1,039. Standalone validation roughly 0.95021. Blending did not beat the original
  baseline on the evaluation half. Not promoted. This experiment also changes
  sampling and model capacity, so it does not isolate the effect of tie ranks.
- **Conservative exact-text duplicate expansion:** 999,943 to 1,003,991 validation
  candidates; just one additional true edge. Macro gain about 0.000002 at the
  existing threshold. Not promoted; too little benefit.

## Running / planned evaluation

A rank-independent model uses only the first 37 existing pair features. Its main
purpose is to score newly retrieved pairs outside the original top-25/source
selection, not necessarily to replace v1 on its own candidates.

The wider retrieval experiment uses a fixed random sample of 4,000 India
validation businesses, queries the **entire country target pool**, and retains
150 candidates per source. Ground truth is used only after retrieval. Existing
v1 predictions stay the baseline. A rescue threshold is selected on half the
businesses and evaluated on the other half, with a paired bootstrap interval.
These businesses come from the existing validation sample, not a new pristine
holdout; repeat positive findings on a fresh sample before promotion.

## Reproduction (from workspace root)

```bash
code/business_entity_resolution/experiments/run_capped.sh --max-gb 3.5 \
  code/business_entity_resolution/experiments/contextual_train.py
code/business_entity_resolution/experiments/run_capped.sh --max-gb 2 \
  code/business_entity_resolution/experiments/duplicate_probe.py
code/business_entity_resolution/experiments/run_capped.sh --max-gb 3 \
  code/business_entity_resolution/experiments/contextual_train.py \
  --pair-only --rounds 1400 --out artifacts/pair_only_v1
code/business_entity_resolution/experiments/run_capped.sh --max-gb 3.75 \
  code/business_entity_resolution/experiments/wide_probe.py \
  --country India --anchors 4000 --k 150 --workers 2
```

Run heavy experiments sequentially when memory is limited. Existing prediction
work runs separately and has not been stopped or overwritten. The launcher's
memory cap is per process tree, not a global cap shared across experiments.

Reports and reusable artifacts:

- `artifacts/contextual_v1/report.json`
- `artifacts/duplicate_probe/report.json`
- `artifacts/pair_only_v1/report.json`
- `artifacts/wide_India_4000_k150/`
- `artifacts/logs/{contextual_train,duplicate_probe,pair_only_train,wide_india_probe}.log`

Run tests from `code/business_entity_resolution/` with `python3 -m pytest -q`.
