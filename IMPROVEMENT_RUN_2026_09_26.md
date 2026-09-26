# Improvement experiments, 26 September 2026

User-reported public scores: v1 0.943, v2 0.948. No experiment here establishes
a public score of 0.99. Do not substitute a candidate ceiling for model F0.5.

## Starting point

The completed `matcher_sel_300k` report records local F0.5 0.965403 at threshold
0.8, or 0.965814 at the best threshold (0.75) on the reused development set.
Its candidate ceiling is 0.987980. Production queue round 4 was already running
when this work started and has been preserved.

## Completed audit

`experiments/error_audit.py` scores saved validation candidates in small batches,
using the feature order stored with the model. It writes predictions, per-anchor
scores, and error records with their source text. It never injects labels into
candidate generation. Inputs and the matcher have recorded hashes.

On 20,000 development businesses at threshold 0.75:

- 2,401 true edges absent from the candidate set (1,961 businesses).
- 2,759 retrieved true edges rejected (2,463 businesses).
- 618 false predicted edges (577 businesses).
- 1,236 of the rejected true edges have an empty target address.
- Score loss attributable to candidate coverage: 0.012020.
- Remaining decision loss within available candidates: 0.022167.

Artifacts: `artifacts/error_audit_sel/{report.json,errors.parquet,predictions.parquet,anchors.parquet}`.

## Changes

- `char_retrieve.py`: country- and field-scoped character retrieval. It streams
  the entire target pool, retaining only query-relevant postings. Common grams
  are discarded, not truncated to whichever target IDs arrive first. This is
  intended for memory-bounded pilots, not yet for all-test deployment.
- `char_retrieval_probe.py`: fixed random 1,000-business development sample,
  full target pool, labels consulted only after candidate generation; optional
  high-confidence baseline matches as extra query views. These predictions
  supply text aliases, not ground-truth seeds.
- `train_raw_matcher.py`: trained a new 50k-business matcher using raw pair
  features only. Unlike the older pair-only experiment, it excludes ret_score
  as well as ranks and candidate-context features. Saved in
  `artifacts/raw_matcher_50k/matcher.lgb` with its exact feature order.
- `char_rescue_probe.py`: tests novel candidates with the raw matcher. Existing
  baseline matches and claimed targets are preserved. Threshold tuning and
  measurement use different halves of the pilot; this is still development
  evaluation because the parent set has been used previously.
- `exact_decision.py`: exact expected F0.5, tested against exhaustive enumeration
  of Bernoulli label outcomes. Its mathematical assumption is independent,
  calibrated pair probabilities and no unseen matches. It is NOT deployed.
- `normalize.py`: equivalent fast Indic detection, with an ASCII shortcut and
  a compiled pattern matching exactly the existing character table. No change
  to normalized output, features, or normalization version is intended.

## Measured experiments

### Name character retrieval

`artifacts/char_probe_name_1000/report.json`: candidate ceiling 0.988446 ->
0.989884 at top 30/source, recovering 6 additional true edges and increasing
average candidates from 59.825 to 63.255. This is a small-sample coverage gain,
NOT a final matcher gain. More candidates at top 100 found no extra positives.

### Exact expected-F0.5 decisions: rejected

`artifacts/exact_decision_sel/report.json`: threshold-rule baseline 0.965793
on the decision holdout, exact-expectation rule 0.964024. Delta -0.001768,
paired bootstrap 95% interval [-0.003052, -0.000470]. No production change.

### Normalization speed

On 17,334 error-record text fields, the old Indic detector took 0.3415 s and
the equivalent new detector took 0.00194 s. This measures ONLY the detector,
not total pipeline speed. Equality was verified, and all 184 project tests
passed (12 existing multiprocessing/fork warnings).

## Promotion requirements

Candidate probes and rescue must finish before drawing conclusions. Do not
replace submission files merely because coverage improved. For a promising
rescue, lock parameters and confirm on fresh held-out businesses against the
full target pool before integrating it into training and test inference.
Character indexing at test scale will need a persistent disk-backed index;
repeatedly rescanning all targets for small query batches is not a production
solution. The production candidate export must contain every scored pair.

### Address character retrieval

`artifacts/char_probe_address_1000_fast/report.json`: top 30/source recovered
23 additional true edges, raising the same pilot ceiling from 0.988446 to
0.991363. Candidates/business increased from 59.825 to 90.109. Top 100 found
no further positives. The combined name/address union has 27 novel, unclaimed
true edges available for rescue.

### Matching-score rescue: promising pilot

`artifacts/char_rescue_1000/report.json`: the new raw matcher evaluated 61,871
novel, unclaimed name/address candidates. Threshold 0.95 was chosen on the
pilot's tuning half. On the other half, macro F0.5 rose from 0.976483 to
0.978003: delta +0.001520, paired bootstrap 95% interval [0.000478, 0.002796].
This small split happens to have a stronger baseline than the full 20k set;
do NOT present 0.978003 as the overall validation or public score. Preserve
threshold 0.95 for subsequent confirmation rather than retuning on holdout.
