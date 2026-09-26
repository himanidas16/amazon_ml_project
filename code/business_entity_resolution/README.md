# Business Entity Resolution — Amazon ML Challenge 2026

For each Source 1 business record, find every Source 2 / Source 3 record that
describes the same real-world business. Scored by **F0.5 per Source 1 record,
averaged** (singletons included).

Pipeline: **normalize → blocking (keys + rarity ranking) → 56 pair features →
LightGBM matcher → threshold + one-owner rule → the two TSV files.**
Only the supplied records and local code are used; no external data, APIs or
pretrained weights. LightGBM (MIT) is the only model.

A plain-language walkthrough of every step, result and known weakness is in
`PROJECT_GUIDE.md` at the repository root.

---

## 1. Setup

Python 3.13 on Linux (tested). From this folder:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=$PWD/src          # the package lives in src/
python3 -m pytest -q                # ~150 tests, a few seconds, no data needed
```

The data is not in this repository. Place the organizers' `student_resource/`
so that these paths exist (any location works; pass it as `--data-dir`):

```
<DATA>/train/train_source1.tsv  train_source2.tsv  train_source3.tsv  train_ground_truth.tsv
<DATA>/test/test_source1.tsv    test_source2.tsv   test_source3.tsv
```

## 2. Reproduce both output files end to end

Run from this folder. `DATA` is the `dataset/` folder, `ART` any empty working
folder, `OUT` where the two submission files go.

```bash
DATA=/path/to/student_resource/dataset
ART=artifacts
OUT=output

python3 -m business_er split       --data-dir $DATA --artifacts $ART   # folds by business (~2 min)
python3 -m business_er train-freq  --data-dir $DATA --artifacts $ART   # word counts, train minus fold 0 (~5 min)
python3 -m business_er build-pairs --data-dir $DATA --artifacts $ART   # labelled pairs + features (~20 min)
python3 -m business_er train       --artifacts $ART --tag final        # LightGBM + validation score (~10 min)
python3 -m business_er token-freq  --data-dir $DATA --split test --output $ART/token_freq_test.npz   # (~5 min)
python3 -m business_er predict     --data-dir $DATA --split test \
    --model $ART/models/matcher_final.lgb --freq $ART/token_freq_test.npz \
    --output-dir $OUT --work-dir $ART/test_work                         # (~70 min)
```

Then check the files with the organizers' validator (from `student_resource/`):

```bash
python3 utils/validate_submission.py --matching $OUT/matching_results.tsv \
    --candidate $OUT/candidate_pairs.tsv --test-dir dataset/test
```

**Memory.** The pipeline was built on a 15 GB laptop. Peak use is about 6–8 GB
(test prediction, India). On Linux with systemd, run heavy steps through
`experiments/run_capped.sh`, which caps memory at (free memory − 2.5 GB) and
stops the whole job cleanly instead of freezing the machine:

```bash
experiments/run_capped.sh -m business_er predict ...   # same arguments as above
```

**Resumable.** `predict` saves each country's blocking and scores (and scoring
progress every chunk) in `--work-dir`, and refuses to reuse a work dir built
with a different model, word table or code version (`manifest.json`).

**Deterministic.** Fixed seeds for the split, samples and LightGBM
(`deterministic=True`), so reruns on the same machine give the same files.

## 3. Code map

| Module (`src/business_er/`) | Role |
|---|---|
| `io.py` | Strict TSV reading; submission writers that enforce every format rule (incl. streaming writers for 86M candidate ids) |
| `ids.py` | `S2-47` ↔ int32, for memory |
| `metrics.py` | Exact competition metric, blocking ceiling, fast vectorised form |
| `splits.py` | Folds grouped by business, with a leak check |
| `normalize.py` | Text views (lower, plain, accent-folded, no-legal-suffix) + Indic → Latin transliteration; country-scoped abbreviation table (France) |
| `retrieve.py` | Blocking: 9 key channels, word-rarity table, per-country index, IDF-ranked top-25 per source |
| `features.py` | 56 pair features (name/address/number similarity, missingness, retrieval, rival context) |
| `labels.py` | Candidate labels from ground truth, consistent local indexing |
| `train.py` | LightGBM matcher (weights, early stopping on a business-level dev split, saved feature order) |
| `evaluate.py` | Decision rules: threshold, one-owner, (rejected: rescue, expected-F0.5) |
| `pipeline.py` | `split`, `train-freq`, `build-pairs`, `train` |
| `predict.py` | Full test inference, resumable, manifest-guarded |
| `pair_data.py`, `contextual.py` | Loader for sharded pair tables; experimental tie-aware context features (not in the final model) |
| `__main__.py` | The CLI above |

`experiments/` holds the scripts behind every measured decision (blocking
channel study, cap/top-K sweeps, decision-rule comparison, wider-retrieval
probes). They are not needed to reproduce the submission. `tests/` has one
test file per module.

## 4. Key settings (all chosen from measurements on held-out validation)

| Setting | Value | Evidence |
|---|---|---|
| Blocking cap (ignore keys shared by more records) | 3000 | cap 10,000: ceiling +0.0004 for 3× the time |
| Candidates kept | top 25 per source (≤ 50) | recall 0.943, oracle F0.5 0.979 |
| Decision threshold | 0.80 | flat optimum 0.77–0.82, tuned on one half of validation, confirmed on the other |
| One-owner rule | on | a target never belongs to two S1 records in 7.6M training pairs |

## 5. Results (validation: 20,000 fold-0 businesses vs the full 10.3M-record pool)

| | Macro F0.5 |
|---|---|
| Predict nothing | 0.056 |
| **Model v1** (threshold 0.80 + one owner) | **0.953** (US 0.962, India 0.941) |
| Perfect matcher on our candidates (ceiling) | 0.979 |

First public leaderboard score (model v1): **0.943**. The gap to validation is
consistent with France (15% of test, absent from training) scoring lower.

## 6. Dataset facts (verified on the training data)

| Fact | Value | Consequence |
|---|---|---|
| Country agreement within true pairs | 100% of 7,638,365 | Blocking within country is safe |
| S2/S3 records matched to 2+ S1 records | 0 | One-owner rule |
| Singleton rate | 5.6% | Empty submission scores ~0.056 |
| Matches per non-singleton | 3.67 mean, 11 max | Multi-match, never top-1 |
| Test country mix | India 47%, US 38%, **France 15%** | No US/India-specific logic anywhere |

## 7. Compliance

- No external databases, APIs, geocoders, gazetteers or scraped data.
- Unsupervised statistics on the test files (word counts) — confirmed allowed
  by the organizers.
- The French abbreviation table in `normalize.py` is small and hand-written,
  applied only to records labelled France (country treated as an open set).
- Model: LightGBM 4.7.0 (MIT). No pretrained weights.
