# Business Entity Resolution — Amazon ML Challenge 2026

Matching noisy business records across three independent sources. For each
Source-1 record, find every Source-2 / Source-3 record describing the same
real-world business. Scored by **macro F0.5, averaged per S1 entity**.

> **Status: in progress.** Steps 1–2 are implemented and tested; steps 3–10 are
> not written yet. Nothing here claims a leaderboard score.

---

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The dataset is **not** in this repo — each file is 121–486 MB (over GitHub's
100 MB limit), and it is the organizers' data to distribute. Download
`student_resource/` from the challenge portal and point the pipeline at it:

```
<data-dir>/
├── train/  train_source1.tsv  train_source2.tsv  train_source3.tsv  train_ground_truth.tsv
└── test/   test_source1.tsv   test_source2.tsv   test_source3.tsv
```

## Run the tests

```bash
cd code/business_entity_resolution
python3 -m pytest -q          # 44 tests, ~1s, no dataset required
```

The tests use tiny invented fixtures. Those are software fixtures only — never
training data.

---

## What is implemented

| Module | Purpose |
|---|---|
| `src/business_er/metrics.py` | The exact competition metric, plus a blocking-ceiling diagnostic |
| `src/business_er/io.py` | Strict TSV reading and submission writing |
| `src/business_er/ids.py` | Compact int32 id representation (memory) |

### `metrics.py`

`entity_f05` scores one S1 record; `macro_entity_f05` averages over all of them,
and that average is the leaderboard score. Verified against the worked example in
the problem statement (predict 3, 2 correct, truth 2 → **0.714**).

Two deliberate design choices:

- **It refuses incomplete predictions.** If any S1 record is missing it raises,
  rather than averaging over a subset — which would silently flatter a broken run.
- **`blocking_report` reports an oracle ceiling**: the score a perfect model
  would achieve from a given candidate set. A true match that blocking never
  proposes is unrecoverable, so this number caps the final score and tells us
  whether to invest in blocking or in the model.

### `io.py`

Reading is strict on purpose: tab separator only, everything stays a string,
`keep_default_na=False` so a business genuinely named `"NA Foods"` is not turned
into a missing value, and a malformed line raises instead of being skipped.

Writing enforces the submission contract itself — one row per S1 record in input
order, a *real* empty field for singletons (never `[]`, `None` or `nan`), ids
sorted and de-duplicated so two runs are byte-identical, and `QUOTE_NONE` so a
delimiter hiding inside an id crashes here rather than being rejected upstream.

Verified with the organizers' own `utils/validate_submission.py`: `PASS`,
exit 0, no warnings, including with `--check-ids`.

### `ids.py`

Every id is `S{1,2,3}-` plus ≤9 digits with no leading zeros — verified across
all six source files, zero exceptions — so an id fits in an `int32`. That is
~4 bytes instead of ~70 for a Python string: roughly 40 MB rather than 800 MB
for the 10.3M test target records. With 15 GB of RAM and 1.7 × 10¹³ possible
pairs, this is what makes the later stages fit in memory at all.

---

## Dataset facts established so far

Measured from the training data. The first two were verified exactly over all
7,638,365 true pairs, not sampled.

| Fact | Value | Why it matters |
|---|---|---|
| Country agreement within true pairs | **100%**, 0 exceptions | Blocking by country is safe and cuts the search space ~3× |
| S2/S3 records claimed by 2+ S1 records | **0** | Targets are exclusive, so competing claims are a usable precision signal |
| Singleton rate | 5.6% | An all-empty submission scores only ~0.056 |
| Matches per non-singleton | 3.67 mean, 11 max | Genuinely multi-match; never top-1 |
| Test country mix | US 38%, India 47%, **France 15%** | France never appears in training |

Noise actually present: nine Indic scripts across Indian names (~23% of
S2-India), which are phonetic transliterations of the Latin name; injected typos
and accents (`Autrey`→`Atsryi`, `Empire`→`Émpire`); drifting street numbers
(`2046`→`2048`, `3077`→`3077-D`); replaced city names (`New Castle`→`Ossining`);
reordered tokens; and French legal/address vocabulary (`SARL`, `R.`, `5 bis`)
entirely absent from training.

Channel coverage on sampled true pairs — no single signal suffices:

| Signal alone | True pairs it would lose |
|---|---|
| Name similarity ≥ 60 | 13% (but 83% of those share an address number) |
| Shares an address number | 19% |
| Shares any address token | 4% |

---

## Roadmap

| Step | Status |
|---|---|
| 1. Metric + strict I/O | ✅ implemented, tested, official validator passes |
| 2. Data audit | ✅ findings above |
| 3. Entity-group validation split | ⬜ |
| 4. Text normalization | ⬜ |
| 5. Candidate generation (blocking) | ⬜ |
| 6. Pair features | ⬜ |
| 7. LightGBM matcher | ⬜ |
| 8. Threshold tuning on the real metric | ⬜ |
| 9. Test inference + both output files | ⬜ |
| 10. Packaging | ⬜ |

## Compliance

No external business databases, lookup APIs, geocoders, or scraped data. Every
signal comes from the supplied files and local string processing. LightGBM is
MIT licensed; no pretrained model weights are used.
