# Business Entity Resolution — Amazon ML Challenge 2026

Finding every record that describes the **same real-world business** across
three messy data sources: **1.7 million businesses** matched against **~10
million records**, with typos, abbreviations, missing fields and
mixed scripts (English, Hindi, Kannada, Punjabi and others).

| | |
|---|---|
| **Best public leaderboard score** | **0.948** macro F0.5 <!-- update with your final score / rank --> |
| **Held-out validation score** | **0.965** macro F0.5 (20,000 unseen businesses vs a 10.3M-record pool) |
| **Countries** | India, US, France (France appears **only** in the test set) |
| **Model** | LightGBM (MIT licence) on 70 engineered features, plus a second-stage model |
| **Hardware** | one 16 GB laptop, no GPU |

---

## Contents

1. [The problem](#1-the-problem)
2. [How the solution works](#2-how-the-solution-works)
3. [Results](#3-results)
4. [Repository layout](#4-repository-layout)
5. [Setup](#5-setup)
6. [Reproducing the submission](#6-reproducing-the-submission)
7. [Engineering for a 16 GB laptop](#7-engineering-for-a-16-gb-laptop)
8. [What I learned](#8-what-i-learned)
9. [Rules and data policy](#9-rules-and-data-policy)
10. [Further documentation](#10-further-documentation)

---

## 1. The problem

There are three sources of business records:

| Source | Train | Test | Example |
|---|---|---|---|
| Source 1 (the "anchor") | 2.2M | 1.73M | `Cozy Pizza Inc \| 4585 Austin Road, Unit 08, Brownsville, TX` |
| Source 2 | 5.0M | 4.9M | `COZY PÍZZA INC \| 4585 Austin Rd, # UNIT 08, Brownsville, Texas` |
| Source 3 | 5.3M | 5.1M | `[Inc] C0zy Pizza \| # UNIT 08, Texas, 4585 Austin Rd, Brownsville` |

For every Source 1 business, the task is to output **all** Source 2/3 records
that belong to the same business. There may be none (a *singleton*) or up to
dozens.

The data is designed to be hard:

- **Noise:** typos (`C0zy`, `5unview`), accents (`PÍZZA`), reordered address
  parts, abbreviations (`Rd`/`Road`, `Pvt`/`Private`), brackets and junk
  (`(ID: [21092)]`), empty addresses.
- **Scripts:** the same Indian business written in Latin script and in
  Devanagari, Kannada, Gurmukhi or Gujarati.
- **Decoys:** records that look almost identical but belong to a different
  business, e.g. the same name at `8617 Seagate Dr` instead of `8616`.

**Metric:** F0.5 per Source 1 business, averaged over all businesses.
F0.5 weights **precision twice as much as recall**, so a wrong match hurts
more than a missed one. A singleton scores 1 only if nothing is predicted.

Two files are submitted: `matching_results.tsv` (the final matches) and
`candidate_pairs.tsv` (the candidate set produced by blocking, before any
learned model).

---

## 2. How the solution works

```
 raw records
     │
     ▼
 ① Normalisation ──── lowercase, accents, legal suffixes, abbreviations,
     │                 own Indic→Latin transliteration, number extraction
     ▼
 ② Blocking ───────── 10 key channels, rarity-weighted, top 25 per source
     │                 + fixed-formula re-selection from a wider top 150
     ▼                 → candidate_pairs.tsv
 ③ Pair features ──── 70 features: string similarities, number agreement,
     │                 postcode, field availability, "how common is this key"
     ▼
 ④ LightGBM matcher ─ probability that a pair is the same business
     │
     ▼
 ⑤ Stage 2 ────────── re-judges pairs using the competing claims around them
     │
     ▼
 ⑥ Decision ───────── threshold + "one owner" rule → matching_results.tsv
```

### ① Normalisation — [`normalize.py`](code/business_entity_resolution/src/business_er/normalize.py)

Each record is converted into several "views": a cleaned name, the name
without legal words (`Pvt`, `Ltd`, `LLC`, `SARL`, `Cie`…), a glued name
without spaces, a cleaned address with abbreviations expanded (with a separate
French table), the set of numbers in the address, and the postcode.

Indian scripts (Devanagari, Kannada, Gurmukhi, Gujarati, Tamil, …) are
**transliterated to Latin with a hand-written table**, so
`यूनिवर्सल सॉफ्टवेयर` can match `Universal Software`. No external library or
dictionary is used.

### ② Blocking (candidate generation) — [`retrieve.py`](code/business_entity_resolution/src/business_er/retrieve.py), [`select.py`](code/business_entity_resolution/src/business_er/select.py)

Comparing 1.7M × 10M records is impossible (1.7 × 10¹³ pairs). Blocking
proposes a short list of likely candidates for each business.

- **10 key channels:** exact name, sorted name tokens, single rare name
  token, name token pairs, address tokens, numbers, number sets, name+number,
  glued name, exact address. Keys are hashed to 64-bit integers and grouped
  per country (true matches always share a country).
- **Rarity scoring:** a shared key is worth `log(N / block size)`, so sharing
  a rare word such as `yamuanth` counts far more than sharing `limited`.
  Blocks larger than 3,000 records are skipped.
- **Wide re-selection:** for each business and source, take the top 150
  candidates, re-rank them with a *fixed formula* (the average token-set
  similarity of the name and address fields that are present), and keep
  the **blocking top 25 plus the formula top 10**. The formula is not learned,
  so this still counts as candidate generation.

| Candidate set | Recall of true matches | Best possible score |
|---|---|---|
| Blocking top 25 per source | 0.943 | 0.979 |
| **+ formula top 10 from top 150 (used)** | **0.965** | **0.988** |

### ③ Pair features — [`features.py`](code/business_entity_resolution/src/business_er/features.py)

**70 features** per candidate pair, including:

- name and address similarity (token set, partial ratio, Jaro-Winkler, glued names)
- number evidence: do the house/unit numbers agree, overlap, or differ?
  Truncated numbers (`B-57` vs `B-5-7`) are handled separately
- postcode agreement at the original width
- **missing-aware features:** an empty address is recorded as *missing*, not
  as "similarity 0", and ranks are computed only among candidates that have the field
- **context features:** how this candidate ranks among all candidates of the
  same business, and how many records share its name/address key in the pool

### ④ LightGBM matcher — [`train.py`](code/business_entity_resolution/src/business_er/train.py), [`pipeline.py`](code/business_entity_resolution/src/business_er/pipeline.py)

- Trained on candidate pairs of **300,000 training businesses**, labelled
  from the ground truth.
- Folds are **grouped by business**, so no business appears in both train
  and validation.
- Easy negatives are subsampled (30%) and re-weighted, which cuts memory
  without biasing the model.
- Validation uses the **real metric** on 20,000 held-out businesses, searched
  against the **full** 10.3M-record pool, so the validation distractors are
  realistic.

### ⑤ Stage 2: competing claims — [`stage2.py`](code/business_entity_resolution/src/business_er/stage2.py), [`competition.py`](code/business_entity_resolution/src/business_er/competition.py)

A record belongs to exactly one business, so if two businesses both claim
the same record, at most one is right. The second-stage model sees, for each
pair, the stage-1 probability **plus** how strongly other businesses claim the
same record and how the pair ranks within its own group. It is trained on a
"complete world": every fold-0 business against every fold-0 record.

### ⑥ Decision — [`predict.py`](code/business_entity_resolution/src/business_er/predict.py)

- Keep pairs above the probability threshold (0.75, tuned for F0.5).
- **One-owner rule:** in the training data no record belongs to two
  businesses, so each record is given only to its highest-scoring claimant.

---

## 3. Results

### Version history

| Version | Main change | Validation F0.5 | Best possible (ceiling) | Public LB |
|---|---|---|---|---|
| v1 | first complete pipeline (56 features) | 0.9533 | 0.979 | 0.943 |
| v2 | France-aware word counts and abbreviations (same model) | — | 0.979 | **0.948** |
| v1.2 features | 70 features, missing-aware, number forensics | 0.9558 | 0.979 | — |
| + 3× training data | 100k → 300k businesses | 0.9586 | 0.979 | — |
| v3 | + wide candidate re-selection + stage 2 | **0.9654** | **0.988** | 0.943 |

Per country (validation): US **0.972**, India **0.956**. France has no
labels; the +0.005 leaderboard jump from v1 to v2 came entirely from
France-specific fixes.

### Why v3 scored lower on the leaderboard than on validation

v3 was clearly better on validation (+0.012) but did not improve the
leaderboard. Analysis of the test files explains why:

| | Businesses | Records in sources 2+3 | Records per business |
|---|---|---|---|
| Train | 2,206,821 | 10,320,219 | 4.68 |
| Test | 1,732,544 | 9,969,589 | **5.75** |

The test set has about **23% more records per business**, which works out to
roughly **twice as many decoys** per business. v3's wider search found more
of them, e.g.:

```
Anchor:  Delhi Vinayak Limited   | H.No - G-52/15 ... Okhla, New Delhi
Decoy:   Delhi Limited Services  | H.no - G-52/17 ... Okhla, New Delhi
```

The model had seen fewer such decoys in training and accepted too many, and
F0.5 punishes those false matches heavily. The lesson: **when validation and
test differ, check the data distribution before trusting validation gains.**
The follow-up plan was stricter thresholds (0.85/0.90 variants were produced)
and features that compare house/unit numbers more strictly.

### Where the remaining error is (validation)

- **Blocking misses:** about half the gap to 1.0. True matches never reach
  the candidate list, mostly because of truncated numbers and very generic
  names.
- **Decision errors:** decoys with the same name and nearby numbers, and
  name-only records with an empty address.
- **Hopeless pairs:** only 0.06% of true pairs are dissimilar in **both**
  name and address.

---

## 4. Repository layout

```
.
├── README.md                          ← you are here
├── PROJECT_REPORT.md                  full version history, experiments, rejected ideas
├── PROJECT_GUIDE.md                   plain-language walkthrough of every step
├── Documentation_template.md          methodology write-up for the submission
├── AWS_SETUP.md                       how to run the pipeline on an AWS machine
├── amazon_ml_challenge_problem_statement.md
└── code/business_entity_resolution/
    ├── requirements.txt               pinned versions (Python 3.13)
    ├── pyproject.toml
    ├── src/business_er/
    │   ├── __main__.py                command-line interface
    │   ├── normalize.py               cleaning, abbreviations, transliteration
    │   ├── retrieve.py                blocking keys, index, rarity scoring
    │   ├── select.py                  wide candidate re-selection (fixed formula)
    │   ├── features.py                70 pair features
    │   ├── train.py / pipeline.py     training and validation
    │   ├── stage2.py / competition.py competing-claims model
    │   ├── predict.py                 end-to-end test inference (resumable)
    │   ├── metrics.py / evaluate.py   the competition metric
    │   ├── splits.py / labels.py      business-grouped folds, labels
    │   ├── io.py / pair_data.py       streaming readers/writers, disk-backed data
    │   └── package.py                 builds and checks the final ZIP
    ├── tests/                         200+ unit and property tests
    └── experiments/                   probes, run queues, memory-capped launcher,
                                       submission checker
```

The dataset, trained models and outputs are **not** in the repository (see
[§9](#9-rules-and-data-policy)).

---

## 5. Setup

Tested on Ubuntu with Python 3.13.

```bash
git clone https://github.com/himanidas16/amazon_ml_project.git
cd amazon_ml_project

python3 -m venv .venv && source .venv/bin/activate
pip install -r code/business_entity_resolution/requirements.txt

export PYTHONPATH=$PWD/code/business_entity_resolution/src
python3 -m pytest -q code/business_entity_resolution/tests    # no data needed
```

Dependencies: `pandas`, `numpy`, `scipy`, `scikit-learn`, `lightgbm`,
`rapidfuzz`, `pyarrow`, `PyYAML`.

Put the organizers' data so that these paths exist:

```
amazon_ml_dataset/student_resource/dataset/train/train_source{1,2,3}.tsv
amazon_ml_dataset/student_resource/dataset/train/train_ground_truth.tsv
amazon_ml_dataset/student_resource/dataset/test/test_source{1,2,3}.tsv
```

---

## 6. Reproducing the submission

All commands run from the repository root. `run_capped.sh` starts each step
with a memory limit so a heavy step cannot freeze the machine (see §7). On a
machine with plenty of RAM, `python3` can be used instead.

```bash
export PYTHONPATH=code/business_entity_resolution/src
RUN=code/business_entity_resolution/experiments/run_capped.sh
DATA=amazon_ml_dataset/student_resource/dataset
SEL="--k-wide 150 --k-formula 10"

# 1. business-grouped folds and word counts (fold 0 is held out)
$RUN -m business_er split      --data-dir $DATA --artifacts artifacts
$RUN -m business_er train-freq --data-dir $DATA --artifacts artifacts

# 2. labelled candidate pairs + features, then the matcher
$RUN -m business_er build-pairs --data-dir $DATA --artifacts artifacts --out-name pairs_sel \
     --train-anchors 300000 --easy-rate 0.3 $SEL
$RUN -m business_er train --artifacts artifacts --pairs-name pairs_sel --tag sel_300k

# 3. stage 2 (competing claims), trained on the complete fold-0 world
$RUN -m business_er build-pairs --data-dir $DATA --artifacts artifacts --out-name world_sel \
     --train-anchors 0 --val-world $SEL
$RUN -m business_er train-stage2 --world artifacts/world_sel \
     --stage1-model artifacts/models/matcher_sel_300k.lgb --out artifacts/models/stage2_sel

# 4. test inference (word counts on the test files are unsupervised statistics)
$RUN -m business_er token-freq --data-dir $DATA --split test --output artifacts/token_freq_test.npz
$RUN -m business_er predict --data-dir $DATA --split test \
     --model artifacts/models/matcher_sel_300k.lgb --freq artifacts/token_freq_test.npz \
     --output-dir output --work-dir artifacts/test_work --threshold 0.75 $SEL \
     --stage2-model artifacts/models/stage2_sel.lgb --stage2-threshold 0.7

# 5. check the two files
python3 code/business_entity_resolution/experiments/check_submission.py output $DATA/test
```

Notes:

- `predict` is **resumable**: it checkpoints every chunk to `--work-dir`,
  and a re-run with only a new threshold reuses the saved scores (~10 minutes
  instead of hours).
- Leave out `--stage2-model` for a stage-1-only submission.
- Approximate times on a 22-core, 16 GB laptop: build pairs ~2.5 h, train
  ~30 min, stage 2 ~1.5 h, test prediction ~2–3 h.

Build the final submission ZIP:

```bash
$RUN -m business_er package --team-name TEAM --repo-root . --output-dir output \
     --doc Documentation_template.md --dest dist
```

---

## 7. Engineering for a 16 GB laptop

The largest intermediate data (India's test candidates: ~48M pairs × 70
features) does not fit in 16 GB of RAM. These techniques made the pipeline
run anyway:

| Problem | Solution |
|---|---|
| Heavy step freezes the laptop | `run_capped.sh` runs each step as a systemd scope with a memory cap computed from free RAM, so it stops instead of crashing the machine |
| Blocking index too large | one index per country; keys hashed to `uint64`; the cleaned text is kept inside the index, so there is no second pass over the files |
| Wide selection runs out of memory | anchors processed in ranges; each range is **spilled to disk**, the index is freed, then the results are gathered one column at a time |
| Scoring 100M+ pairs | candidates **memory-mapped** from `.npy` files and scored in chunks with checkpoints |
| Training matrix too large | features written to disk and memory-mapped; easy negatives subsampled and re-weighted |
| A crash wastes hours | every step checks a manifest (versions + settings) and resumes from the last finished chunk |

The code is covered by **200+ tests**, including property tests (e.g. "the
spilled and in-memory selections are identical") and a CLI smoke test.

---

## 8. What I learned

- **Blocking recall caps everything.** No model can match a pair that
  blocking never proposed; widening the candidate list gave the single
  biggest validation gain.
- **Missing is not the same as different.** Treating an empty address as
  "similarity 0" biased the model against genuine name-only records.
- **The metric shapes the decision.** F0.5 plus singletons make a false match
  expensive, so the threshold and the one-owner rule matter as much as the model.
- **Validation must look like test.** A +0.012 validation gain did not carry
  over because the test set had twice the decoy density.
- **Memory is a design constraint.** Streaming, spilling, memory-mapping and
  checkpoints turned an impossible job into an overnight run on a laptop.

---

## 9. Rules and data policy

- Only the organizers' data and local code are used: no external databases,
  geocoders, gazetteers, address parsers or hosted LLM APIs. The abbreviation
  and transliteration tables are small and hand-written.
- LightGBM is MIT-licensed. No pretrained weights are used.
- Word-frequency statistics on the test files are unsupervised, which the
  organizers confirmed is allowed.
- **The dataset is not included** and must not be uploaded here; it belongs
  to the organizers. `.gitignore` excludes all data, outputs, models and
  archives.

---

## 10. Further documentation

| File | Contents |
|---|---|
| [`PROJECT_REPORT.md`](PROJECT_REPORT.md) | every version, experiment, measurement and rejected idea |
| [`PROJECT_GUIDE.md`](PROJECT_GUIDE.md) | beginner-friendly explanation of each step |
| [`Documentation_template.md`](Documentation_template.md) | methodology write-up submitted with the solution |
| [`AWS_SETUP.md`](AWS_SETUP.md) | running the pipeline on an AWS EC2 machine |
| [`code/business_entity_resolution/README.md`](code/business_entity_resolution/README.md) | package-level usage notes |

---

*Built by Himani Das for the Amazon ML Challenge 2026.*
