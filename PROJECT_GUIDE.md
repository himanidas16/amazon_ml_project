# Project Guide: Business Entity Resolution (Amazon ML Challenge 2026)

A plain-language explanation of everything built so far. You do not need to
know the project to read this. Start at the top and read down.

Last updated: 25 September 2026 (after the first test submission).

---

## Contents

1. [The problem in one example](#1-the-problem-in-one-example)
2. [How we are scored](#2-how-we-are-scored)
3. [What the data taught us](#3-what-the-data-taught-us)
4. [The pipeline at a glance](#4-the-pipeline-at-a-glance)
5. [Each step, explained](#5-each-step-explained)
6. [Every code file and what it does](#6-every-code-file-and-what-it-does)
7. [How to run things](#7-how-to-run-things)
8. [Results so far (all measured)](#8-results-so-far-all-measured)
9. [Weak spots and risks ("loopholes")](#9-weak-spots-and-risks-loopholes)
10. [The rules we must follow](#10-the-rules-we-must-follow)
11. [How to improve the leaderboard score](#11-how-to-improve-the-leaderboard-score)
12. [What is still left to do](#12-what-is-still-left-to-do)
13. [Glossary](#13-glossary)

---

## 1. The problem in one example

Three independent sources describe businesses. None of them shares an ID with
the others, and the text is messy.

| Source | ID | Name | Address |
|---|---|---|---|
| Source 1 | S1-001 | Shree Best Business Private Limited | 406, Manas Nagar Colony, Ankit Apartment, Lucknow, Uttar Pradesh |
| Source 2 | S2-010 | श्री बेस्ट बिजनेस प्राइवेट लिमिटेड | 406, MANAS NAGAR COLONY, ANKIT APARTMENT, LUCKNOW, उत्तर प्रदेश |
| Source 3 | S3-020 | Shree Best Business Private-Ltd | Door No 851 406, Lucknow, UP |

All three are the same business. **Our job:** for every Source 1 record, list
every Source 2 and Source 3 record that is the same business.

- A Source 1 record can have **zero, one, or many** matches.
- Matches can come from both sources, and several from the same source.
- Source 1 has no duplicates inside itself. It is the "reference" list.

**Size of the job (test set):** 1.73 million Source 1 records, and 9.97
million Source 2 + Source 3 records to search through. Comparing every pair
would be 1.7 × 10¹³ comparisons, which is impossible. That shapes the whole
design.

---

## 2. How we are scored

The metric is **F0.5**, calculated **separately for each Source 1 record** and
then **averaged**.

For one Source 1 record:

- **Precision** = of the matches we listed, how many are correct?
- **Recall** = of the true matches, how many did we list?
- **F0.5** combines them and cares **twice as much about precision**.

In counts: `F0.5 = 1.25·TP / (1.25·TP + FP + 0.25·FN)`
(TP = correct matches listed, FP = wrong matches listed, FN = true matches missed).

What this means in practice:

| Situation | Effect |
|---|---|
| One **wrong** match listed | Costs **4 times** as much as one missed match |
| A record with **no** true matches ("singleton"), we list nothing | Scores **1.0** |
| A singleton, we list even one wrong match | Scores **0.0** |
| A record with true matches, we list nothing | Scores **0.0** |

So the score rewards being **careful**: list a match only when confident.

**Leaderboards:**
- The **public** leaderboard scores a hidden subset of the test set. It is
  feedback during the challenge.
- The **private** leaderboard scores the rest. **It decides the final ranking.**
- We always upload predictions for all test records. We can upload 5 times
  per day.

---

## 3. What the data taught us

These facts were measured on the training data (Step 2, the audit). Each one
changed a design decision.

| Fact | Consequence |
|---|---|
| Train: 2.2M Source 1, 5.0M Source 2, 5.3M Source 3 records. 7.64M true pairs. | Everything must be memory-efficient. |
| **Country agrees in 100% of the 7.64M true pairs.** | We only search within the same country. This is safe and makes the search ~3× smaller. |
| **No Source 2/3 record belongs to two Source 1 records** (0 of 7.64M). | If two records claim the same target, at most one is right. Used in Step 8. |
| Only **5.6%** of Source 1 records are singletons. | Predicting nothing scores just 0.056. The points are in matching. |
| On average **3.67** matches per record, up to 11. | This is a multi-match task. Never "pick the best one". |
| **23%** of Indian Source 2 names are in one of **9 Indic scripts** (Hindi, Tamil, Kannada, ...). | Name comparison fails on these unless we convert the script to Latin letters. |
| Noise includes typos (`Autrey` → `Atsryi`), drifted street numbers (`2046` → `2048`), replaced cities, reordered words, `Street` ↔ `Saint`. | No single rule catches everything. We combine many signals. |
| **Test only:** France is 15% of the test set and never appears in training. | Nothing may be US/India-specific. France is our biggest unknown. |

Test set by country: India 809,986 (47%), US 663,106 (38%), France 259,452 (15%).

---

## 4. The pipeline at a glance

```
 raw TSV files
      │
      ▼
 [Normalize]   clean text: lowercase, remove accents, Hindi/Tamil/... → Latin
      │
      ▼
 [Blocking]    for each Source 1 record, find ~50 plausible candidates
      │         (out of millions) using shared "keys"
      ▼
 [Features]    for each (record, candidate) pair, compute 56 numbers:
      │         name similarity, address similarity, shared numbers, ...
      ▼
 [Model]       LightGBM reads the 56 numbers → probability "same business"
      │
      ▼
 [Decision]    keep candidates with probability ≥ 0.80, then the
      │         "one owner" rule
      ▼
 matching_results.tsv   (uploaded to the leaderboard)
 candidate_pairs.tsv    (the ~50 candidates per record; goes in the final ZIP)
```

Two ideas run through everything:

1. **Blocking finds, the model decides.** Blocking is generous (it keeps
   ~50 candidates). The model is strict (it keeps ~3).
2. **Measure, don't guess.** Every setting was chosen from a measured number
   on held-out data.

---

## 5. Each step, explained

### Step 0: Setup
Folder structure required by the submission rules, pinned package versions
(`requirements.txt`), and a `.gitignore` so the 2.4 GB dataset and big output
files never reach GitHub.

### Step 1: The scorer and file handling
- We wrote the competition metric ourselves and tested it against the
  worked example in the problem statement (0.714). Every later decision is
  judged by this scorer, so it has to be exactly right.
- Files are read "strictly": pandas would otherwise silently turn the text
  `NA` into a missing value.
- The writer enforces the output rules (one row per record, empty field for
  no matches, no duplicates, only S2/S3 IDs).
- IDs like `S2-566025912` are stored as small integers instead of text. This
  cuts memory from roughly 800 MB to 40 MB for the test IDs.

### Step 2: Audit the data
Measured the facts in Section 3.

### Step 3: Honest validation split
We have no test answers, so we hide part of the training data and score
ourselves on it.

**The trap:** if you split *pairs* randomly, the same business lands on both
sides, and your score looks great but means nothing.

**Our fix:** split by *business*. A Source 1 record and all its true matches
always stay together. The training set is split into 5 "folds", each a fair
miniature (same country mix, same singleton rate). **Fold 0 is our validation
set.** Nothing from fold 0 is ever used for training.

### Step 4: Text cleaning ("normalization")
Cleaning is a bet: every rule can be wrong (`St` means Street *or* Saint). So
instead of one cleaned version, each name and address gets several **views**:

| View | Example |
|---|---|
| raw | `PH Émpire Douglas LLC` |
| lower | `ph émpire douglas llc` |
| translit (Indic → Latin) | `श्री बेस्ट` → `shri best` |
| plain (punctuation → spaces) | `private-ltd` → `private ltd` |
| folded (accents removed) | `ph empire douglas llc` |
| nosuffix (legal words removed) | `ph empire douglas` |

**Transliteration** is our own code. The nine Indic scripts share one Unicode
layout, so one table converts all of them. Measured on real true pairs,
median name similarity for cross-script pairs went from **9.8 to 78.6** (out
of 100). No external data is involved; it is pure character mapping.

### Step 5: Blocking (finding candidates)
Comparing every pair is impossible, so each record gets a set of short
**keys**. Two records become candidates when they share a key. There are 9
key types ("channels"), all starting with the country:

| Channel | Key example | Catches |
|---|---|---|
| name_exact | `US\|nematech solutions llc` | clean duplicates |
| name_sorted | words sorted alphabetically | word order changes |
| name_token | one rare name word | typos in the other words |
| name_pair | two name words together | common names with one rare word |
| addr_token | one rare address word | a name that changed completely |
| num_token | street number + address word | cross-script names (addresses stay Latin) |
| num_set | all numbers in the address | reordered addresses |
| name_num | name word + street number | same name, different branch |
| name_glued | name with spaces removed | `vdrcornerstonepegasus.com` |

**Ranking:** each candidate gets a score: the sum over shared keys of
`log(10 million / how many records share that key)`. A key shared by 2
records is strong evidence; one shared by 3,000 is weak. Keys shared by more
than **3,000** records are ignored (the "cap"). We keep the **top 25 per
source** (up to 50 candidates per record).

**Three bugs found and fixed by measuring at full scale:**
1. Leading zeros: `09585` and `9585` were treated as different numbers.
2. Each record kept the *alphabetically first* 6 words instead of the
   *rarest* 6. A true match with an identical address ranked 720th because
   words like "apartment" and "floor" were kept. Fixed with a word-count table.
3. Website-style names (`richmondacademy.com`) shared no word with the real
   name. Fixed with the name_glued channel.

**Memory:** keys are stored as numbers in flat arrays, and the index is built
one country at a time. The laptop crashed twice before this (see Section 9).

### Step 6: Pair features
For every (record, candidate) pair we compute **56 numbers**, for example:

| Group | Examples |
|---|---|
| Name similarity | edit ratio, Jaro-Winkler, token-sort, token-set, same without legal words |
| Word overlap | plain overlap, and overlap weighted by how rare the shared words are |
| Address similarity | the same measures on addresses |
| Numbers | shared numbers, first number equal, closest gap (2046 vs 2048), postal code equal/conflict |
| Missing data | address empty on either side |
| Blocking | retrieval score, rank, which channels found it |
| Context | is this the best candidate among this record's ~50 rivals, and by how much |

**Rule:** if either side of a field is empty, its similarity is "unknown"
(NaN), never "identical". Two blank addresses are not evidence of a match.

We deliberately left out country (the model would memorize US vs India, and
France never appears in training) and IDs.

Single-feature power (how well each number alone separates matches, 0.5 =
useless, 1.0 = perfect): address token-set **0.970**, retrieval score 0.964,
name similarity only ~0.79. **Addresses separate matches much better than
names**, because blocking already picked candidates with similar names.

### Step 7: The model
A **LightGBM** classifier (a fast decision-tree model, MIT licensed) learns
from 5 million labelled pairs (100,000 training businesses) which
combinations of the 56 numbers mean "same business". It stops adding trees
when a separate "dev" slice (10% of training businesses) stops improving.

Most useful features: `combo_rank` (rank among rivals) 34%, retrieval score
23%, address similarity 10%.

### Step 8: The decision rule
The model gives each pair a probability. Tested four rules, each tuned on one
half of the validation businesses and scored on the other half:

| Rule | Held-out score | Verdict |
|---|---|---|
| Keep if probability ≥ threshold | 0.9531 | good |
| **Threshold + one owner** | **0.9532** | **chosen** |
| Threshold + "rescue" the best candidate of empty records | 0.9528 | rejected (hurts singletons) |
| Expected-F0.5 per record | 0.9515 | rejected (hurts singletons) |

**Final rule:** keep candidates with probability **≥ 0.80**, then if two
records keep the same target, only the higher probability keeps it.
The score is flat between 0.77 and 0.82, so the choice is stable.

### Step 9: Predict on the test set
`python -m business_er predict` runs the whole pipeline on the test set, one
country at a time, saving each stage to disk so a stopped run can resume.
It took about 65 minutes. The output passed the official validator (`PASS`),
and every one of 86.6 million candidate IDs exists in the test set.

---

## 6. Every code file and what it does

All code is under `code/business_entity_resolution/`.

### `src/business_er/`: the pipeline

| File | What it does |
|---|---|
| `ids.py` | Converts IDs like `S2-47` to integers and back, for memory. |
| `io.py` | Reads TSV files strictly; writes the two submission files with every format rule enforced, including streaming writers for the 86M-ID file. |
| `metrics.py` | The exact competition score (per record, then averaged), the blocking "ceiling", and a fast version for millions of pairs (tested to agree with the exact one). |
| `splits.py` | Builds the train/validation folds by business and proves no true match crosses a fold. |
| `normalize.py` | The text views and the Indic → Latin transliteration. |
| `retrieve.py` | Blocking: keys, word-rarity table, the index (per country), ranking, top-K. |
| `features.py` | The 56 pair features, computed in parallel blocks. |
| `train.py` | Trains, saves and loads the LightGBM model; refuses a wrong feature order. |
| `evaluate.py` | The decision rules (threshold, one owner, rescue, expected-F0.5) and scoring of a set of decisions. |
| `predict.py` | The complete test-set run: blocking → features → model → decision → files. Resumable. |
| `__main__.py` | The command line: `python -m business_er predict ...` |

### `experiments/`: the measurements behind every decision

| File | What it measured |
|---|---|
| `run_capped.sh` | **Not an experiment.** Launches any script with a hard memory cap so it cannot freeze the laptop. Always use it for big runs. |
| `build_splits.py` | Builds and saves the 5-fold and 25-fold splits. |
| `blocking_channels.py` | First look: recall of each key channel on a small slice. |
| `blocking_rank.py` | Ranking + top-K on the small slice. |
| `blocking_fullscale.py` | Blocking at full scale (10.3M records), the diagnosis of misses, and the 3 fixes. |
| `build_pairs.py` | Builds the labelled training (5M) and validation (1M) pair tables with features. |
| `train_eval.py` | Trains the model and scores validation with the real metric. |
| `decision_rules.py` | Compares the four decision rules with held-out tuning. |

### `tests/`: 142 automatic checks
One test file per module. They use tiny invented records, need no dataset,
and run in about 3 seconds: `python3 -m pytest -q`. Among other things they
check the scorer against hand calculations, that blank fields never count as
a match, that France works with no special code, that parallel runs equal
serial runs, and that the output passes the official validator.

### Not in git (too big or private), all in the project root
- `amazon_ml_dataset/`: the competition data (download it from the portal).
- `artifacts/`: splits, word-count tables, pair tables, the trained model,
  cached test stages, logs.
- `output/`: the two submission files.

---

## 7. How to run things

Run everything from the project root. **Always launch heavy work through
`run_capped.sh`.** It sets a memory cap from the memory that is free right
now and stops the job cleanly if it exceeds it.

```bash
# tests (no data needed)
cd code/business_entity_resolution && python3 -m pytest -q && cd ../..

# full test prediction (about 65 min; close Firefox first)
code/business_entity_resolution/experiments/run_capped.sh -m business_er predict \
    --data-dir amazon_ml_dataset/student_resource/dataset --split test \
    --model artifacts/models/matcher_v1.lgb \
    --freq artifacts/token_freq_excl_fold0.npz \
    --output-dir output --work-dir artifacts/test_work

# official validator (run from student_resource/)
cd amazon_ml_dataset/student_resource
python3 utils/validate_submission.py --matching ../../output/matching_results.tsv \
    --candidate ../../output/candidate_pairs.tsv --test-dir dataset/test
```

Useful checks while a job runs:

```bash
systemctl --user list-units 'run-*.service' --no-pager            # is it running?
systemctl --user status 'run-*.service' --no-pager | grep Memory   # its memory
tail -f artifacts/logs/<log file>                                  # progress (Ctrl+C stops watching only)
free -h                                                            # laptop memory
```

Training commands (splits → word counts → pair tables → model) currently live
in the `experiments/` scripts. Section 12 lists turning them into one
documented command as remaining work.

---

## 8. Results so far (all measured)

All validation numbers: 20,000 fold-0 businesses searched against all 10.3M
training records, with the exact competition metric. Singletons and missed
matches are included.

| What | Value |
|---|---|
| Predict nothing (baseline) | 0.056 |
| Blocking recall (share of true matches in the ~50 candidates) | 0.943 |
| Blocking ceiling (score of a perfect model on those candidates) | 0.979 |
| **Our model, validation score** | **0.953** |
| Pair precision of kept matches | about 0.99 |

Blocking improvements (recall at 50 candidates):
start 0.9295 → leading zeros 0.9345 → rarest words 0.9414 → glued names 0.9429.
A larger cap (10,000) was tested: +0.0004 ceiling for 3× the time, rejected.

Where the remaining validation loss (0.047) comes from:

| Mistake | Loss |
|---|---|
| Has matches, we missed some | 0.025 |
| Has matches, we predicted nothing | 0.011 |
| Has matches, we added a wrong one | 0.009 |
| Singleton, we gave it a wrong match | 0.002 |

**More than half of all lost points come from records where blocking missed a
true match.** The model is very precise. It mostly lacks the chance to see
some true matches.

Test predictions (first submission):

| Country | Records | Given ≥1 match | Matches per record |
|---|---|---|---|
| France | 259,452 | 95.6% | 3.55 |
| India | 809,986 | 92.9% | 3.03 |
| US | 663,106 | 93.8% | 3.19 |

(In training, 94.4% of records truly have matches.)

---

## 9. Weak spots and risks ("loopholes")

Here "loopholes" means weaknesses, failure modes and risks in our solution:
places where points are lost or things could go wrong. It does not mean ways
around the rules.

### Accuracy weak spots

1. **Blocking misses 5.7% of true matches.** 4.4% are found but ranked below
   the top 50; 1.3% are never found (no shared key, e.g. `Shree Products` vs
   `Gildzeta`). This is the largest single source of lost points.
2. **France is untested.** The model never saw a French example. French word
   rarity currently falls back to US/India counts (it does not know `rue` is
   common in France). Our abbreviation list does not cover `R.` = `rue`,
   `BD` = `boulevard`, `5 bis`.
3. **Only 6% of the training data was used** (100,000 of ~1.77M training
   businesses). The model is probably under-trained.
4. **Token-set similarity can be fooled.** `nematech` vs `nematech consulting`
   scores 1.0 because one is contained in the other. The model has other
   features to compensate, but it is a known weakness.
5. **Transliteration is rule-based.** `praivet limited` vs `private limited`
   still differ, and removing legal words hurts cross-script names.
6. **Validation is slightly easier than the test.** Validation has ~4.7
   records per business to search; the test has ~5.8.
7. **Only one model, untuned.** No hyperparameter search, no ensemble.

### Engineering risks

1. **Memory.** The laptop (15 GB) crashed twice from running out of memory,
   and one run froze. Now every heavy job goes through `run_capped.sh`,
   `earlyoom` is installed, the index is built per country, and text is
   processed in chunks. Still: close Firefox during big runs.
2. **Scores are saved per country, not per chunk.** A stop in the middle of a
   country redoes that country's scoring (~25 min).
3. **The official validator's strict mode (`--check-ids`) runs out of
   memory** on the laptop. We check IDs with our own lighter script instead.
4. **Reproducibility gap.** The training steps are spread over experiment
   scripts, and the trained model is not in git. The final ZIP must let a
   reviewer regenerate the outputs, so this must be closed (Section 12).
5. **Public leaderboard overfitting.** Tuning to the public score can hurt
   the private score that decides the ranking. Trust local validation.

---

## 10. The rules we must follow

From the problem statement plus the organizers' answers to participant
questions:

| Allowed | Not allowed |
|---|---|
| Any algorithm using only the provided records | External databases, APIs, geocoding, entity lookup |
| RapidFuzz, scikit-learn, LightGBM, pandas, jellyfish | Packages that bundle geo/postal/business data (e.g. libpostal, gazetteers, city/state tables) |
| Small hand-written normalization dictionaries | Internet-sourced data augmentation |
| Unsupervised statistics on the **test** files (word counts, TF-IDF, index) | Hosted LLM APIs (ChatGPT, Claude, Gemini) inside the solution |
| Self-training / pseudo-labels / synthetic pairs from the provided records | AWS Entity Resolution |
| Pretrained open models if **MIT or Apache-2.0**, **≤ 8B parameters each**, run offline, fine-tuned only on the provided data | Sharing predictions with other teams |
| Using AI coding assistants | |

Other points: country may be used as long as it is treated as an open set of
labels (no hard-coding US/India). Keep **all** records with identical text (do
not keep only one). The model's license is checked for every model used.

Our current solution uses only local code and LightGBM (MIT). No external
data, no pretrained weights.

---

## 11. How to improve the leaderboard score

Ordered by expected value for the effort, based on our measurements.

| # | Idea | Why it should help | Effort |
|---|---|---|---|
| 1 | **Word counts from the test files** (now officially allowed) | Gives France its own word rarity; fixes a known France weakness | Small |
| 2 | **Train on 10× more data** (1M businesses instead of 100k) | Tree models usually improve with more data; we used 6% | Small–medium |
| 3 | **French abbreviation dictionary** (`R.`→`rue`, `AV`→`avenue`, `BD`→`boulevard`, `bis/ter`) | Small hand-written dictionaries are allowed; France is 15% of the test set | Small |
| 4 | **Pseudo-labels for France** (allowed) | Use the model's most confident French matches as extra training examples so it learns French patterns | Medium |
| 5 | **Keep 100 candidates instead of 50** | Blocking ceiling rises from 0.979 to 0.984 (measured); doubles the run time | Medium |
| 6 | **Re-rank before cutting to 50** | 4.4% of true matches are found but ranked too low; a cheap name/address similarity could lift them | Medium |
| 7 | **Simulate an unseen country** (train without India, test on India) | Measures how much a new country costs us; guides France work | Medium |
| 8 | **Error analysis → new features** | Look at the most confident mistakes and add a feature for the common pattern | Ongoing |
| 9 | **Tune LightGBM settings** | Usually a small, reliable gain | Small |
| 10 | **Neural re-ranker on a GPU** (e.g. a small multilingual model, MIT/Apache) | Reads the text directly; understands Hindi/French/website names; use only on the ~1–2% uncertain pairs | Large; needs a GPU and time |

**How to work:** change one thing at a time, measure it on validation, keep it
only if it helps. Submit to the leaderboard only to confirm real improvements
(5 per day). Remember that the private leaderboard decides the ranking.

---

## 12. What is still left to do

**Required (the final ZIP, Step 10):**

```
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv        ✅ have it (must be the same file as the leaderboard upload)
│   └── candidate_pairs.tsv         ✅ have it
├── code/business_entity_resolution/
│   ├── src/                        ✅ have it
│   ├── README.md                   ⚠️ outdated, must be rewritten
│   └── requirements.txt            ⚠️ re-check versions
└── Documentation_template.md       ❌ not filled in yet
```

- Rewrite the README with one exact end-to-end reproduction command.
- Make the training steps runnable from `src/` (or clearly documented
  scripts), and decide how the trained model is included.
- Fill in the organizers' `Documentation_template.md` using the measured
  numbers in this guide.
- Build the ZIP and check the uploaded file and the zipped file are
  byte-identical (same SHA-256 fingerprint).

**Optional:** the improvements in Section 11, time permitting.

---

## 13. Glossary

| Term | Meaning |
|---|---|
| Entity resolution | Deciding which records describe the same real-world thing |
| Anchor | A Source 1 record we are finding matches for |
| Candidate | A Source 2/3 record proposed as a possible match |
| Singleton | A Source 1 record with no true matches |
| Blocking | Cheaply narrowing millions of records to a short candidate list |
| Key / channel | A short string two records must share to become candidates; a channel is one kind of key |
| Cap | Keys shared by more records than this are ignored |
| Top-K | How many candidates we keep per source (25) |
| Recall | Share of true matches we found |
| Precision | Share of our listed matches that are correct |
| F0.5 | Score combining precision and recall, weighting precision more |
| Macro average | Score each record separately, then average |
| Oracle / ceiling | The score a perfect model would get with our candidates |
| Feature | One number describing a pair (e.g. name similarity) |
| AUC | How well one number separates matches from non-matches (0.5 useless, 1.0 perfect) |
| LightGBM | A fast decision-tree learning library (MIT license) |
| Threshold | The probability above which we say "match" (0.80) |
| Fold | One part of the training data held out for honest measurement |
| Leak | When information from the validation data sneaks into training, making scores look better than they are |
| NaN | "Not a number": our marker for "unknown", e.g. when a field is empty |
| Transliteration | Writing a word from one script in another (`श्री` → `shri`) |
| Pseudo-label | Using the model's own confident predictions on unlabelled data as training labels |
| OOM | "Out of memory": the laptop ran out of RAM |
