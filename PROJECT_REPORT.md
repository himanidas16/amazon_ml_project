# Project Report — Business Entity Resolution (Amazon ML Challenge 2026)

A complete, plain-language record of what was built, **why**, and what each
change measured. Read `PROJECT_GUIDE.md` first if you are new: it explains the
problem, the metric and the basic pipeline. This report is the detailed
history on top of it.

Status as of 26 September 2026, ~17:00. All numbers are **measured**; anything
not yet measured is labelled *planned* or *in progress*.

---

## Contents

1. [Scoreboard: every version at a glance](#1-scoreboard-every-version-at-a-glance)
2. [Version 1 — the first complete pipeline](#2-version-1--the-first-complete-pipeline)
3. [Version 2 — France fixes](#3-version-2--france-fixes)
4. [Version 3 — better features, more data, wider candidates, stage 2](#4-version-3--better-features-more-data-wider-candidates-stage-2)
5. [Version 4 (in progress) — extra retrieval channels](#5-version-4-in-progress--extra-retrieval-channels)
6. [Work by the second assistant](#6-work-by-the-second-assistant)
7. [Ideas tested and rejected](#7-ideas-tested-and-rejected)
8. [Engineering: memory, speed, safety](#8-engineering-memory-speed-safety)
9. [How the pipeline runs now](#9-how-the-pipeline-runs-now)
10. [Rules and compliance](#10-rules-and-compliance)
11. [Where the remaining points are](#11-where-the-remaining-points-are)
12. [File map](#12-file-map)

---

## 1. Scoreboard: every version at a glance

"Validation" = 20,000 held-out training businesses (fold 0) searched against
the full 10.3M-record training pool, scored with the exact competition metric
(F0.5 per business, averaged; singletons and missed matches included).
"Ceiling" = the score a perfect model would get with our candidate lists.

| Version | What changed | Validation | Ceiling | Public leaderboard |
|---|---|---|---|---|
| v1 | first full pipeline | 0.9533 | 0.979 | **0.943** |
| v2 | France-aware word counts, French abbreviations, `cie`/`ets` (same model as v1) | (no France labels) | 0.979 | **0.948** |
| v1.2 features, 100k | missing-aware context, number forensics, pool counts | 0.9558 | 0.979 | — |
| v1.2 features, 300k | + 3× training data | 0.9586 | 0.979 | — |
| **v3 (stage 1)** | + wide candidate selection | **0.9654** | **0.988** | *in progress* |
| v3 + stage 2 | + competing-claims model | +0.0024 on the complete world* | 0.988 | *in progress* |
| v4 *(planned)* | + extra address channels | — | **0.9905** (probe) | — |

\* Stage 2 was measured on a different evaluation (all 441k fold-0 businesses
against fold-0 records), so its gain cannot simply be added to 0.9654.

Per country (validation): v1 US 0.962 / India 0.941 → v3 US **0.972** /
India **0.956**. India is still the weaker country; France has no labels, and
the v1→v2 leaderboard jump (+0.005 with the same model) suggests France gained
about +0.03 from the France fixes.

---

## 2. Version 1 — the first complete pipeline

Built step by step, each step measured before the next (details in
`PROJECT_GUIDE.md`):

| Step | What | Why |
|---|---|---|
| Scorer, I/O, ids | exact metric; strict TSV reading; writers that enforce every output rule | a wrong scorer makes every later decision wrong; a malformed file is rejected |
| Audit | country agrees in 100% of 7.6M true pairs; no record has two owners; 5.6% singletons; 23% of Indian S2 names in Indic scripts | each fact shaped a design choice |
| Split | 5 folds by business, fold 0 = validation, no record of fold 0 ever in training | random pair splits leak the answer |
| Normalization | 6 text views + our own Indic→Latin transliteration | cross-script name similarity 9.8 → 78.6 |
| Blocking | 9 key channels, country-scoped, IDF-ranked, top 25/source | 10¹³ pairs → ~50 per business; recall 0.943 after 3 measured fixes |
| Features | 56 numbers per pair | the model cannot read text |
| Model | LightGBM on 100k businesses (5M pairs) | fast, handles missing values, MIT licence |
| Decision | threshold 0.80 + one-owner rule | tuned on one half of validation, confirmed on the other |

Result: validation 0.953, **public 0.943**.

---

## 3. Version 2 — France fixes

France is 15% of the test set and absent from training. Three findings drove v2:

1. **Word rarity was wrong for France.** Blocking keeps each record's rarest
   words as keys, with rarity counted on training data — which has no French.
   French legal forms (`sarl`, `sasu`) looked "never seen, so rare" and took
   the best key slots. **Fix:** count words on the **test files**
   (unsupervised statistics — the organizers confirmed this is allowed).
2. **French abbreviations.** `R.` appears ~30k times for `rue`, plus `AV`,
   `BD`, `ALL`, `CH`, `ST` (= Saint). Measured first on US/India: `st` is
   Street in 74k US addresses, `r` an initial in 8k Indian ones — so a global
   table would damage them. **Fix:** a small hand-written table applied **only
   to records labelled France** (country treated as an open set).
3. **French legal forms** `cie`, `ets`, `etablissements` added to the legal
   suffix list.

Also added: `python -m business_er token-freq` (word counts for any split) and
a **manifest** in each prediction work folder that refuses to reuse saved
results built with different settings (prevents stale-cache mixing).

Result: **public 0.948** (+0.005 with the same model).

---

## 4. Version 3 — better features, more data, wider candidates, stage 2

### 4.1 Error analysis that drove the changes

On validation, the v1 loss (0.047) split into: 0.021 true matches never in
the candidate list, 0.017 true matches the model rejected, 0.009 wrong matches
accepted. Reading hundreds of real errors showed:

- **`combo_rank` was biased.** The model's most important feature (34% of its
  decisions) ranked candidates by name + address similarity with a *missing*
  field counted as 0. True matches with an empty address or a garbled name were
  ranked last and rejected — 1,896 of ~3,400 rejected true pairs ranked 10th or
  worse.
- **Numbers split or truncated by noise:** `76` vs `7-6`, `201` vs `2-01`,
  `1703` vs `170`.
- **Same address, different name** is sometimes a match (gibberish trade name)
  and sometimes a different business in the same building; **same name, no
  address** is sometimes a match (distinctive name) and sometimes not (common
  name). What separates them: **how many records share that exact address /
  name**.
- Only **0.06%** of true pairs have both name and address dissimilar to S1 —
  the information is almost always in the pair itself.

### 4.2 New features (v1.2, 70 features)

| Feature group | Fixes |
|---|---|
| `avail_mean`, `avail_rank`, `avail_gap`, `name_rank_avail`, `addr_rank_avail`, `n_strong_rivals` | rank candidates on the fields that are **present** — the combo_rank bias |
| `numj_jacc`, `numj_first_equal` | compare numbers with in-number hyphens removed (`7-6` = `76`; spaces never joined) |
| `num_prefix` | one number is the other plus one digit (`170` / `1703`) |
| postal features by raw digit width | `01234` stays a 5-digit postal code |
| `t_name_cnt`, `t_addr_cnt`, `a_name_cnt`, `a_addr_cnt` | how many pool records share the exact cleaned name / address (shared buildings, chains) |
| `addr_exact` blocking channel | the whole cleaned address as a key (also finds gibberish-name matches) |

Measured: 0.9533 → **0.9558** at 100k businesses. `avail_mean` became the top
feature (48%), replacing the biased combo_rank.

### 4.3 More training data

Training data built with `build-pairs` at 300k businesses. To fit memory, all
positives and "hard" negatives are kept and only 30% of easy negatives, each
weighted ×3.3 so the model's probabilities stay calibrated (a test proves the
unweighted version inflates them). Measured: **0.9586** (+0.0028 over 100k).

### 4.4 Wide candidate selection (the biggest gain)

4.4% of true matches were found by blocking but ranked below the top 25 and
thrown away. New stage in candidate generation (`select.py`):

1. blocking keeps the top **150** per source instead of 25;
2. a **fixed formula** — the average token-set similarity of the name and
   address fields that are present — ranks those 150;
3. keep blocking's top 25 **plus** the formula's top 10 per source.

Measured (probe): ceiling 0.979 → **0.988**. A learned ranker was only
slightly better (0.9886) but would have forced a ~7 GB candidate file, because
the organizers require `candidate_pairs.tsv` to contain the input to the
**first scoring model** — a fixed formula is part of blocking, so the export
stays at the final ~60 candidates.

Retrained on selected candidates: validation **0.9654** (US 0.972, India
0.956). Best threshold 0.75.

### 4.5 Stage 2 — judging competing claims

Every S2/S3 record belongs to at most one S1 business. Stage 1 judges each
pair alone; stage 2 (`stage2.py`) adds, per pair: how many businesses claim
the record, the best **other** claim's probability, this claim's margin and
rank, and the business-side equivalents. It only re-judges pairs with stage-1
probability ≥ 0.005 (~8%); the others keep their probability.

Measured honestly on a **complete** validation world (all 441k fold-0
businesses vs fold-0 records): each half scored with a threshold chosen on the
other half, one-owner always seeing every claim: **0.9774 → 0.9798
(+0.0024)**. Caveat: this world's pool is smaller than the test pool.

A second finding: in the complete world the best stage-1 threshold was
**0.32**, not 0.80 — with every competitor present, the one-owner rule removes
most wrong claims itself. The test set is a complete world, so lower
thresholds will be tested on the leaderboard (a new threshold only re-runs the
final decision, a few minutes).

### 4.6 v3 test run *(in progress)*

The queue (`experiments/queue_round4.sh`) runs: stage 2 on the selected world
→ test prediction → submission check. It will produce **two files** from one
run, so the leaderboard shows each part's effect: **v3a** (stage 1 only,
threshold 0.75) and **v3b** (with stage 2, if its held-out gain > 0.001).

---

## 5. Version 4 (in progress) — extra retrieval channels

A review of 400 true matches still missed by the selection (3.5% of true
pairs) found **80% have a name or address ≥ 80% similar** — they are findable;
our keys just do not connect them (truncated numbers `403`→`40`,
`D-199`→`D-99`; generic names outranked by look-alikes). Only 0.2% are truly
hopeless.

Three new channels, measured as additions to the current selection
(`experiments/extra_channels_probe.py`, each adding its own top 10/source):

| Candidate set | Recall | Cand/business | Ceiling |
|---|---|---|---|
| current selection | 0.9654 | 59.9 | 0.9880 |
| + `addr_pair` (pairs of the 3 rarest address words) | 0.9720 | 70.1 | 0.9902 |
| + `addr_nonum` (address without numbers) | 0.9669 | 61.4 | 0.9887 |
| + `num_trunc` (first 2 digits of a number + rarest address word) | 0.9671 | 64.9 | 0.9887 |
| **+ all three** | **0.9728** | 75.8 | **0.9905** |

The ceiling passes 0.99 for the first time. Being integrated as an **opt-in
switch** (off by default, so the running v3 is unaffected); then an overnight
v4 run: rebuild → retrain → stage 2 → test.

---

## 6. Work by the second assistant

A teammate's AI assistant worked in the same folder (reports in
`EXPERIMENTS_2026_09_26.md` and `IMPROVEMENT_RUN_2026_09_26.md`). Key results:

- **Found and fixed a real bug** in `build_pairs_v2.py`: labels mixed global
  and per-country business numbers, which would have marked true matches as
  negatives (`labels.py` + test). The fix was verified before any training
  used it.
- **Error audit** (`error_audit.py`) with exact loss attribution: 0.0120
  coverage loss, 0.0222 decision loss for the selection model.
- **Character-n-gram retrieval** (`char_retrieve.py`, pilot on 1,000
  businesses): address grams raised that pilot's ceiling 0.9884 → 0.9914 at
  the cost of ~30 more candidates per business; name grams 0.9884 → 0.9899.
- **Rescue with a rank-free matcher** (`rescue.py`): on the pilot's held-out
  half, +0.0015 (95% interval +0.0005 to +0.0028). Needs confirmation on fresh
  businesses before use.
- Memory-mapped index builder (`disk_index.py`), competition features for
  selected rows (`competition.py`), a much faster Indic-script detector
  (identical output, tested), pair-table loader (`pair_data.py`).
- **Rejected by measurement:** tie-aware context model, exact-text duplicate
  expansion, rank-free model as a replacement, exact expected-F0.5 decision
  (−0.0018, significant).

Its reviews also correctly flagged: stage-2 memory at test scale, stage-2
threshold chosen on its own rows, the queue continuing after failures — all
fixed (section 8).

---

## 7. Ideas tested and rejected

| Idea | Result | Why rejected |
|---|---|---|
| Blocking cap 10,000 (vs 3,000) | ceiling +0.0004, 3× slower | not worth it |
| "Rescue" the best candidate of empty businesses | 0.9528 vs 0.9532 | hurts singletons |
| Expected-F0.5 set selection (approximate and exact) | −0.0017 / −0.0018 | hurts singletons |
| Sibling/cross-record evidence as the main lever | only 0.06% of true pairs need it | too rare |
| Learned candidate ranker | +0.0006 ceiling over the formula | would force a ~7 GB candidate file |
| Exact-text duplicate expansion | +1 true pair | negligible |

---

## 8. Engineering: memory, speed, safety

The laptop has 15 GB of RAM; most of the engineering effort went into making
the pipeline run reliably within it.

| Problem (what happened) | Fix |
|---|---|
| Two full laptop crashes (a job exceeded free memory) | `experiments/run_capped.sh`: cap = free memory − 2.5 GB, whole job + workers stopped together; `earlyoom` installed |
| A "soft" memory limit froze a run at 1% speed | hard cap only |
| Orphaned worker processes kept memory | workers die with their parent |
| Writing 86M candidate ids took several copies | streaming, sort-free writer |
| Text preparation held all candidates at once | chunked preparation per country |
| A killed run lost a whole country of work | per-chunk checkpoints; resumable `predict` |
| Laptop auto-suspended overnight (3.5 h lost) | `systemd-inhibit` sleep lock during runs |
| Selection re-prepared every pool record (2+ h) | index keeps its cleaned text; slim preparation; more cores |
| Training copied the feature matrix | LightGBM dataset subsets, disk-backed matrix |
| Stage 2 exceeded memory on 26M pairs | numpy competition features; inputs stored only for re-judged rows |
| The organizers' validator OOMs in strict mode | `check_submission.py`: same rules, streaming, low memory |
| A failed step could feed stale results onward | queues stop on any failure and skip finished steps |

Current test suite: **189 tests**, all passing, including end-to-end runs that
pass the organizers' own validator.

---

## 9. How the pipeline runs now

From the repository root (see `code/business_entity_resolution/README.md`):

```bash
export PYTHONPATH=code/business_entity_resolution/src
RUN=code/business_entity_resolution/experiments/run_capped.sh
DATA=amazon_ml_dataset/student_resource/dataset
SEL="--k-wide 150 --k-formula 10"

$RUN -m business_er split       --data-dir $DATA --artifacts artifacts
$RUN -m business_er train-freq  --data-dir $DATA --artifacts artifacts
$RUN -m business_er build-pairs --data-dir $DATA --artifacts artifacts --out-name pairs_sel \
     --train-anchors 300000 --easy-rate 0.3 $SEL
$RUN -m business_er train       --artifacts artifacts --pairs-name pairs_sel --tag sel_300k
$RUN -m business_er build-pairs --data-dir $DATA --artifacts artifacts --out-name world_sel \
     --train-anchors 0 --val-world $SEL
$RUN -m business_er train-stage2 --world artifacts/world_sel \
     --stage1-model artifacts/models/matcher_sel_300k.lgb --out artifacts/models/stage2_sel
$RUN -m business_er token-freq  --data-dir $DATA --split test --output artifacts/token_freq_test.npz
$RUN -m business_er predict     --data-dir $DATA --split test \
     --model artifacts/models/matcher_sel_300k.lgb --freq artifacts/token_freq_test.npz \
     --output-dir output_v3 --work-dir artifacts/test_work_v3 --threshold 0.75 $SEL \
     [--stage2-model artifacts/models/stage2_sel.lgb --stage2-threshold T2]
python3 code/business_entity_resolution/experiments/check_submission.py output_v3 $DATA/test
$RUN -m business_er package --team-name TEAM --repo-root . --output-dir output_v3 \
     --doc Documentation_template.md --dest dist
```

---

## 10. Rules and compliance

From the problem statement and the organizers' answers (query form):

- No external databases, APIs, geocoders or gazetteers — none used.
- Unsupervised statistics on the test files (word counts) — allowed; used.
- Small hand-written normalization dictionaries — allowed; the France table.
- Country as an open set — France runs through the same code as US/India.
- `candidate_pairs.tsv` = input to the first scoring model — satisfied
  (selection is a fixed formula, not a model).
- Model: LightGBM 4.7.0 (MIT). No pretrained weights.
- Other teams' public repositories were deliberately **not** consulted.
- The organizer query-form spreadsheet contains other participants' personal
  data and is excluded from git (`*.xlsx`).

---

## 11. Where the remaining points are

Validation loss of the v3 stage-1 model (0.0346 at the fixed threshold):

| Source | Loss | Next step |
|---|---|---|
| True matches never in the candidate list | 0.0120 | extra channels (ceiling → 0.9905), character retrieval |
| Wrong decisions among candidates | 0.0226 | stage 2 (+0.0024 measured), hard-negative training, more data, possibly a neural reranker |
| France (test only) | unknown | leaderboard readings; self-training only if validated by hiding a country's labels |

Reaching **0.99** needs a ceiling well above 0.99 **and** a decision loss
around a quarter of today's. That is the direction of all remaining work; no
current measurement guarantees it.

---

## 12. File map

**Pipeline (`code/business_entity_resolution/src/business_er/`)**

| File | Role |
|---|---|
| `io.py`, `ids.py` | strict reading; streaming, rule-enforcing writers; compact ids |
| `metrics.py` | exact competition metric, ceiling, fast vectorised form |
| `splits.py`, `labels.py` | business-grouped folds with leak check; candidate labels |
| `normalize.py` | text views, Indic transliteration, France-only abbreviations |
| `retrieve.py` | blocking channels, word-rarity table, per-country index, pool counts |
| `select.py` | wide retrieval + fixed-formula selection |
| `features.py` | 70 pair features (+ cheap features, slim preparation) |
| `train.py`, `pair_data.py` | LightGBM matcher (weights, dataset subsets); shard loader |
| `stage2.py`, `competition.py` | competing-claims model and features |
| `evaluate.py` | decision rules (threshold, one owner, …) |
| `pipeline.py` | `split`, `train-freq`, `build-pairs`, `train` |
| `predict.py` | full test inference (resumable, manifest-guarded, optional stage 2) |
| `package.py` | final ZIP with checks |
| `char_retrieve.py`, `rescue.py`, `exact_decision.py`, `contextual.py`, `disk_index.py` | experimental (second assistant); not in the submission path |
| `__main__.py` | the `python -m business_er` CLI |

**Experiments (`experiments/`)** — the scripts behind every measurement above
(blocking studies, selection/extra-channel probes, stage-2 probe, error audit,
character retrieval, decision rules), the queues (`queue_round*.sh`), the
memory-capped launcher `run_capped.sh` and `check_submission.py`.

**Not in git** (too large or private): the dataset, `artifacts/`, `output*/`,
`share_for_team/`, the query-form spreadsheet.
