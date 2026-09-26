# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]
**Team Members:** [List all team members]
**Submission Date:** [Date]

> DRAFT — numbers marked *(v1)* are from the first model; final numbers are
> filled in after the final run.

---

## 1. Executive Summary

A two-part pipeline built only from the supplied records: **multi-channel
blocking** (country-scoped keys, ranked by key rarity) proposes about 50
candidates per Source 1 record, and a **LightGBM matcher** over ~90 engineered
pair features decides which candidates are the same business, followed by a
threshold tuned on the exact macro F0.5 metric and a one-owner rule (a Source
2/3 record never belongs to two Source 1 records). Key contributions: our own
rule-based **Indic→Latin transliteration** (nine scripts), **missing-aware
rival-context features**, **number forensics** for the injected numeric noise,
and **exact name/address pool counts** that separate shared buildings and chain
names from genuine matches.

---

## 2. Methodology

### 2.1 Problem Analysis

Measured on the training data (2.2M S1, 5.0M S2, 5.3M S3 records; 7.64M true
pairs):

- **Country agrees in 100% of true pairs** → search within country only.
- **No S2/S3 record belongs to two S1 records** (0 of 7.64M) → one-owner rule.
- **5.6% singletons**; 3.67 matches per non-singleton on average (max 11).
- **23% of Indian S2 names** are in one of nine Indic scripts (Devanagari,
  Tamil, Kannada, ...) — phonetic spellings of the Latin name.
- Injected noise: typos (`Autrey`→`Atsryi`), accents (`Émpire`), reordered
  words, legal-suffix changes, website-style names (`vdrcornerstonepegasus.com`),
  gibberish trade names with an unchanged address, drifting or split street
  numbers (`2046`→`2048`, `76`→`7-6`, `1703`→`170`), replaced cities, empty
  addresses, `Street`↔`Saint`.
- Only **0.06%** of true pairs have *both* name and address dissimilar to S1:
  the evidence is almost always in the pair itself.
- Test adds **France (15% of test)**, absent from training: French legal forms
  (SARL, SAS, SCI), `R.`/`AV`/`BD` abbreviations, `5 bis`, region vs department.

### 2.2 Solution Strategy

**Approach Type:** Blocking + Classifier (with rule-based post-processing)
**Core Innovation:** rarity-ranked multi-channel blocking with transliteration,
plus missing-aware context, number-forensics and pool-count features that
target the measured error types.

Validation: business-grouped 5-fold split (each S1 record and all its matches
stay in one fold; stratified by country and match count). Fold 0 is validation
and is excluded from training *and* from the training blocking index (no
validation record ever appears in a training pair, even as a negative).
Validation searches the full 10.3M-record pool, as the test set does.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used** (all prefixed with the country; 10 channels):
  exact normalized name, sorted name tokens, rare name token, pair of name
  tokens, rare address token, street number + address token, number set,
  name token + number, glued name (spaces/legal/web tokens removed), exact
  normalized address.
- **Ranking:** a candidate's score is Σ over shared keys of
  `log(10⁷ / block size)`; keys shared by >3,000 records are ignored; the
  top 25 per source are kept (≤ 50 per S1).
- **Token rarity:** each record keeps its 6 rarest tokens as keys (counted on
  training records outside the validation fold; on the test files at test time
  — unsupervised statistics, confirmed allowed).
- **Candidate pairs generated (test):** ~86.6M (≈ 50 per S1 record).
- **How we ensured true matches were not lost:** every change was measured by
  recall and the *oracle ceiling* (score of a perfect matcher on the
  candidates) at full scale. Fixes found this way: leading zeros (`09585` =
  `9585`), rarest-not-alphabetical token choice, glued names. Validation:
  recall 0.943, ceiling 0.979 *(v1)*. A larger cap (10,000) gained only
  +0.0004 ceiling for 3× the time and was rejected.

---

## 4. Matching Model

**Features used** (~90, all similarities NaN when a field is empty — two
blanks are never evidence):
- **Name:** edit ratio, Jaro-Winkler, token-sort, token-set (with and without
  legal suffixes), exact and glued-name equality, lengths, Jaccard,
  rarity-weighted token overlap and unmatched rare-token share.
- **Address:** edit ratio, token-sort, token-set, Jaccard, rarity-weighted
  overlap, emptiness flags, length ratio; France-only abbreviation expansion
  (`r`→`rue`, `st`→`saint`) — small hand-written table, applied only to
  records labelled France.
- **Numbers:** shared numbers, first-number equality, closest relative gap,
  postal-code equality/conflict (by raw digit width), hyphen-rejoined numbers
  (`7-6` = `76`), truncated-number flag (`170`/`1703`).
- **Pool counts:** how many records in the searched pool share the exact
  cleaned name / address of each side (shared buildings, chains).
- **Retrieval:** blocking score and rank, channel indicators.
- **Rival context:** rank and gap versus the same S1 record's other
  candidates, computed on the fields that are present (missing fields do not
  push a candidate down), number of strong rivals.

**Model type:** LightGBM binary classifier (MIT), early stopping on a
business-level dev split, deterministic settings.
**Threshold selection method:** exact macro F0.5 (per S1 record, singletons
included, blocking misses counted) on held-out validation; the threshold was
tuned on one half of the validation businesses and confirmed on the other
(flat optimum 0.77–0.82 → **0.80**), followed by the one-owner rule.
Rejected after held-out testing: "rescue the best candidate" and
expected-F0.5 set selection (both hurt singletons).

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro), validation:** 0.953 *(v1)* — US 0.962, India 0.941;
  ceiling 0.979; all-empty baseline 0.056. **Public leaderboard:** 0.943 *(v1)*.
- **Common false positives (wrong merges):** a different business at the
  identical address (shared office buildings); near-identical names with one
  changed unit number (`A/3` vs `A/4`); common names with no address.
- **Common false negatives (missed matches):** gibberish trade names with an
  identical address; empty addresses with a distinctive name; hyphen-split or
  truncated street numbers; true matches ranked below 25 by blocking (4.4% of
  true pairs) or sharing no key (1.3%).
- Loss decomposition *(v1)*: 0.021 blocking misses, 0.017 rejected true pairs,
  0.009 accepted wrong pairs.

---

## 6. Conclusion

Measuring the exact metric at full scale, and letting the measured error types
drive each change, mattered more than model complexity: transliteration,
rarity-ranked blocking and missing-aware features addressed specific losses.
The remaining gap is mostly candidate recall and unseen-country (France)
generalisation.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/` — all source in `src/business_er/`, with
`README.md` (exact commands) and `requirements.txt` (pinned). Entry point:
`python -m business_er <command>`:

```
split → train-freq → build-pairs → train → token-freq (test) → predict → package
```

`predict` writes both `output/matching_results.tsv` and
`output/candidate_pairs.tsv` from the same scored-pair universe (matches ⊆
candidates by construction), is resumable, and refuses to reuse intermediate
results built with different settings. 160+ unit/end-to-end tests
(`python -m pytest -q`), including the organizers' validator on a synthetic run.

### B. Additional Results

| Blocking change | Recall @50 | Ceiling |
|---|---|---|
| baseline | 0.9295 | 0.9734 |
| + leading zeros | 0.9345 | 0.9758 |
| + rarest tokens | 0.9414 | 0.9789 |
| + glued names | 0.9429 | 0.9790 |

| Decision rule (held-out half) | Macro F0.5 |
|---|---|
| threshold | 0.9531 |
| **threshold + one owner** | **0.9532** |
| + rescue | 0.9528 |
| expected-F0.5 | 0.9515 |

**Compliance:** no external data, APIs, geocoders or gazetteers; LightGBM
(MIT) only, no pretrained weights; word counts on test files are unsupervised
statistics (confirmed allowed by the organizers).
