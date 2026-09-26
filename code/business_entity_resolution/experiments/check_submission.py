"""Low-memory submission check (same rules as the organizers' validator).

Streams both files side by side, so it runs on a laptop even while another job
holds most of the memory.  Checks: headers; exactly one row per test S1 id, in
any order; no duplicate S1 rows; ids are S2-/S3- and exist in the test files;
no duplicate id inside a list; every matched id is among that row's candidates.

    python3 check_submission.py <output_dir> <test_dir>
"""
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

out, test = Path(sys.argv[1]), Path(sys.argv[2])
B = 2_000_000_000
rd = lambda f, c: pd.read_csv(f, sep="\t", usecols=[c], dtype=str, keep_default_na=False,
                              quoting=csv.QUOTE_NONE)[c]
s1 = set(rd(test / "test_source1.tsv", "entity_id"))
valid = np.sort(np.concatenate([s * B + rd(test / f"test_source{s}.tsv", "entity_id").str.slice(3)
                                .astype(np.int64).to_numpy() for s in (2, 3)]))
problems, seen, n_match, n_cand, buf = [], set(), 0, 0, []


def flush():
    global buf
    if buf:
        c = np.asarray(buf, np.int64)
        p = np.minimum(np.searchsorted(valid, c), len(valid) - 1)
        bad = int((valid[p] != c).sum())
        if bad:
            problems.append(f"{bad} ids not in the test set")
        buf = []


def ids(field):
    if not field:
        return []
    lst = field.split(",")
    for i in lst:
        if i[:3] not in ("S2-", "S3-") or not i[3:].isdigit():
            raise ValueError(f"bad id {i!r}")
    return lst


with open(out / "matching_results.tsv", encoding="utf-8") as fm, \
        open(out / "candidate_pairs.tsv", encoding="utf-8") as fc:
    if fm.readline().rstrip("\n") != "source1_entity_id\tmatched_entity_ids":
        problems.append("bad matching header")
    if fc.readline().rstrip("\n") != "source1_entity_id\tcandidate_entity_ids":
        problems.append("bad candidate header")
    for lm, lc in zip(fm, fc):
        sm, mm = lm.rstrip("\n").split("\t")
        sc, cc = lc.rstrip("\n").split("\t")
        if sm != sc:
            problems.append(f"row order differs: {sm} vs {sc}"); break
        if sm in seen:
            problems.append(f"duplicate row {sm}")
        seen.add(sm)
        m, c = ids(mm), ids(cc)
        if len(m) != len(set(m)) or len(c) != len(set(c)):
            problems.append(f"duplicate id inside a list for {sm}")
        if not set(m) <= set(c):
            problems.append(f"{sm}: matched ids outside its candidates")
        n_match += len(m); n_cand += len(c)
        buf.extend(int(i[1]) * B + int(i[3:]) for i in c)
        if len(buf) > 5_000_000:
            flush()
    flush()
if seen != s1:
    problems.append(f"S1 coverage: {len(s1 - seen)} missing, {len(seen - s1)} unknown")
print(f"{len(seen):,} rows; {n_match:,} matched ids; {n_cand:,} candidate ids")
print("PASS" if not problems else "FAIL:\n  " + "\n  ".join(problems[:20]))
sys.exit(0 if not problems else 1)
