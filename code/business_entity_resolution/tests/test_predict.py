"""End-to-end test of the inference pipeline on a tiny invented dataset.

Records are made up for software testing only.  The model is trained on
random numbers -- the point is the plumbing and the output contract, not
the predictions.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np

from business_er.features import FEATURE_NAMES
from business_er.io import read_id_list_tsv
from business_er.predict import run
from business_er.retrieve import build_token_freq
from business_er.train import train_matcher

HEADER = "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
S1 = [("S1-11", "Nematech Solutions LLC", "2046 Autrey Road, Austin TX", "US"),
      ("S1-12", "Boulangerie Lemaire SARL", "5 Rue Victor Hugo, 59000 Lille", "France"),
      ("S1-13", "Zzyzx Quux", "999999 Nowhere", "US"),                      # no candidates
      ("S1-14", "Shree Best Business Pvt Ltd", "406 Manas Nagar, Lucknow", "India")]
S2 = [("S2-10", "Nematech Solutions", "2046 Autrey Rd, Austin", "US"),
      ("S2-11", "Nematech Solutions", "2046 Autrey Rd, Austin", "US"),     # identical twin
      ("S2-13", "Boulangerie Lemaire", "5 R. Victor Hugo, Lille", "France"),
      ("S2-15", "श्री बेस्ट बिजनेस प्राइवेट लिमिटेड", "406, MANAS NAGAR, LUCKNOW", "India")]
S3 = [("S3-10", "Nematek Solutions", "2048 Autrey Rd", "US"),
      ("S3-20", "Boulangerie Lemaire", "5 Rue Victor Hugo Lille", "France")]

VALIDATOR = Path(__file__).resolve().parents[3] / "amazon_ml_dataset/student_resource/utils/validate_submission.py"


def _write(path, rows):
    path.write_text(HEADER + "".join("\t".join(r) + "\n" for r in rows), encoding="utf-8")


def test_end_to_end_outputs_obey_the_contract(tmp_path):
    d = tmp_path / "data" / "test"
    d.mkdir(parents=True)
    _write(d / "test_source1.tsv", S1); _write(d / "test_source2.tsv", S2); _write(d / "test_source3.tsv", S3)

    freq = build_token_freq({2: str(d / "test_source2.tsv")})
    freq.save(tmp_path / "freq.npz")
    rng = np.random.default_rng(0)
    X = rng.random((2000, len(FEATURE_NAMES))).astype(np.float32)
    y = (X[:, FEATURE_NAMES.index("name_tset")] > 0.3).astype(np.int8)
    m = train_matcher(X, y, np.repeat(np.arange(200), 10),
                      params={"num_leaves": 3, "min_data_in_leaf": 5}, num_rounds=10,
                      early_stopping=5, log_every=0)
    m.save(tmp_path / "m.lgb")

    out, work = tmp_path / "output", tmp_path / "work"
    summary = run(tmp_path / "data", "test", tmp_path / "m.lgb", tmp_path / "freq.npz", out, work,
                  workers=1, predict_threads=1, log=lambda _: None)
    assert set(summary) == {"US", "France", "India"}           # France handled, no allowlist

    match = read_id_list_tsv(out / "matching_results.tsv")
    cand = read_id_list_tsv(out / "candidate_pairs.tsv")
    ids = [r[0] for r in S1]
    targets = {r[0] for r in S2 + S3}
    for f in (match, cand):
        assert list(f) == ids                                   # every S1 once, file order
        for lst in f.values():
            assert len(lst) == len(set(lst)) and set(lst) <= targets
    for sid in ids:
        assert set(match[sid]) <= set(cand[sid])               # matches within candidates
    assert cand["S1-13"] == [] and match["S1-13"] == []        # empty row still written
    assert {"S2-10", "S2-11"} <= set(cand["S1-11"])            # twins both considered
    assert {"S2-13", "S3-20"} <= set(cand["S1-12"])            # France retrieval works
    assert "S2-15" in cand["S1-14"]                            # Devanagari retrieval works

    # a rerun resumes from the saved stages and gives identical bytes
    first = (out / "matching_results.tsv").read_bytes()
    run(tmp_path / "data", "test", tmp_path / "m.lgb", tmp_path / "freq.npz", out, work,
        workers=1, predict_threads=1, log=lambda _: None)
    assert (out / "matching_results.tsv").read_bytes() == first

    if VALIDATOR.exists():                                      # the organizers' own checker
        r = subprocess.run([sys.executable, str(VALIDATOR), "--matching", str(out / "matching_results.tsv"),
                            "--candidate", str(out / "candidate_pairs.tsv"), "--test-dir", str(d)],
                           capture_output=True, text=True)
        assert r.returncode == 0 and "PASS" in r.stdout, r.stdout + r.stderr
