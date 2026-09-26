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


def test_stale_work_dir_is_refused(tmp_path):
    import json
    import pytest
    from business_er.predict import check_manifest

    w = tmp_path / "work"; w.mkdir()
    check_manifest(w, {"freq_sha256": "aaa"})                  # first run writes the manifest
    check_manifest(w, {"freq_sha256": "aaa"})                  # same settings: fine
    with pytest.raises(RuntimeError, match="freq_sha256"):
        check_manifest(w, {"freq_sha256": "bbb"})              # changed word table: refused
    legacy = tmp_path / "legacy"; legacy.mkdir()
    (legacy / "test_US_cands.npz").write_bytes(b"x")
    with pytest.raises(RuntimeError, match="no manifest"):
        check_manifest(legacy, {"freq_sha256": "aaa"})         # old cache of unknown origin


def test_scoring_resumes_mid_country(tmp_path, monkeypatch):
    """A run killed mid-scoring must resume from its last finished chunk and
    produce the same output bytes as an uninterrupted run."""
    import business_er.predict as pred

    d = tmp_path / "data" / "test"
    d.mkdir(parents=True)
    rows1 = [(f"S1-{i}", f"Nematech Shop {i}", f"{i} Autrey Road, Austin", "US") for i in range(1, 41)]
    rows2 = [(f"S2-{i}", f"Nematech Shop {i}", f"{i} Autrey Rd, Austin", "US") for i in range(1, 41)]
    _write(d / "test_source1.tsv", rows1); _write(d / "test_source2.tsv", rows2)
    _write(d / "test_source3.tsv", [])
    freq = build_token_freq({2: str(d / "test_source2.tsv")}); freq.save(tmp_path / "f.npz")
    rng = np.random.default_rng(0)
    X = rng.random((2000, len(FEATURE_NAMES))).astype(np.float32)
    y = (X[:, FEATURE_NAMES.index("name_tset")] > 0.3).astype(np.int8)
    train_matcher(X, y, np.repeat(np.arange(200), 10), params={"num_leaves": 3, "min_data_in_leaf": 5},
                  num_rounds=10, early_stopping=5, log_every=0).save(tmp_path / "m.lgb")
    kw = dict(workers=1, predict_threads=1, anchors_per_chunk=5, log=lambda _: None)

    clean = tmp_path / "clean"
    run(tmp_path / "data", "test", tmp_path / "m.lgb", tmp_path / "f.npz", clean, tmp_path / "w1", **kw)

    calls = {"n": 0}
    real = pred.compute_features
    def dies_on_third(*a, **k):
        calls["n"] += 1
        if calls["n"] == 3:
            raise MemoryError("simulated kill")
        return real(*a, **k)
    monkeypatch.setattr(pred, "compute_features", dies_on_third)
    out = tmp_path / "resumed"
    import pytest
    with pytest.raises(MemoryError):
        run(tmp_path / "data", "test", tmp_path / "m.lgb", tmp_path / "f.npz", out, tmp_path / "w2", **kw)
    assert (tmp_path / "w2" / "test_US_scores.partial.done").read_text() != "0"
    monkeypatch.setattr(pred, "compute_features", real)
    run(tmp_path / "data", "test", tmp_path / "m.lgb", tmp_path / "f.npz", out, tmp_path / "w2", **kw)
    for f in ("matching_results.tsv", "candidate_pairs.tsv"):
        assert (out / f).read_bytes() == (clean / f).read_bytes()
    assert not list((tmp_path / "w2").glob("*.partial*"))


def test_stage2_path_end_to_end(tmp_path):
    import lightgbm as lgb
    import pytest
    from business_er.stage2 import STAGE2_NAMES

    d = tmp_path / "data" / "test"
    d.mkdir(parents=True)
    _write(d / "test_source1.tsv", S1); _write(d / "test_source2.tsv", S2); _write(d / "test_source3.tsv", S3)
    build_token_freq({2: str(d / "test_source2.tsv")}).save(tmp_path / "f.npz")
    rng = np.random.default_rng(0)
    X = rng.random((2000, len(FEATURE_NAMES))).astype(np.float32)
    y = (X[:, FEATURE_NAMES.index("name_tset")] > 0.3).astype(np.int8)
    train_matcher(X, y, np.repeat(np.arange(200), 10), params={"num_leaves": 3, "min_data_in_leaf": 5},
                  num_rounds=10, early_stopping=5, log_every=0).save(tmp_path / "m.lgb")
    X2 = rng.random((500, len(STAGE2_NAMES))).astype(np.float32)
    lgb.train({"objective": "binary", "verbosity": -1, "min_data_in_leaf": 5},
              lgb.Dataset(X2, label=(X2[:, 0] > 0.5).astype(int)), 5).save_model(str(tmp_path / "s2.lgb"))
    kw = dict(workers=1, predict_threads=1, log=lambda _: None)
    out, work = tmp_path / "out", tmp_path / "work"
    run(tmp_path / "data", "test", tmp_path / "m.lgb", tmp_path / "f.npz", out, work,
        stage2_path=tmp_path / "s2.lgb", t2=0.5, **kw)
    match = read_id_list_tsv(out / "matching_results.tsv")
    cand = read_id_list_tsv(out / "candidate_pairs.tsv")
    assert list(match) == [r[0] for r in S1]
    assert all(set(match[s]) <= set(cand[s]) for s in match)
    assert (work / "test_US_keep.npy").exists()
    # the same work dir can also produce the stage-1-only decision (for a
    # separate leaderboard reading) -- blocking and stage-1 scores are shared
    out1 = tmp_path / "out_stage1"
    run(tmp_path / "data", "test", tmp_path / "m.lgb", tmp_path / "f.npz", out1, work, **kw)
    assert list(read_id_list_tsv(out1 / "candidate_pairs.tsv").values()) == list(cand.values())
    # but a stage-1-only work dir cannot serve stage 2 (no stored inputs)
    work2 = tmp_path / "work2"
    run(tmp_path / "data", "test", tmp_path / "m.lgb", tmp_path / "f.npz", tmp_path / "o2", work2, **kw)
    with pytest.raises(RuntimeError, match="stores_stage2_inputs"):
        run(tmp_path / "data", "test", tmp_path / "m.lgb", tmp_path / "f.npz", tmp_path / "o3", work2,
            stage2_path=tmp_path / "s2.lgb", t2=0.5, **kw)


def test_predict_with_wide_selection(tmp_path):
    d = tmp_path / "data" / "test"
    d.mkdir(parents=True)
    _write(d / "test_source1.tsv", S1); _write(d / "test_source2.tsv", S2); _write(d / "test_source3.tsv", S3)
    build_token_freq({2: str(d / "test_source2.tsv")}).save(tmp_path / "f.npz")
    rng = np.random.default_rng(0)
    X = rng.random((2000, len(FEATURE_NAMES))).astype(np.float32)
    y = (X[:, FEATURE_NAMES.index("name_tset")] > 0.3).astype(np.int8)
    train_matcher(X, y, np.repeat(np.arange(200), 10), params={"num_leaves": 3, "min_data_in_leaf": 5},
                  num_rounds=10, early_stopping=5, log_every=0).save(tmp_path / "m.lgb")
    out = tmp_path / "out"
    run(tmp_path / "data", "test", tmp_path / "m.lgb", tmp_path / "f.npz", out, tmp_path / "w",
        k=1, k_wide=10, k_formula=1, workers=1, predict_threads=1, log=lambda _: None)
    match = read_id_list_tsv(out / "matching_results.tsv")
    cand = read_id_list_tsv(out / "candidate_pairs.tsv")
    assert list(cand) == [r[0] for r in S1]
    assert all(set(match[s]) <= set(cand[s]) for s in match)
    assert all(len(v) <= 2 * (1 + 1) for v in cand.values())   # <= (k + k_formula) per source
