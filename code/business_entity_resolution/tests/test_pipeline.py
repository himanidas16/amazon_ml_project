"""End-to-end test of the training commands on a tiny invented dataset
(software fixture only -- not competition data)."""

import json

import numpy as np
import pandas as pd

from business_er.pipeline import build_pairs, build_splits, train_and_validate, train_token_freq
from business_er.splits import load_split

HEADER = "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
WORDS = ["alpha", "bravo", "cobalt", "delta", "ember", "fjord", "garnet", "harbor", "indigo",
         "juniper", "kestrel", "lumen", "maple", "nimbus", "onyx", "pioneer", "quartz", "raven"]


def _dataset(root):
    d = root / "train"
    d.mkdir(parents=True)
    s1, s2, s3, gt = [], [], [], []
    for i in range(1, 121):
        c = "US" if i % 3 else "India"
        w1, w2 = WORDS[i % len(WORDS)], WORDS[(i * 7) % len(WORDS)]
        name, addr = f"{w1.title()} {w2.title()} Traders {i}", f"{100 + i} {w2.title()} Road, Town{i}"
        s1.append((f"S1-{i}", name, addr, c))
        m = []
        if i % 10:                                   # every 10th business is a singleton
            s2.append((f"S2-{i}", name.upper(), addr.upper(), c)); m.append(f"S2-{i}")
            if i % 2:
                s3.append((f"S3-{i}", f"{w1.title()}  {w2.title()} Trader", f"{100 + i} {w2} Rd", c))
                m.append(f"S3-{i}")
        s2.append((f"S2-{1000 + i}", f"{w2.title()} Other {i}", f"{900 + i} Elm Street", c))  # distractor
        gt.append((f"S1-{i}", ",".join(m)))
    for fn, rows in (("train_source1.tsv", s1), ("train_source2.tsv", s2), ("train_source3.tsv", s3)):
        (d / fn).write_text(HEADER + "".join("\t".join(r) + "\n" for r in rows), encoding="utf-8")
    (d / "train_ground_truth.tsv").write_text(
        "source1_entity_id\tmatched_entity_ids\n" + "".join(f"{a}\t{b}\n" for a, b in gt), encoding="utf-8")


def test_training_pipeline_end_to_end(tmp_path):
    _dataset(tmp_path / "data")
    art = tmp_path / "art"
    quiet = lambda _: None
    build_splits(tmp_path / "data", art, log=quiet)
    train_token_freq(tmp_path / "data", art, workers=1, log=quiet)
    out = build_pairs(tmp_path / "data", art, train_anchors=60, val_anchors=15, workers=1,
                      anchors_per_chunk=7, log=quiet)

    rep = json.loads((out / "build_report.json").read_text())
    assert rep["train_pos"] > 0 and rep["val_pos"] > 0

    # leak guard: no validation-fold target appears in any training pair
    sp = load_split(art / "split_5fold.npz")
    fold0 = {s: set(sp.tgt_values[s][sp.tgt_fold[s] == 0].tolist()) for s in (2, 3)}
    tr = pd.concat([pd.read_parquet(f) for f in out.glob("train_*.parquet")])
    for code in tr["target_code"]:
        assert int(code % 2_000_000_000) not in fold0[int(code // 2_000_000_000)]

    # labels agree with the ground truth, checked by id strings
    gt = pd.read_csv(tmp_path / "data/train/train_ground_truth.tsv", sep="\t", dtype=str,
                     keep_default_na=False)
    truth = {r.source1_entity_id: set(filter(None, r.matched_entity_ids.split(","))) for r in gt.itertuples()}
    for r in tr.itertuples():
        tid = f"S{int(r.target_code // 2_000_000_000)}-{int(r.target_code % 2_000_000_000)}"
        assert r.label == int(tid in truth[f"S1-{r.s1_value}"])

    report = train_and_validate(art, tag="t", log=quiet)
    assert (art / "models" / "matcher_t.lgb").exists()
    assert 0 <= report["macro_f05"] <= report["blocking_ceiling"] + 1e-9 <= 1 + 1e-9
    assert set(report["per_country"]) <= {"US", "India"}


def test_val_world_is_complete_and_self_contained(tmp_path):
    _dataset(tmp_path / "data")
    art = tmp_path / "art"
    quiet = lambda _: None
    build_splits(tmp_path / "data", art, log=quiet)
    train_token_freq(tmp_path / "data", art, workers=1, log=quiet)
    out = build_pairs(tmp_path / "data", art, out_name="world", train_anchors=0, val_world=True,
                      workers=1, log=quiet)
    assert not list(out.glob("train_*.parquet"))
    sp = load_split(art / "split_5fold.npz")
    anchors = pd.read_parquet(out / "anchors_val.parquet")
    assert len(anchors) == int((sp.s1_fold == 0).sum())                   # every fold-0 business
    fold0 = {s: set(sp.tgt_values[s][sp.tgt_fold[s] == 0].tolist()) for s in (2, 3)}
    v = pd.concat([pd.read_parquet(f) for f in out.glob("val_*.parquet")])
    for code in v["target_code"]:
        assert int(code % 2_000_000_000) in fold0[int(code // 2_000_000_000)]   # fold-0 records only


def test_build_pairs_with_wide_selection(tmp_path):
    _dataset(tmp_path / "data")
    art = tmp_path / "art"
    quiet = lambda _: None
    build_splits(tmp_path / "data", art, log=quiet)
    train_token_freq(tmp_path / "data", art, workers=1, log=quiet)
    out = build_pairs(tmp_path / "data", art, out_name="wide", train_anchors=60, val_anchors=15,
                      k=2, k_wide=20, k_formula=2, workers=1, log=quiet)
    sp = load_split(art / "split_5fold.npz")
    fold0 = {s: set(sp.tgt_values[s][sp.tgt_fold[s] == 0].tolist()) for s in (2, 3)}
    tr = pd.concat([pd.read_parquet(f) for f in out.glob("train_*.parquet")])
    for code in tr["target_code"]:                       # leak guard still holds
        assert int(code % 2_000_000_000) not in fold0[int(code // 2_000_000_000)]
    rep = json.loads((out / "build_report.json").read_text())
    assert rep["k_wide"] == 20 and rep["train_pos"] > 0
