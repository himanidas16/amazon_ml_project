"""Wide retrieval + fixed-formula selection (tiny invented records)."""

import numpy as np

from business_er.features import prepare
from business_er.retrieve import build_index, compute_keys, generate_candidates
from business_er.select import prepare_pool, select_candidates, slice_keys

HEADER = "entity_id\tbusiness_name\tbusiness_address\tcountry\n"


def _write(path, rows):
    path.write_text(HEADER + "".join("\t".join(r) + "\n" for r in rows), encoding="utf-8")
    return str(path)


def test_selection_is_blocking_top_union_formula_top(tmp_path):
    from business_er.select import formula_score
    rows = [(f"S2-{i}", f"Zorbex Mart {i % 6}", f"{i % 9} Cabin Lane, Pune", "India") for i in range(1, 60)]
    rows.append(("S2-99", "Irilyragild", "WS 2, Cabin 1, SV Complx, Pune", "India"))
    paths = {2: _write(tmp_path / "s2.tsv", rows)}
    idx = build_index(paths)
    anchors = [("Zorbex Hospital", "WS 2, Cabin 1, SV Complex, Pune", "India"),
               ("Zorbex Mart 3", "3 Cabin Lane, Pune", "India")]
    n, a, c = zip(*anchors)
    ak = compute_keys(list(n), list(a), list(c))
    A = prepare(list(n), list(a), countries=list(c))
    T = prepare_pool(paths, idx, workers=1)
    assert len(T) == idx.n_targets
    wide = generate_candidates(idx, ak, 2, k=50)
    fs = formula_score(A["name"], A["addr"], T["name"], T["addr"], wide.anchor.astype(np.int64),
                       wide.target, wide.score, threads=1)
    sel, counts = select_candidates(idx, ak, 2, A["name"], A["addr"], T["name"], T["addr"],
                                    k_keep=3, k_wide=50, k_formula=2, workers=1, threads=1)
    for anc in (0, 1):
        m = wide.anchor == anc
        top_block = set(wide.target[m & (wide.rank < 3)].tolist())
        order = np.lexsort((wide.target[m], -fs[m]))
        top_formula = set(wide.target[m][order[:2]].tolist())
        got = set(sel.target[sel.anchor == anc].tolist())
        assert got == top_block | top_formula
    assert len(counts["t_addr_cnt"]) == len(sel)


def test_ranges_give_identical_selection(tmp_path):
    rows = [(f"S2-{i}", f"Shop {i % 7} Zorbex", f"{i % 5} Main Road, Pune", "India") for i in range(1, 60)]
    paths = {2: _write(tmp_path / "s2.tsv", rows)}
    idx = build_index(paths)
    anchors = [(f"Shop {i} Zorbex", f"{i} Main Road, Pune", "India") for i in range(9)]
    n, a, c = zip(*anchors)
    ak = compute_keys(list(n), list(a), list(c))
    A = prepare(list(n), list(a), countries=list(c)); T = prepare_pool(paths, idx, workers=1)
    kw = dict(k_keep=2, k_wide=10, k_formula=2, workers=1, threads=1)
    one, _ = select_candidates(idx, ak, 9, A["name"], A["addr"], T["name"], T["addr"],
                               anchors_per_range=100, **kw)
    many, _ = select_candidates(idx, ak, 9, A["name"], A["addr"], T["name"], T["addr"],
                                anchors_per_range=2, **kw)
    for f in ("anchor", "target", "rank"):
        assert np.array_equal(getattr(one, f), getattr(many, f))
    s = slice_keys(ak, 3, 5)
    assert all((r >= 0).all() and (r < 2).all() for _, r in s.values())


def test_index_text_equals_prepared_pool(tmp_path):
    rows = [("S2-1", "श्री बेस्ट बिजनेस", "406, MANAS NAGAR, उत्तर प्रदेश", "India"),
            ("S2-2", "Boulangerie Lemaire", "5 R. Victor Hugo, Lille", "France"),
            ("S2-3", "", "", "US")]
    paths = {2: _write(tmp_path / "s2.tsv", rows)}
    with_text = build_index(paths, keep_text=True)
    plain = build_index(paths)
    a = prepare_pool(paths, with_text, workers=1)          # comes from the index
    b = prepare_pool(paths, plain, workers=1)              # separate pass
    assert a["name"].tolist() == b["name"].tolist() and a["addr"].tolist() == b["addr"].tolist()
    assert "rue" in a["addr"].iloc[1].split()             # France table applied


# ---- optional extra address channels ------------------------------------------

def test_extra_keys_only_when_enabled():
    from business_er.retrieve import EXTRA_CHANNELS, record_keys, keys_for_rows
    plain = keys_for_rows(["Eye Group"], ["403 8th Street, Buckeye, AZ"], ["US"])
    extra = keys_for_rows(["Eye Group"], ["403 8th Street, Buckeye, AZ"], ["US"], extra=True)
    assert not any(ch in plain for ch in EXTRA_CHANNELS)
    for ch in EXTRA_CHANNELS:
        assert ch in extra
    raw = extra["_raw"][0]
    assert any("|nt|40|" in k for k in raw["num_trunc"])          # 403 -> "40" + rare word
    assert raw["addr_nonum"] and not any(c.isdigit() for c in raw["addr_nonum"][0].split("|an|")[1])


def test_extra_selection_is_normal_selection_plus_channel_tops(tmp_path):
    from business_er.retrieve import EXTRA_CHANNELS
    rows = [(f"S2-{i}", f"Eye Group {i % 5}", f"{400 + i} Kerrigan Buckeye Lane, AZ", "US") for i in range(1, 40)]
    rows += [(f"S2-{100 + i}", f"Shop {i}", f"{i} Kerrigan Buckeye Lane", "US") for i in range(1, 20)]
    paths = {2: _write(tmp_path / "s2.tsv", rows)}
    anchors = [("Eye Group", "403 Kerrigan Buckeye Lane, AZ", "US"), ("Shop 7", "7 Kerrigan Lane", "US")]
    n, a, c = zip(*anchors)
    A = prepare(list(n), list(a), countries=list(c))
    idx = build_index(paths, extra=True, keep_text=True)
    ak = compute_keys(list(n), list(a), list(c), extra=True)
    T = prepare_pool(paths, idx)
    kw = dict(k_keep=2, k_wide=6, k_formula=1, workers=1, threads=1)
    base, _ = select_candidates(idx, ak, 2, A["name"], A["addr"], T["name"], T["addr"], **kw)
    more, cnt = select_candidates(idx, ak, 2, A["name"], A["addr"], T["name"], T["addr"], extra_k=2, **kw)
    pairs = lambda cand: set(zip(cand.anchor.tolist(), cand.target.tolist()))
    want = pairs(base)
    for ch in EXTRA_CHANNELS:
        want |= pairs(generate_candidates(idx, ak, 2, k=2, channels=[ch]))
    assert pairs(more) == want
    assert len(pairs(more)) == len(more)                  # no duplicate pairs
    assert len(cnt["t_addr_cnt"]) == len(more)
    assert (np.diff(more.anchor) >= 0).all()             # still grouped by business
