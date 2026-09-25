"""Reading the challenge TSVs and writing the two submission files.

Three ideas drive this file:

1. Read strictly.  Pandas is helpful by default, and that is the danger: it
   will happily turn the text "NA" into a missing value and the id "01" into
   the number 1.  We switch all of that off so what we read is what is on disk.

2. Read cheaply.  These files are big (train S2 is ~490 MB, 5.0M rows) and the
   machine has ~15 GB of RAM.  So we read only the columns we need, and we
   store the id column as a plain string once rather than copying it around.

3. Write exactly.  A submission is rejected on formatting alone, so the writer
   enforces the contract itself instead of trusting the caller.
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Sequence

import pandas as pd

SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
GT_COLUMNS = ["source1_entity_id", "matched_entity_ids"]

# Shared read options.  Every one of these is here for a reason -- see read_source.
_READ_KW = dict(
    sep="\t",
    dtype=str,
    keep_default_na=False,
    na_filter=False,
    encoding="utf-8-sig",
    quoting=csv.QUOTE_NONE,
    on_bad_lines="error",
)


def _read_tsv(path: Path, columns: Sequence[str], usecols: Sequence[str] | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=list(usecols) if usecols else None, **_READ_KW)
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}; found {list(df.columns)}")
    return df


def read_source(
    path: str | Path,
    expected_prefix: str | None = None,
    usecols: Sequence[str] | None = None,
    check_duplicates: bool = True,
) -> pd.DataFrame:
    """Read one *_source{1,2,3}.tsv file.

    The keyword arguments matter more than they look:
      sep="\\t"             tabs only; without it pandas returns ONE column
      dtype=str             ids stay exact strings, never numbers
      keep_default_na=False missing cells become "" and the text "NA"/"null"
                            stays the literal text it is on disk
      na_filter=False       belt and braces for the same thing
      encoding="utf-8-sig"  eats a leading byte-order-mark if the file has one
      quoting=QUOTE_NONE    a stray " inside an address must not swallow the row
      on_bad_lines="error"  never silently drop a malformed row

    usecols lets a caller load just ["entity_id", "business_name"] when that is
    all a stage needs, which on a 500 MB file is a real memory saving.
    """
    path = Path(path)
    df = _read_tsv(path, SOURCE_COLUMNS if usecols is None else usecols, usecols)

    # Strip only outer whitespace from ids -- never from name/address, because
    # the inner spacing is real signal the normalizer should decide about.
    df["entity_id"] = df["entity_id"].str.strip()

    if check_duplicates:
        n_dup = int(df["entity_id"].duplicated().sum())
        if n_dup:
            examples = df["entity_id"][df["entity_id"].duplicated()].head(5).tolist()
            raise ValueError(f"{path.name}: {n_dup} duplicate entity_id values, e.g. {examples}")

    if expected_prefix is not None:
        bad = ~df["entity_id"].str.startswith(expected_prefix)
        n_bad = int(bad.sum())
        if n_bad:
            raise ValueError(
                f"{path.name}: {n_bad} ids do not start with {expected_prefix!r}, "
                f"e.g. {df.loc[bad, 'entity_id'].head(5).tolist()}"
            )

    return df.reset_index(drop=True)


def iter_source(path: str | Path, chunksize: int = 500_000) -> Iterator[pd.DataFrame]:
    """Stream a source file in chunks, for passes that never need it all at once
    (counting, hashing, building an inverted index incrementally)."""
    for chunk in pd.read_csv(path, chunksize=chunksize, **_READ_KW):
        chunk["entity_id"] = chunk["entity_id"].str.strip()
        yield chunk.reset_index(drop=True)


def read_ground_truth(path: str | Path) -> Dict[str, List[str]]:
    """Read train_ground_truth.tsv into {s1_id: [matched ids]}.

    An empty second field is a real, meaningful value: that S1 record is a
    singleton with zero matches.  We keep it as an empty list rather than
    dropping the row -- dropping it would silently delete the rows where
    predicting nothing earns a full 1.0.
    """
    path = Path(path)
    df = _read_tsv(path, GT_COLUMNS, GT_COLUMNS)

    out: Dict[str, List[str]] = {}
    for s1, raw in zip(df["source1_entity_id"].to_numpy(), df["matched_entity_ids"].to_numpy()):
        s1 = s1.strip()
        if s1 in out:
            raise ValueError(f"{path.name}: duplicate source1_entity_id {s1!r}")
        # "" -> [] for singletons; strip guards against ", " style separators
        out[s1] = [tok for tok in (t.strip() for t in raw.split(",")) if tok]
    return out


@dataclass
class Dataset:
    """One split (train or test) held together in a single object."""

    s1: pd.DataFrame
    s2: pd.DataFrame
    s3: pd.DataFrame
    truth: Dict[str, List[str]] | None = None

    @property
    def s1_ids(self) -> List[str]:
        return self.s1["entity_id"].tolist()

    @property
    def target_ids(self) -> set:
        """Every id we are allowed to output: the S2 and S3 records."""
        return set(self.s2["entity_id"]) | set(self.s3["entity_id"])

    @property
    def total_pairs(self) -> int:
        """N1 * (N2 + N3) -- the brute-force comparison count we must avoid.
        For this dataset that is ~1.7e13, which is why blocking is the whole
        game rather than an optimisation."""
        return len(self.s1) * (len(self.s2) + len(self.s3))


def load_split(
    data_dir: str | Path,
    split: str,
    usecols: Sequence[str] | None = None,
    with_truth: bool = True,
) -> Dataset:
    """Load dataset/train or dataset/test in one call.

    split is "train" or "test"; files are named <split>_source1.tsv etc.
    """
    d = Path(data_dir) / split
    ds = Dataset(
        s1=read_source(d / f"{split}_source1.tsv", "S1-", usecols),
        s2=read_source(d / f"{split}_source2.tsv", "S2-", usecols),
        s3=read_source(d / f"{split}_source3.tsv", "S3-", usecols),
    )
    gt = d / f"{split}_ground_truth.tsv"
    if with_truth and gt.exists():
        ds.truth = read_ground_truth(gt)
    return ds


def check_truth_consistency(ds: Dataset) -> List[str]:
    """Report label problems instead of quietly patching them.

    We return the problems rather than raising, because a real dataset may have
    one or two oddities that are worth looking at, not worth crashing over.
    Silently "fixing" contradictory labels is how you end up training on your
    own assumptions instead of the organizers' data.
    """
    problems: List[str] = []
    if ds.truth is None:
        return ["no ground truth loaded"]

    s1_ids = set(ds.s1_ids)
    labelled = set(ds.truth)
    if labelled - s1_ids:
        problems.append(f"ground truth mentions {len(labelled - s1_ids)} unknown S1 ids")
    if s1_ids - labelled:
        problems.append(f"{len(s1_ids - labelled)} S1 records have NO ground-truth row")

    targets = ds.target_ids
    dangling = {i for ids in ds.truth.values() for i in ids} - targets
    if dangling:
        problems.append(
            f"{len(dangling)} matched ids do not exist in S2/S3, e.g. {sorted(dangling)[:5]}"
        )

    # Does any S2/S3 record belong to more than one S1?  The task never promises
    # exclusivity, so we only report it -- it is a fact to know, not an error to
    # fix.  If it is zero, that is a strong constraint we can exploit later.
    owner: Dict[str, str] = {}
    shared = 0
    dup_lists = 0
    for s1, ids in ds.truth.items():
        if len(ids) != len(set(ids)):
            dup_lists += 1
        for i in set(ids):
            if i in owner:
                shared += 1
            else:
                owner[i] = s1
    if dup_lists:
        problems.append(f"{dup_lists} S1 rows repeat an id inside their own match list")
    if shared:
        problems.append(f"{shared} S2/S3 ids are claimed by more than one S1 record")

    return problems


# --------------------------------------------------------------------------
# writing
# --------------------------------------------------------------------------

def write_id_list_tsv(
    path: str | Path,
    s1_ids: Sequence[str],
    ids_by_s1: Mapping[str, Iterable[str]],
    column: str,
    allowed_ids: set | None = None,
) -> None:
    """Write matching_results.tsv or candidate_pairs.tsv.

    The contract, enforced here rather than hoped for:
      * exactly one row per S1 id, in the original file order
      * a real trailing tab when the list is empty (never "[]", "nan", "None")
      * ids sorted and de-duplicated, so two runs give byte-identical output
      * only S2-/S3- ids, and (when allowed_ids is given) only ids that exist
      * no tab, comma or newline hiding inside an id

    quoting=QUOTE_NONE with no escapechar makes csv raise rather than silently
    add quote marks -- we want the crash here, not a rejected submission.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if len(set(s1_ids)) != len(s1_ids):
        raise ValueError("duplicate S1 ids passed to the writer")

    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n", quoting=csv.QUOTE_NONE)
        w.writerow(["source1_entity_id", column])
        for sid in s1_ids:
            ids = sorted(set(ids_by_s1.get(sid, ())))
            for i in ids:
                if not (i.startswith("S2-") or i.startswith("S3-")):
                    raise ValueError(f"{sid}: {i!r} is not an S2-/S3- id")
                if any(ch in i for ch in ",\t\n\r"):
                    raise ValueError(f"{sid}: id {i!r} contains a delimiter")
                if allowed_ids is not None and i not in allowed_ids:
                    raise ValueError(f"{sid}: id {i!r} does not exist in this split")
            w.writerow([sid, ",".join(ids)])


def read_id_list_tsv(path: str | Path) -> Dict[str, List[str]]:
    """Read one of our own output files back -- used to prove a round trip."""
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE))
    if not rows:
        raise ValueError(f"{path}: file is empty")
    if rows[0][0] != "source1_entity_id":
        raise ValueError(f"{path}: unexpected header {rows[0]!r}")

    out: Dict[str, List[str]] = {}
    for row in rows[1:]:
        if len(row) != 2:
            raise ValueError(f"{path}: row has {len(row)} fields, expected 2: {row!r}")
        sid, raw = row
        if sid in out:
            raise ValueError(f"{path}: duplicate row for {sid}")
        out[sid] = [t for t in raw.split(",") if t]
    return out


def file_sha256(path: str | Path) -> str:
    """Checksum, so we can prove the leaderboard file and the zipped file match."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
