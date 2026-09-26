"""Step 10: build and verify the final submission ZIP.

    <team>_submission.zip
    ├── output/matching_results.tsv
    ├── output/candidate_pairs.tsv
    ├── code/business_entity_resolution/{src/, README.md, requirements.txt, ...}
    └── Documentation_template.md

Checks before and after writing: every required path exists; the packaged
matching_results.tsv is byte-identical to the one given (the leaderboard
upload); no dataset, artifacts, caches or personal files are included.
"""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path
from typing import Iterable, List

REQUIRED = (
    "output/matching_results.tsv",
    "output/candidate_pairs.tsv",
    "code/business_entity_resolution/src/business_er/__main__.py",
    "code/business_entity_resolution/README.md",
    "code/business_entity_resolution/requirements.txt",
    "Documentation_template.md",
)
# never packaged: data, generated artifacts, caches, editor/OS junk
EXCLUDE_PARTS = {"__pycache__", ".pytest_cache", ".venv", "artifacts", "output", ".git",
                 "amazon_ml_dataset", "student_resource", "share_for_team"}
EXCLUDE_SUFFIXES = {".pyc", ".tsv", ".npz", ".npy", ".parquet", ".lgb", ".xlsx", ".zip", ".log"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _code_files(code_dir: Path) -> Iterable[Path]:
    for p in sorted(code_dir.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(code_dir)
        if set(rel.parts) & EXCLUDE_PARTS or p.suffix in EXCLUDE_SUFFIXES or p.name.startswith("."):
            continue
        yield p


def build_zip(team: str, repo_root: str | Path, output_dir: str | Path, doc_path: str | Path,
              dest_dir: str | Path, log=print) -> Path:
    root, out = Path(repo_root), Path(output_dir)
    code_dir = root / "code" / "business_entity_resolution"
    zpath = Path(dest_dir) / f"{team}_submission.zip"
    members: List[tuple] = [
        (out / "matching_results.tsv", "output/matching_results.tsv"),
        (out / "candidate_pairs.tsv", "output/candidate_pairs.tsv"),
        (Path(doc_path), "Documentation_template.md"),
    ]
    members += [(p, f"code/business_entity_resolution/{p.relative_to(code_dir).as_posix()}")
                for p in _code_files(code_dir)]
    for src, _ in members:
        if not src.is_file():
            raise FileNotFoundError(src)
    names = {arc for _, arc in members}
    missing = [r for r in REQUIRED if r not in names]
    if missing:
        raise RuntimeError(f"required paths missing: {missing}")

    zpath.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for src, arc in members:
            z.write(src, arc)

    # verify: required paths present, leaderboard file byte-identical inside
    with zipfile.ZipFile(zpath) as z:
        inside = set(z.namelist())
        assert all(r in inside for r in REQUIRED)
        h = hashlib.sha256()
        with z.open("output/matching_results.tsv") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    want = sha256(out / "matching_results.tsv")
    if h.hexdigest() != want:
        raise RuntimeError("zipped matching_results.tsv differs from the original")
    log(f"{zpath}: {len(inside)} files, {zpath.stat().st_size / 1e6:.0f} MB; "
        f"matching_results.tsv sha256 {want}")
    return zpath
