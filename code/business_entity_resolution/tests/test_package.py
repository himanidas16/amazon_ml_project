import zipfile

import pytest

from business_er.package import build_zip


def _repo(tmp_path):
    code = tmp_path / "code" / "business_entity_resolution"
    (code / "src" / "business_er").mkdir(parents=True)
    (code / "src" / "business_er" / "__main__.py").write_text("x")
    (code / "README.md").write_text("r"); (code / "requirements.txt").write_text("q")
    (code / "src" / "business_er" / "__pycache__").mkdir()
    (code / "src" / "business_er" / "__pycache__" / "a.pyc").write_text("junk")
    (code / "artifacts").mkdir(); (code / "artifacts" / "big.npz").write_text("data")
    out = tmp_path / "output"; out.mkdir()
    (out / "matching_results.tsv").write_text("source1_entity_id\tmatched_entity_ids\nS1-1\t\n")
    (out / "candidate_pairs.tsv").write_text("source1_entity_id\tcandidate_entity_ids\nS1-1\t\n")
    (tmp_path / "doc.md").write_text("# doc")
    return out


def test_zip_has_required_paths_and_no_junk(tmp_path):
    out = _repo(tmp_path)
    z = build_zip("team", tmp_path, out, tmp_path / "doc.md", tmp_path / "dist", log=lambda _: None)
    names = set(zipfile.ZipFile(z).namelist())
    assert {"output/matching_results.tsv", "output/candidate_pairs.tsv", "Documentation_template.md",
            "code/business_entity_resolution/README.md"} <= names
    assert not any("__pycache__" in n or "artifacts" in n for n in names)


def test_missing_doc_refused(tmp_path):
    out = _repo(tmp_path)
    with pytest.raises(FileNotFoundError):
        build_zip("team", tmp_path, out, tmp_path / "nope.md", tmp_path / "dist", log=lambda _: None)
