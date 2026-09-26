"""The command line must load and every subcommand must parse its options
(a duplicated keyword in __main__.py once broke every command)."""

import pytest

from business_er.__main__ import main


@pytest.mark.parametrize("cmd", ["predict", "token-freq", "split", "train-freq", "build-pairs",
                                 "train", "train-stage2", "package"])
def test_subcommand_help(cmd, capsys):
    with pytest.raises(SystemExit) as e:
        main([cmd, "--help"])
    assert e.value.code == 0
    assert "usage" in capsys.readouterr().out


def test_predict_accepts_all_selection_options(capsys):
    with pytest.raises(SystemExit):
        main(["predict", "--help"])
    out = capsys.readouterr().out
    for opt in ("--k-wide", "--k-formula", "--extra-k", "--stage2-model", "--stage2-threshold"):
        assert opt in out
