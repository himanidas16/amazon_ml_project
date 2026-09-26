"""Command-line entry point:  python -m business_er <command> [options]

    split        train/validation folds grouped by business
    train-freq   word counts over training records outside the validation fold
    build-pairs  labelled candidate pairs with features (train + validation)
    train        LightGBM matcher + competition score on validation
    token-freq   word counts over a split's files (test: used at prediction)
    predict      blocking -> features -> model -> decision -> the two TSV files

Every path is an argument; nothing depends on a developer's home directory.
"""

from __future__ import annotations

import argparse
import json
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m business_er")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("predict", help="full inference on a split, writes both submission files")
    p.add_argument("--data-dir", required=True, help="folder containing train/ and test/")
    p.add_argument("--split", default="test")
    p.add_argument("--model", required=True, help="matcher .lgb file (its .json sits next to it)")
    p.add_argument("--freq", required=True, help="token frequency table .npz")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--work-dir", required=True, help="resumable intermediate results")
    p.add_argument("--cap", type=int, default=3000)
    p.add_argument("--k", type=int, default=25)
    p.add_argument("--threshold", type=float, default=0.80)
    p.add_argument("--no-one-owner", action="store_true")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--predict-threads", type=int, default=12)
    p.add_argument("--k-wide", type=int, default=0, help="wide retrieval depth per source (0 = off)")
    p.add_argument("--k-formula", type=int, default=0, help="extra formula-selected candidates per source")
    p.add_argument("--extra-k", type=int, default=0,
                   help="top-k per source from EACH extra address channel (0 = off)")
    p.add_argument("--stage2-model", default=None, help="optional stage-2 model (competing claims)")
    p.add_argument("--stage2-threshold", type=float, default=None)

    f = sub.add_parser("token-freq", help="count word frequencies (unsupervised) over a split's source files")
    f.add_argument("--data-dir", required=True)
    f.add_argument("--split", default="test")
    f.add_argument("--output", required=True, help=".npz path")
    f.add_argument("--workers", type=int, default=6)

    sp_ = sub.add_parser("split", help="train/validation folds grouped by business")
    sp_.add_argument("--data-dir", required=True)
    sp_.add_argument("--artifacts", required=True)

    tf = sub.add_parser("train-freq", help="word counts over training records outside the validation fold")
    tf.add_argument("--data-dir", required=True)
    tf.add_argument("--artifacts", required=True)
    tf.add_argument("--workers", type=int, default=6)

    bp = sub.add_parser("build-pairs", help="labelled candidate pairs with features (train + val)")
    bp.add_argument("--data-dir", required=True)
    bp.add_argument("--artifacts", required=True)
    bp.add_argument("--out-name", default="pairs")
    bp.add_argument("--train-anchors", type=int, default=100_000)
    bp.add_argument("--val-anchors", type=int, default=20_000)
    bp.add_argument("--easy-rate", type=float, default=1.0,
                    help="<1 keeps that share of easy negatives, weighted 1/rate")
    bp.add_argument("--workers", type=int, default=6)
    bp.add_argument("--k-wide", type=int, default=0)
    bp.add_argument("--k-formula", type=int, default=0)
    bp.add_argument("--extra-k", type=int, default=0)
    bp.add_argument("--val-world", action="store_true",
                    help="all fold-0 businesses vs fold-0 records only (complete competition)")

    tr = sub.add_parser("train", help="train the matcher and score validation")
    tr.add_argument("--artifacts", required=True)
    tr.add_argument("--pairs-name", default="pairs")
    tr.add_argument("--tag", default="final")
    tr.add_argument("--threshold", type=float, default=0.80)
    tr.add_argument("--max-train-anchors", type=int, default=0,
                    help="train on a random subset of this many businesses (learning curve)")

    s2 = sub.add_parser("train-stage2", help="train the competing-claims model on a complete world")
    s2.add_argument("--world", required=True, help="build-pairs --val-world output folder")
    s2.add_argument("--stage1-model", required=True)
    s2.add_argument("--out", required=True, help="path stem; writes .lgb and .json")

    pk = sub.add_parser("package", help="build and verify the final submission ZIP")
    pk.add_argument("--team-name", required=True)
    pk.add_argument("--repo-root", required=True)
    pk.add_argument("--output-dir", required=True, help="folder with the two TSV files")
    pk.add_argument("--doc", required=True, help="filled-in Documentation_template.md")
    pk.add_argument("--dest", required=True)

    args = ap.parse_args(argv)
    if args.cmd == "train-stage2":
        from .stage2 import train_stage2
        train_stage2(args.world, args.stage1_model, args.out, log=lambda m: print(m, flush=True))
        return 0
    if args.cmd == "package":
        from .package import build_zip
        build_zip(args.team_name, args.repo_root, args.output_dir, args.doc, args.dest,
                  log=lambda m: print(m, flush=True))
        return 0
    say = lambda m: print(m, flush=True)
    if args.cmd == "split":
        from .pipeline import build_splits
        build_splits(args.data_dir, args.artifacts, log=say)
        return 0
    if args.cmd == "train-freq":
        from .pipeline import train_token_freq
        train_token_freq(args.data_dir, args.artifacts, workers=args.workers, log=say)
        return 0
    if args.cmd == "build-pairs":
        from .pipeline import build_pairs
        build_pairs(args.data_dir, args.artifacts, out_name=args.out_name,
                    train_anchors=args.train_anchors, val_anchors=args.val_anchors,
                    easy_rate=args.easy_rate, workers=args.workers, val_world=args.val_world,
                    k_wide=args.k_wide, k_formula=args.k_formula, extra_k=args.extra_k,
                    log=say)
        return 0
    if args.cmd == "train":
        from .pipeline import train_and_validate
        train_and_validate(args.artifacts, pairs_name=args.pairs_name, tag=args.tag,
                           t=args.threshold, max_train_anchors=args.max_train_anchors, log=say)
        return 0
    if args.cmd == "token-freq":
        # Word counts over ALL records of the split, labels unused.  The
        # organizers confirmed unsupervised statistics on the test files are
        # allowed; this gives unseen countries (France) their own word rarity.
        from pathlib import Path
        from .retrieve import build_token_freq
        d = Path(args.data_dir) / args.split
        freq = build_token_freq({s: str(d / f"{args.split}_source{s}.tsv") for s in (1, 2, 3)},
                                workers=args.workers, log=lambda m: print(m, flush=True))
        freq.save(args.output)
        print(f"saved {len(freq.hashes):,} tokens, countries {sorted(freq.countries)} -> {args.output}")
        return 0
    if args.cmd == "predict":
        from .predict import run
        summary = run(
            args.data_dir, args.split, args.model, args.freq, args.output_dir, args.work_dir,
            cap=args.cap, k=args.k, t=args.threshold, use_one_owner=not args.no_one_owner,
            workers=args.workers, predict_threads=args.predict_threads,
            stage2_path=args.stage2_model, t2=args.stage2_threshold,
            k_wide=args.k_wide, k_formula=args.k_formula, extra_k=args.extra_k,
            log=lambda m: print(m, flush=True),
        )
        print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
