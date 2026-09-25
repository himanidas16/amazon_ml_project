"""Command-line entry point:  python -m business_er <command> [options]

    predict   blocking -> features -> model -> decision -> the two TSV files

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

    args = ap.parse_args(argv)
    if args.cmd == "predict":
        from .predict import run
        summary = run(
            args.data_dir, args.split, args.model, args.freq, args.output_dir, args.work_dir,
            cap=args.cap, k=args.k, t=args.threshold, use_one_owner=not args.no_one_owner,
            workers=args.workers, predict_threads=args.predict_threads,
            log=lambda m: print(m, flush=True),
        )
        print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
