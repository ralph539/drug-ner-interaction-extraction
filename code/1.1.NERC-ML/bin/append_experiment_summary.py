#!/usr/bin/env python3

import argparse
import re
from pathlib import Path


def extract_f1(prefix: str, text: str) -> str:
    match = re.search(rf"^{re.escape(prefix)}\s+.*?(\d+\.\d+%)\s*$", text, re.M)
    return match.group(1) if match else "NA"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("label", help="Base label without extension, e.g. devel-MEM-C1-lbfgs-2000")
    parser.add_argument("model", help="CRF|MEM|SVM")
    parser.add_argument(
        "params",
        help='Human-readable params string, e.g. "C=1 solver=lbfgs max_iter=2000"',
    )
    parser.add_argument(
        "--stats",
        default=None,
        help="Path to .stats file. Defaults to results_c_change/<label>.stats",
    )
    parser.add_argument(
        "--summary",
        default="results_c_change/experiment-summary.txt",
        help="Path to experiment summary file",
    )
    args = parser.parse_args()

    stats_path = Path(args.stats) if args.stats else Path("results_c_change") / f"{args.label}.stats"
    summary_path = Path(args.summary)

    stats_text = stats_path.read_text(encoding="utf-8", errors="replace")
    f1_mavg = extract_f1("m.avg", stats_text)
    f1_mavg_noclass = extract_f1("m.avg(no class)", stats_text)

    if summary_path.exists():
        summary_text = summary_path.read_text(encoding="utf-8", errors="replace")
    else:
        summary_text = (
            "Experiment summary (devel)\n\n"
            "Run\tModel\tParams\tm.avg F1\tm.avg(no class) F1\n"
            "---\t-----\t------\t--------\t-----------------\n"
        )

    if args.label in summary_text:
        return

    summary_text += (
        f"{args.label}.stats\t{args.model}\t{args.params}\t{f1_mavg}\t{f1_mavg_noclass}\n"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(summary_text, encoding="utf-8")


if __name__ == "__main__":
    main()
