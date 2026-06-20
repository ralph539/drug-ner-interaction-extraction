"""Re-run NER_eval_format on saved .json predictions with the fixed
examples.py, then re-invoke util/evaluator.py to regenerate .stats.

Backs up each original .out / .stats to .out.buggy / .stats.buggy once.

Usage (from bin/):
    python3 reeval.py
"""
import os, sys, json, subprocess, shutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from examples import Examples

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, "..", "results"))
EVALUATOR = os.path.abspath(os.path.join(HERE, "..", "..", "..", "util", "evaluator.py"))
DATA = os.path.abspath(os.path.join(HERE, "..", "..", "..", "data"))


def infer_split(base):
    if "-test" in base:  return "test"
    if "-devel" in base: return "devel"
    if "-train" in base: return "train"
    return None


def reformat(ex):
    return Examples.NER_eval_format(None, ex, ex["predicted"])


def main():
    jsons = sorted(f for f in os.listdir(RESULTS) if f.endswith(".json"))
    print(f"Found {len(jsons)} .json files in {RESULTS}")
    for fname in jsons:
        base = fname[:-5]
        split = infer_split(base)
        if split is None:
            print(f"  SKIP {fname}: cannot infer split")
            continue

        jpath    = os.path.join(RESULTS, fname)
        outpath  = os.path.join(RESULTS, base + ".out")
        stpath   = os.path.join(RESULTS, base + ".stats")

        for p in (outpath, stpath):
            bk = p + ".buggy"
            if os.path.exists(p) and not os.path.exists(bk):
                shutil.copy2(p, bk)

        with open(jpath) as f:
            annotated = json.load(f)
        for ex in annotated:
            ex["evaluator"] = reformat(ex)

        with open(outpath, "w") as of:
            for e in annotated:
                if e["evaluator"]:
                    of.write("\n".join(e["evaluator"]))
                    of.write("\n")

        xml = os.path.join(DATA, split + ".xml")
        r = subprocess.run(
            ["python3", EVALUATOR, "NER", xml, outpath, stpath],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            print(f"  FAIL {fname}: {r.stderr.strip()[:200]}")
        else:
            print(f"  OK   {fname}  (split={split})")


if __name__ == "__main__":
    main()
