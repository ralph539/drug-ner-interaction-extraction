#!/usr/bin/env python3
# [MOD-2.1] tune_thresholds.py — per-class threshold optimisation
#
# Idea: rather than a single global stage-1 threshold T, use a per-class
# threshold T_c on the joint probability P(c) = P(positive) * P(c|positive).
# We predict class c iff P(c) >= T_c, and among classes that pass we
# pick the one with highest P(c). If none passes, predict null.
#
# Tuning: for each class c we sweep T_c on devel, optimising per-class F1.
# (This is a coordinate-descent / independent-tuning approximation; the
# true joint optimum needs nested search, but per-class is fast and works.)

import sys, os, pickle, argparse, json
from collections import defaultdict
import numpy as np
import scipy
from xml.dom.minidom import parse
from dataset import Dataset

POSITIVE = ("advise", "effect", "int", "mechanism")

def build_probs(model_path, feat_path):
    """Returns list of (sid, e1, e2, gold, P(positive), {c: P(c|positive)})"""
    with open(model_path, "rb") as f:
        m = pickle.load(f)
    stage1, stage2, fidx = m["stage1"], m["stage2"], m["fidx"]
    ds = Dataset(feat_path)
    s1_classes = list(stage1.classes_)
    pos_idx = s1_classes.index("positive")
    s2_classes = list(stage2.classes_)
    rows = []
    for ex in ds.instances():
        ri, ci, da = [], [], []
        for feat in ex["features"]:
            if feat in fidx:
                ri.append(0); ci.append(fidx[feat]); da.append(1)
        X = scipy.sparse.csr_matrix((da, (ri, ci)), shape=(1, len(fidx)))
        p_pos = stage1.predict_proba(X)[0][pos_idx]
        p_c = stage2.predict_proba(X)[0]
        cprobs = {c: float(p_pos * p_c[s2_classes.index(c)]) for c in POSITIVE}
        rows.append((ex["sid"], ex["e1"], ex["e2"], ex["label"], float(p_pos), cprobs))
    return rows

def load_gold_xml(xmlfile):
    g = {}
    tree = parse(xmlfile)
    for s in tree.getElementsByTagName("sentence"):
        sid = s.attributes["id"].value
        for p in s.getElementsByTagName("pair"):
            e1 = p.attributes["e1"].value
            e2 = p.attributes["e2"].value
            lab = p.attributes["type"].value if p.attributes["ddi"].value == "true" else "null"
            g[(sid, e1, e2)] = lab
    return g

def predict_with_thresholds(rows, thresholds):
    """Apply per-class thresholds, return per-row predicted label."""
    preds = []
    for sid, e1, e2, gold, p_pos, cprobs in rows:
        # candidate classes whose joint prob clears their threshold
        candidates = [(c, cprobs[c]) for c in POSITIVE if cprobs[c] >= thresholds[c]]
        if candidates:
            label = max(candidates, key=lambda x: x[1])[0]
        else:
            label = "null"
        preds.append((sid, e1, e2, label))
    return preds

def eval_preds(rows, preds, gold):
    """Compute per-class TP/FP/FN and macro/micro F1."""
    by_class = {c: {"tp": 0, "fp": 0, "fn": 0} for c in POSITIVE}
    for (sid, e1, e2, p_lab), (gs, ge1, ge2, glab, *_) in zip(preds, rows):
        if p_lab == "null" and glab == "null": continue
        if p_lab == glab:
            if p_lab != "null":
                by_class[p_lab]["tp"] += 1
        else:
            if p_lab != "null": by_class[p_lab]["fp"] += 1
            if glab != "null": by_class[glab]["fn"] += 1
    f1s = {}
    for c, s in by_class.items():
        p = s["tp"]/(s["tp"]+s["fp"]) if (s["tp"]+s["fp"])>0 else 0
        r = s["tp"]/(s["tp"]+s["fn"]) if (s["tp"]+s["fn"])>0 else 0
        f = 2*p*r/(p+r) if (p+r)>0 else 0
        f1s[c] = (p, r, f, s["tp"], s["fp"], s["fn"])
    return f1s

def micro_f1(f1s):
    tp = sum(v[3] for v in f1s.values())
    fp = sum(v[4] for v in f1s.values())
    fn = sum(v[5] for v in f1s.values())
    p = tp/(tp+fp) if (tp+fp)>0 else 0
    r = tp/(tp+fn) if (tp+fn)>0 else 0
    return 2*p*r/(p+r) if (p+r)>0 else 0

def macro_f1(f1s):
    return sum(v[2] for v in f1s.values()) / len(f1s)

def tune(rows, gold, grid):
    """Independent per-class threshold tuning."""
    # Start from a high threshold for each class so they "don't fire"
    cur = {c: 1.0 for c in POSITIVE}
    # For each class independently, sweep its threshold while others stay at cur
    best_thr = {}
    for c in POSITIVE:
        best_f = -1
        best_t = grid[0]
        for t in grid:
            cur[c] = t
            preds = predict_with_thresholds(rows, cur)
            f1s = eval_preds(rows, preds, gold)
            if f1s[c][2] > best_f:
                best_f = f1s[c][2]
                best_t = t
        cur[c] = best_t
        best_thr[c] = best_t
        print(f"  best T_{c} = {best_t:.3f}  → class F1 = {best_f:.3f}", file=sys.stderr)
    return cur

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--devel_feat", required=True)
    ap.add_argument("--devel_xml", required=True)
    ap.add_argument("--test_feat", default=None)
    ap.add_argument("--test_xml", default=None)
    ap.add_argument("--out_devel", default=None)
    ap.add_argument("--out_test", default=None)
    args = ap.parse_args()

    print("Building devel probabilities…", file=sys.stderr)
    devel_rows = build_probs(args.model, args.devel_feat)
    gold = load_gold_xml(args.devel_xml)

    grid = [round(x, 3) for x in np.arange(0.05, 0.80, 0.02)]
    print(f"Tuning over {len(grid)} threshold values per class…", file=sys.stderr)
    thr = tune(devel_rows, gold, grid)

    print("=== Devel @ tuned per-class thresholds ===")
    print(f"thresholds: {thr}")
    preds = predict_with_thresholds(devel_rows, thr)
    f1s = eval_preds(devel_rows, preds, gold)
    print(f"{'class':<10}  P     R     F1")
    for c in POSITIVE:
        p, r, f, *_ = f1s[c]
        print(f"{c:<10}  {p:.3f} {r:.3f} {f:.3f}")
    print(f"M-F1: {macro_f1(f1s):.4f}    m-F1: {micro_f1(f1s):.4f}")

    if args.out_devel:
        with open(args.out_devel, "w") as out:
            for sid, e1, e2, label in preds:
                if label != "null":
                    print("|".join([sid, e1, e2, label]), file=out)

    if args.test_feat and args.test_xml:
        print("\nBuilding test probabilities…", file=sys.stderr)
        test_rows = build_probs(args.model, args.test_feat)
        test_gold = load_gold_xml(args.test_xml)
        preds = predict_with_thresholds(test_rows, thr)
        f1s = eval_preds(test_rows, preds, test_gold)
        print("=== Test @ tuned per-class thresholds (from devel) ===")
        for c in POSITIVE:
            p, r, f, *_ = f1s[c]
            print(f"{c:<10}  {p:.3f} {r:.3f} {f:.3f}")
        print(f"M-F1: {macro_f1(f1s):.4f}    m-F1: {micro_f1(f1s):.4f}")
        if args.out_test:
            with open(args.out_test, "w") as out:
                for sid, e1, e2, label in preds:
                    if label != "null":
                        print("|".join([sid, e1, e2, label]), file=out)

if __name__ == "__main__":
    main()
