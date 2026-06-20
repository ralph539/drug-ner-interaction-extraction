#!/usr/bin/env python3
# [MOD-2.1] confmat.py — build a per-class confusion matrix
# Reads gold (devel.xml or test.xml) and predictions (devel-MEM-*.out),
# emits a 5x5 confusion table (advise/effect/int/mechanism/null).
import sys, os
from xml.dom.minidom import parse
from collections import defaultdict

def load_gold(xmlfile):
    gold = {}
    tree = parse(xmlfile)
    for s in tree.getElementsByTagName("sentence"):
        sid = s.attributes["id"].value
        for p in s.getElementsByTagName("pair"):
            e1 = p.attributes["e1"].value
            e2 = p.attributes["e2"].value
            if p.attributes["ddi"].value == "true":
                lab = p.attributes["type"].value
            else:
                lab = "null"
            gold[(sid, e1, e2)] = lab
    return gold

def load_pred(outfile):
    pred = {}
    with open(outfile) as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) != 4: continue
            sid, e1, e2, label = parts
            pred[(sid, e1, e2)] = label
    return pred

def confusion(gold, pred):
    classes = ["advise", "effect", "int", "mechanism", "null"]
    mat = {g: defaultdict(int) for g in classes}
    for k, gl in gold.items():
        pl = pred.get(k, "null")
        mat[gl][pl] += 1
    return classes, mat

def main(xmlfile, outfile):
    gold = load_gold(xmlfile)
    pred = load_pred(outfile)
    classes, mat = confusion(gold, pred)
    # header
    header = "GOLD\\PRED " + "".join(f"{c:>10}" for c in classes) + "    Σ"
    print(header)
    for g in classes:
        row = mat[g]
        tot = sum(row.values())
        cells = "".join(f"{row[p]:>10}" for p in classes)
        print(f"{g:<10}{cells}{tot:>6}")
    # per-class precision/recall using positive-only convention (null excluded)
    print()
    print("Per-class (positive classes only):")
    print(f"{'class':<10}{'TP':>6}{'FP':>6}{'FN':>6}  P     R     F1")
    for c in classes:
        if c == "null": continue
        tp = mat[c][c]
        # FP: predicted c, gold != c
        fp = sum(mat[g][c] for g in classes if g != c)
        # FN: gold c, predicted != c
        fn = sum(mat[c][p] for p in classes if p != c)
        p = tp/(tp+fp) if tp+fp else 0
        r = tp/(tp+fn) if tp+fn else 0
        f1 = 2*p*r/(p+r) if p+r else 0
        print(f"{c:<10}{tp:>6}{fp:>6}{fn:>6}  {p:.3f} {r:.3f} {f1:.3f}")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
