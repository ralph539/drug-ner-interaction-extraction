#!/usr/bin/env python3
# [MOD-2.1] ensemble.py — combine single-stage and two-stage predictions.
#
# Ensemble strategies tested:
#   - OR (union):       predict positive if either model predicts positive.
#                       on disagreement (different positives), prefer the
#                       single-stage model's label.
#   - AND (intersect):  predict positive only if both predict the SAME label.
#   - prob-avg:         average per-class probabilities and argmax (with
#                       global threshold tuned on devel).
#
# Inputs are two .out files in the standard DDI eval format
# (sid|e1|e2|label, null rows omitted).
import sys, argparse
from collections import defaultdict

def load_out(path):
    """Returns dict {(sid, e1, e2): label} for non-null entries."""
    out = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) == 4:
                out[(parts[0], parts[1], parts[2])] = parts[3]
    return out

def load_gold_xml(xmlfile):
    from xml.dom.minidom import parse
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

def ensemble_or(A, B, gold_keys):
    """A's prediction takes priority. If A is null but B isn't, use B."""
    preds = {}
    for k in gold_keys:
        a = A.get(k, "null")
        b = B.get(k, "null")
        if a != "null":
            preds[k] = a
        elif b != "null":
            preds[k] = b
        # else: stays null (omitted)
    return preds

def ensemble_and(A, B, gold_keys):
    """Only output a label when both models agree on the SAME label."""
    preds = {}
    for k in gold_keys:
        a = A.get(k, "null")
        b = B.get(k, "null")
        if a != "null" and a == b:
            preds[k] = a
    return preds

def ensemble_priority_B(A, B, gold_keys):
    """B takes priority instead of A."""
    preds = {}
    for k in gold_keys:
        a = A.get(k, "null")
        b = B.get(k, "null")
        if b != "null":
            preds[k] = b
        elif a != "null":
            preds[k] = a
    return preds

def write_out(path, preds):
    with open(path, "w") as f:
        for (sid, e1, e2), lab in preds.items():
            print("|".join([sid, e1, e2, lab]), file=f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("A_out")
    ap.add_argument("B_out")
    ap.add_argument("gold_xml")
    ap.add_argument("--out_or", default=None)
    ap.add_argument("--out_and", default=None)
    ap.add_argument("--out_or_B", default=None)
    args = ap.parse_args()

    A = load_out(args.A_out)
    B = load_out(args.B_out)
    gold = load_gold_xml(args.gold_xml)
    keys = list(gold.keys())

    print(f"A has {len(A)} positive predictions; B has {len(B)} positive predictions")
    same = sum(1 for k in keys if A.get(k, "null") == B.get(k, "null") and A.get(k, "null") != "null")
    a_only = sum(1 for k in keys if A.get(k, "null") != "null" and B.get(k, "null") == "null")
    b_only = sum(1 for k in keys if B.get(k, "null") != "null" and A.get(k, "null") == "null")
    both_diff = sum(1 for k in keys if A.get(k, "null") != "null" and B.get(k, "null") != "null"
                    and A.get(k, "null") != B.get(k, "null"))
    print(f"Same label (positive): {same}")
    print(f"A-only positive:       {a_only}")
    print(f"B-only positive:       {b_only}")
    print(f"Both positive, diff:   {both_diff}")

    if args.out_or:
        write_out(args.out_or, ensemble_or(A, B, keys))
    if args.out_and:
        write_out(args.out_and, ensemble_and(A, B, keys))
    if args.out_or_B:
        write_out(args.out_or_B, ensemble_priority_B(A, B, keys))

if __name__ == "__main__":
    main()
