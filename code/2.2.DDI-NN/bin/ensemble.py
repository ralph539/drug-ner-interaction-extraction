#!/usr/bin/env python3
# [MOD-2.2] ensemble.py — combine two .out files into one with voting rules.
# Strategies (same as System 2.1):
#   - OR  (A-priority): predict positive if either model predicts positive;
#                       on disagreement keep A's label.
#   - AND (intersect):  predict positive only when both agree on the SAME label.
#   - OR-B (B-priority): like OR but B's label wins disagreements.
import sys, argparse
from collections import Counter

def load_out(p):
    out = {}
    with open(p) as f:
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

def ensemble_or(A, B, keys):
    preds = {}
    for k in keys:
        a = A.get(k, "null"); b = B.get(k, "null")
        if a != "null": preds[k] = a
        elif b != "null": preds[k] = b
    return preds

def ensemble_and(A, B, keys):
    preds = {}
    for k in keys:
        a = A.get(k, "null"); b = B.get(k, "null")
        if a != "null" and a == b: preds[k] = a
    return preds

def ensemble_orB(A, B, keys):
    preds = {}
    for k in keys:
        a = A.get(k, "null"); b = B.get(k, "null")
        if b != "null": preds[k] = b
        elif a != "null": preds[k] = a
    return preds

def write_out(p, preds):
    with open(p, "w") as f:
        for (sid, e1, e2), lab in preds.items():
            print("|".join([sid, e1, e2, lab]), file=f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("A")
    ap.add_argument("B")
    ap.add_argument("gold_xml")
    ap.add_argument("--out_or", default=None)
    ap.add_argument("--out_and", default=None)
    ap.add_argument("--out_or_B", default=None)
    args = ap.parse_args()
    A = load_out(args.A); B = load_out(args.B); g = load_gold_xml(args.gold_xml)
    keys = list(g.keys())
    same = sum(1 for k in keys if A.get(k,"null") == B.get(k,"null") and A.get(k,"null") != "null")
    a_only = sum(1 for k in keys if A.get(k,"null") != "null" and B.get(k,"null") == "null")
    b_only = sum(1 for k in keys if B.get(k,"null") != "null" and A.get(k,"null") == "null")
    both_diff = sum(1 for k in keys if A.get(k,"null") != "null" and B.get(k,"null") != "null"
                    and A.get(k,"null") != B.get(k,"null"))
    print(f"A positives: {len(A)}, B positives: {len(B)}")
    print(f"Same label : {same}")
    print(f"A-only     : {a_only}")
    print(f"B-only     : {b_only}")
    print(f"Both diff  : {both_diff}")
    if args.out_or: write_out(args.out_or, ensemble_or(A, B, keys))
    if args.out_and: write_out(args.out_and, ensemble_and(A, B, keys))
    if args.out_or_B: write_out(args.out_or_B, ensemble_orB(A, B, keys))

if __name__ == "__main__":
    main()
