#!/usr/bin/env python3
# [MOD-2.1] error_strata.py — stratified error analysis.
# Splits errors by (a) document source (MedLine vs DrugBank) and
# (b) sentence length bucket. Helps the report's discussion show
# WHERE the classifier fails, not just by how much.
import sys, argparse
from xml.dom.minidom import parse
from collections import defaultdict, Counter

POSITIVE = ("advise", "effect", "int", "mechanism")
ALL = POSITIVE + ("null",)

def load_data(xmlfile):
    """Returns dict {(sid,e1,e2): {'gold':lab, 'source':doc, 'slen':words}}"""
    info = {}
    tree = parse(xmlfile)
    for s in tree.getElementsByTagName("sentence"):
        sid = s.attributes["id"].value
        stext = s.attributes["text"].value
        slen = len(stext.split())
        source = "MedLine" if "MedLine" in sid else "DrugBank"
        for p in s.getElementsByTagName("pair"):
            e1 = p.attributes["e1"].value
            e2 = p.attributes["e2"].value
            lab = p.attributes["type"].value if p.attributes["ddi"].value == "true" else "null"
            info[(sid, e1, e2)] = {"gold": lab, "source": source, "slen": slen}
    return info

def load_pred(outfile):
    pred = {}
    with open(outfile) as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) == 4:
                pred[(parts[0], parts[1], parts[2])] = parts[3]
    return pred

def f1_from_counts(tp, fp, fn):
    p = tp/(tp+fp) if (tp+fp)>0 else 0
    r = tp/(tp+fn) if (tp+fn)>0 else 0
    return 2*p*r/(p+r) if (p+r)>0 else 0, p, r

def stratify(info, pred, bucket_fn):
    """Bucket pairs by bucket_fn(meta) and compute per-class F1 + macro/micro."""
    buckets = defaultdict(lambda: {c: {"tp": 0, "fp": 0, "fn": 0} for c in POSITIVE})
    bucket_counts = defaultdict(lambda: Counter())
    for k, meta in info.items():
        b = bucket_fn(meta)
        g = meta["gold"]
        p = pred.get(k, "null")
        bucket_counts[b][g] += 1
        if g != "null":
            if p == g: buckets[b][g]["tp"] += 1
            else:      buckets[b][g]["fn"] += 1
        if p != "null" and p != g:
            buckets[b][p]["fp"] += 1
    return buckets, bucket_counts

def print_bucketed(buckets, bucket_counts, name):
    print(f"\n=== Stratified by {name} ===")
    print(f"{'bucket':<20} {'#pairs':>7}", end="")
    for c in POSITIVE:
        print(f"  {c+'-F1':>10}", end="")
    print(f"  {'M-F1':>7}  {'m-F1':>7}")
    for b in sorted(buckets.keys()):
        s = buckets[b]
        npairs = sum(bucket_counts[b].values())
        print(f"{str(b):<20} {npairs:>7}", end="")
        f1s = []
        tp_all, fp_all, fn_all = 0, 0, 0
        for c in POSITIVE:
            f, _, _ = f1_from_counts(s[c]["tp"], s[c]["fp"], s[c]["fn"])
            f1s.append(f)
            tp_all += s[c]["tp"]; fp_all += s[c]["fp"]; fn_all += s[c]["fn"]
            print(f"  {f:>10.3f}", end="")
        macro = sum(f1s)/len(f1s)
        micro, _, _ = f1_from_counts(tp_all, fp_all, fn_all)
        print(f"  {macro:>7.3f}  {micro:>7.3f}")
        # gold distribution
        gd = bucket_counts[b]
        print(f"{'  gold dist:':<20} ", end="")
        for c in ALL:
            print(f"  {c}={gd.get(c,0)}", end="")
        print()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xml")
    ap.add_argument("pred_out")
    args = ap.parse_args()
    info = load_data(args.xml)
    pred = load_pred(args.pred_out)

    # by source
    bs, bc = stratify(info, pred, lambda m: m["source"])
    print_bucketed(bs, bc, "document source")

    # by sentence length bucket
    def slen_bucket(m):
        n = m["slen"]
        if n <= 10: return "01-short(<=10)"
        elif n <= 25: return "02-medium(11-25)"
        elif n <= 50: return "03-long(26-50)"
        else: return "04-vlong(>50)"
    bs, bc = stratify(info, pred, slen_bucket)
    print_bucketed(bs, bc, "sentence length")

if __name__ == "__main__":
    main()
