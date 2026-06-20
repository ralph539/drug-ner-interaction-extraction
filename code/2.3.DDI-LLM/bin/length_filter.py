#!/usr/bin/env python3
# [MOD-2.3] length_filter.py — post-process FT inference .out by forcing
# null on long sentences. Motivated by error_strata showing the >50-word
# bucket collapses to ~M=12 F1; replacing those predictions with null
# is a free inference-time guardrail.
import sys
from xml.dom.minidom import parse

def sentence_lengths(xmlfile):
    lens = {}
    tree = parse(xmlfile)
    for s in tree.getElementsByTagName("sentence"):
        lens[s.attributes["id"].value] = len(s.attributes["text"].value.split())
    return lens

if len(sys.argv) != 5:
    print("Usage: length_filter.py <gold.xml> <pred.out> <out.out> <max_len>", file=sys.stderr)
    sys.exit(1)

gold, src, dst, maxlen = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
lens = sentence_lengths(gold)

kept = dropped = 0
with open(src) as fi, open(dst, "w") as fo:
    for line in fi:
        sid = line.split("|", 1)[0]
        if lens.get(sid, 0) > maxlen:
            dropped += 1
            continue
        fo.write(line)
        kept += 1
print(f"kept={kept} dropped={dropped} (max_len={maxlen})", file=sys.stderr)
