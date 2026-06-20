#!/usr/bin/env python3
# [MOD-2.1] feature_weights.py — inspect top features per class from
# the saved single-stage MEM (LogisticRegression). Tells us *why* the
# classifier votes for each class.
import sys, pickle, argparse

def top_features(model_path, top_n=15):
    with open(model_path, "rb") as f:
        tagger = pickle.load(f)
    with open(model_path + ".idx", "rb") as f:
        fidx = pickle.load(f)
    inv_fidx = {v: k for k, v in fidx.items()}
    classes = list(tagger.classes_)
    coef = tagger.coef_   # shape: (n_classes, n_features) ; for binary problems (n_features,)
    if coef.ndim == 1:
        # binary case — not expected but handle
        coef = coef.reshape(1, -1)
    print(f"Classes: {classes}")
    print(f"Feature count: {coef.shape[1]}")
    for i, c in enumerate(classes):
        print(f"\n=== class={c} top +{top_n} features (most positive weights) ===")
        w = coef[i]
        idx_pos = w.argsort()[::-1][:top_n]
        for j in idx_pos:
            print(f"  {w[j]:+7.3f}  {inv_fidx.get(j, '<unk>')}")
        print(f"--- bottom {top_n} (most negative weights) ---")
        idx_neg = w.argsort()[:top_n]
        for j in idx_neg:
            print(f"  {w[j]:+7.3f}  {inv_fidx.get(j, '<unk>')}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()
    top_features(args.model, args.top)
