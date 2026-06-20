#!/usr/bin/env python3
# [MOD-2.1] two_stage.py — two-stage DDI classifier
# Stage 1: binary (null vs positive) LR
# Stage 2: 4-way LR trained ONLY on positive examples
# At inference: if stage 1 says "positive" with prob >= threshold, run stage 2
# Goal: decouple the heavy null/positive imbalance from the inter-positive
# class boundaries, which the per-class confusion shows is the main error
# mode in the single-stage MEM (positives → null).

import sys, os, pickle, argparse
import numpy as np
import scipy
from sklearn.linear_model import LogisticRegression
from dataset import Dataset

POSITIVE = ("advise", "effect", "int", "mechanism")
ALL = POSITIVE + ("null",)

def _build_X(ds):
    fidx = ds.feature_index()
    X, Y = ds.csr_matrix()
    return X, Y, fidx

def train(train_feat, model_path, C=1.0, solver="lbfgs", max_iter=1500,
          stage1_balanced=True, stage2_balanced=False):
    """Train stage 1 (binary null vs positive) and stage 2 (4-way on positives)."""
    ds = Dataset(train_feat)
    fidx = ds.feature_index()
    X, Y = ds.csr_matrix()

    # Stage 1: relabel as 'null' or 'positive'
    Y1 = ["null" if y == "null" else "positive" for y in Y]
    cw1 = "balanced" if stage1_balanced else None
    stage1 = LogisticRegression(C=C, solver=solver, max_iter=max_iter,
                                class_weight=cw1, verbose=0)
    print(f"Training stage1 (binary, class_weight={cw1}) on {X.shape[0]} examples...",
          file=sys.stderr)
    stage1.fit(X, Y1)

    # Stage 2: keep only positive examples, train 4-way
    pos_idx = [i for i, y in enumerate(Y) if y != "null"]
    print(f"Stage2 will train on {len(pos_idx)} positive examples (balanced={stage2_balanced})", file=sys.stderr)
    X2 = X[pos_idx]
    Y2 = [Y[i] for i in pos_idx]
    cw2 = "balanced" if stage2_balanced else None
    stage2 = LogisticRegression(C=C, solver=solver, max_iter=max_iter,
                                class_weight=cw2, verbose=0)
    stage2.fit(X2, Y2)

    with open(model_path, "wb") as f:
        pickle.dump({"stage1": stage1, "stage2": stage2, "fidx": fidx}, f)
    print(f"Saved 2-stage model to {model_path}", file=sys.stderr)


def predict(devel_feat, model_path, out_path, threshold=0.5):
    """Predict on devel using saved 2-stage model. threshold = prob of stage1 'positive'."""
    with open(model_path, "rb") as f:
        m = pickle.load(f)
    stage1, stage2, fidx = m["stage1"], m["stage2"], m["fidx"]
    ds = Dataset(devel_feat)
    # Need to encode each instance into a row vector consistent with fidx.
    out = open(out_path, "w")
    s1_classes = list(stage1.classes_)
    pos_idx_in_classes = s1_classes.index("positive")
    for ex in ds.instances():
        # build row
        rowi, colj, data = [], [], []
        for feat in ex["features"]:
            if feat in fidx:
                rowi.append(0)
                colj.append(fidx[feat])
                data.append(1)
        X = scipy.sparse.csr_matrix((data, (rowi, colj)), shape=(1, len(fidx)))
        probs = stage1.predict_proba(X)[0]
        p_pos = probs[pos_idx_in_classes]
        if p_pos >= threshold:
            label = stage2.predict(X)[0]
            print(ex["sid"], ex["e1"], ex["e2"], label, sep="|", file=out)
        # else: predict null → skip (eval format excludes null)
    out.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["train", "predict"])
    ap.add_argument("featfile")
    ap.add_argument("modelfile")
    ap.add_argument("--out", default=None)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--solver", default="lbfgs")
    ap.add_argument("--max_iter", type=int, default=1500)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--no_balance", action="store_true",
                    help="Disable class_weight=balanced in stage1")
    ap.add_argument("--s2_balance", action="store_true",
                    help="Enable class_weight=balanced in stage2 (positive-only 4-way)")
    args = ap.parse_args()
    if args.action == "train":
        train(args.featfile, args.modelfile, C=args.C, solver=args.solver,
              max_iter=args.max_iter, stage1_balanced=not args.no_balance,
              stage2_balanced=args.s2_balance)
    else:
        predict(args.featfile, args.modelfile, args.out, threshold=args.threshold)
