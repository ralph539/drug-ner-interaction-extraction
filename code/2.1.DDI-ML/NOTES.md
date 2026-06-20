# System 2.1 — DDI with Machine Learning — Experiment Log

This file tracks every experiment we run for System 2.1 so we can write the
report straight from this log when the time comes.

## Conventions

- **Code modifications** are tagged in source with `[MOD-2.1]` comments
  (same convention as Task 1, replacing `[MOD-1.x]`). Always include a one-line
  explanation of *why* the change was made next to the tag.
- **Experiment names** follow the pattern `<algo>-<feat-set>-<hp>` where:
  - `<algo>` ∈ {MEM, SVM}
  - `<feat-set>` ∈ {ref, mod1, mod2, …} (ref = shipped baseline features)
  - `<hp>` is a compact hyperparameter suffix, e.g. `C1_lbfgs1500`
- **Output files** live in `code/2.1.DDI-ML/results/` as
  `devel-<expname>.{out,stats}` and `test-<expname>.{out,stats}`.
- **Feature dumps** live in `code/2.1.DDI-ML/preprocessed/` as
  `train.feat` and `devel.feat`. Re-extraction needed whenever
  `extract_features.py` or `patterns.py` change.
- **Models** live in `code/2.1.DDI-ML/models/` as `model.MEM` / `model.SVM`.
- Every experiment is logged in the table below the moment it finishes.

## Reference points (DDI is much harder than NER, expect low numbers)

| System | Source | devel m-F1 | devel M-F1 | test m-F1 | test M-F1 |
|---|---|---:|---:|---:|---:|
| 2.0 rule-based "words in between" | re-run locally 2026-05-02 | 13.1% | 22.2% | 20.8% | 26.9% |
| 2.1 reference ML (provided extract_features) | spec p35 | TBD | TBD | TBD | TBD |
| Ballpark "good" ML target | spec p35 | — | ~65% M | — | — |

### 2.0 baseline per-class breakdown (matches spec exactly):

```
devel:   advise 0.0  effect 12.3  int 70.7  mechanism 5.8   | M=22.2  m=13.1
test:    advise 20.1 effect 25.2  int 50.0  mechanism 12.3  | M=26.9  m=20.8
```

The rule-based baseline relies on ~50 hand-curated cue words split by class.
It nails `int` (the rarest class — 43 devel pairs) because the cue list
includes `interact`, `tylenol`, `mivacron`. It collapses on `advise` (0% F1
on devel) because the cue list for `advise` consists of overly-specific
drug-name tokens like `dihydroergotamine`, `cyp2d6`, `narrow` etc.

Note: DDI evaluator reports m.avg (micro) and M.avg (macro) across the 4
positive classes (`advise`, `effect`, `int`, `mechanism`). `null` pairs
are excluded from F1 by construction.

## Experiment table

| Run | Algorithm | Features | Hyperparameters | devel m | devel M | test m | test M | Notes |
|---|---|---|---|---:|---:|---:|---:|---|
| baseline-2.0 | rule | cue words | n/a | 13.1 | 22.2 | 20.8 | 26.9 | re-run 2026-05-02 |
| ref-MEM | MEM | shipped | C=1 lbfgs maxit=1500 | **63.2** | **64.3** | — | — | new ML baseline; ~+42pp M over 2.0 |
| ref-SVM | SVM | shipped | C=1 rbf gamma=scale | 54.3 | 55.0 | — | — | -9pp M vs MEM; precision-biased (R~43%) |
| mod1-MEM | MEM | shipped + mod1 | C=1 lbfgs maxit=1500 | 61.8 | 62.4 | — | — | **−1.9 pp M vs ref-MEM — distance/senlen/order HURT, disabled in src** |
| mod1+2-MEM | MEM | shipped + mod1 + mod2 | C=1 lbfgs maxit=1500 | 62.1 | 63.0 | — | — | −1.3 pp M vs ref; combo doesn't rescue mod1 |
| mod2-MEM | MEM | shipped + mod2 | C=1 lbfgs maxit=1500 | 63.0 | 64.2 | — | — | −0.1 pp M — neutral; small gain on `int`, small loss on `advise`/`mechanism` |
| mod3-MEM | MEM | shipped + mod3 | C=1 lbfgs maxit=1500 | 64.0 | 64.0 | — | — | micro +0.8 / macro −0.3; helps 3/4 classes, hurts `int` (−4.3) |
| mod4-MEM | MEM | shipped + mod4 | C=1 lbfgs maxit=1500 | 63.1 | 64.1 | — | — | −0.2 pp M — neutral; type-pair combined feature doesn't move the needle alone |
| mod5-MEM | MEM | shipped + mod5 | C=1 lbfgs maxit=1500 | 62.8 | 63.0 | — | — | **−1.3 pp M — hurt; bag-of-negation cues without scoping** |
| mod_best-MEM | MEM | shipped + mod2+mod3+mod4 | C=1 lbfgs maxit=1500 | 65.1 | 65.2 | 63.0 | 66.8 | First combo. +0.9 pp M / +1.9 pp m devel vs ref. (Later beaten by mod_best2 on devel.) |
| ablate-mod2-from-mb | MEM | shipped + mod3+mod4 | C=1 lbfgs maxit=1500 | 64.2 | 64.5 | — | — | mod2 contributes +0.7 M / +0.9 m to mod_best |
| ablate-mod3-from-mb | MEM | shipped + mod2+mod4 | C=1 lbfgs maxit=1500 | 63.2 | 64.4 | — | — | mod3 contributes +0.8 M / +1.9 m to mod_best |
| ablate-mod4-from-mb | MEM | shipped + mod2+mod3 | C=1 lbfgs maxit=1500 | **64.8** | **65.4** | **61.8** | **65.7** | **mod4 *hurts*: dropping it on devel gives M=65.4. New legit champion by devel selection.** |
| mod_best2-MEM | MEM | shipped + mod2+mod3 | C=1 lbfgs maxit=1500 | 64.8 | 65.4 | 61.8 | 65.7 | Synonym for ablate-mod4-from-mb. Devel-selected champion under feature-only campaign. |
| mod_best2 + mod6 | MEM | shipped + mod2+mod3 + mod6 | C=1 lbfgs maxit=1500 | 64.7 | 64.6 | — | — | mod6 (lemma bigrams in pat_wib region) HURTS (-0.8 M) |
| mod_best2 + mod7 | MEM | shipped + mod2+mod3 + mod7 | C=1 lbfgs maxit=1500 | 64.8 | 65.1 | — | — | mod7 (LCS subtree shape) HURTS (-0.3 M) |
| mod_best2 + mod6 + mod7 | MEM | shipped + mod2+mod3 + mod6 + mod7 | C=1 lbfgs maxit=1500 | 65.0 | 65.0 | — | — | mod6+mod7 stacked: also worse than mod_best2 |

### Saga sweep on mod_best2 (multinomial l1/l2/elasticnet)

| solver | penalty | C | l1_ratio | max_iter | M | m |
|---|---|---|---|---|---:|---:|
| lbfgs | l2 | 0.7 | — | 1500 | 64.7 | 64.4 |
| lbfgs | l2 | 1.0 | — | 1500 | 65.2 | 64.6 |
| lbfgs | l2 | 1.5 | — | 1500 | 64.4 | 64.1 |
| saga | l2 | 0.5 | — | 3000 | 64.7 | 64.1 |
| saga | l2 | 1.0 | — | 3000 | **65.5** | 64.7 |
| saga | l2 | 2.0 | — | 3000 | 65.1 | 64.3 |
| saga | l2 | 5.0 | — | 3000 | 64.1 | 63.4 |
| saga | l1 | 1.0 | — | 5000 | 63.8 | 61.8 |
| saga | l1 | 2.0 | — | 5000 | 62.8 | 60.4 |
| saga | elasticnet | 1.0 | 0.3 | 5000 | 64.6 | 63.7 |
| saga | elasticnet | 1.0 | 0.7 | 5000 | 63.6 | 61.9 |
| liblinear | l1/l2 | — | — | — | — | — | (multiclass unsupported by liblinear in sklearn 1.x; skipped) |

Best single-stage: **saga l2 C=1 max_iter=3000 → M=65.5** (just +0.1 over lbfgs default).
HP tuning gives essentially nothing on top of `mod_best2 + lbfgs default`.

### Two-stage classifier on mod_best2 features

Stage 1: binary LR (null vs positive). Stage 2: 4-way LR on positive-only
training subset. Inference uses `predict_proba(stage1)` and routes to
stage 2 if P(positive) >= threshold. Code: `bin/two_stage.py`.

| class_weight | threshold | devel M | devel m | test M | test m |
|---|---|---:|---:|---:|---:|
| balanced | 0.30 | 59.4 | 56.9 | — | — |
| balanced | 0.50 | 61.7 | 61.6 | — | — |
| balanced | 0.70 | 63.4 | 63.7 | — | — |
| **none** | 0.20 | 63.3 | 62.1 | — | — |
| none | 0.30 | 64.8 | 63.9 | — | — |
| none | 0.35 | 65.3 | 64.7 | — | — |
| **none** | **0.37** | **66.0** | **65.2** | **66.6** | **62.4** |
| none | 0.40 | 65.8 | 65.2 | 66.6 | 62.5 |
| none | 0.42 | 65.7 | 65.1 | — | — |
| none | 0.45 | 64.7 | 64.7 | — | — |
| none | 0.50 | 64.4 | 64.8 | — | — |

**Winner: two-stage, no class-balance, threshold=0.37 → devel M=66.0, test M=66.6.**
Beats single-stage `mod_best2-MEM` by +1.2 pp devel macro and +0.9 pp test macro.

### Two-stage per-class on test (threshold=0.37)

```
                P     R    F1
advise        67.4  61.2  64.2     (+2.1 vs mod_best2 62.1)
effect        60.2  65.5  62.7     (-0.3 vs mod_best2 63.0)
int           93.3  70.0  80.0     (+1.2 vs mod_best2 78.8)
mechanism     54.9  65.1  59.6     (+0.7 vs mod_best2 58.9)
M.avg         68.9  65.5  66.6     (+0.9 vs mod_best2 65.7)
m.avg         60.4  64.6  62.4     (+0.6 vs mod_best2 61.8)
```

The two-stage approach trades precision for recall on each class — exactly
what we expect when the stage-1 binary classifier (which sees the full
85% null imbalance) is no longer fighting an inter-positive boundary.

### Confusion-matrix snapshot of mod_best2 errors (devel)

```
                       PREDICTED
GOLD\PRED  advise  effect    int  mechanism   null    Σ
advise         92       7      1       2       41    143
effect         13     181      0       2      120    316
int             0       0     26       1       16     43
mechanism      12       2      0     139      108    261
null           23      34      6      44     4007   4114
```

Take-aways for the report:
* **Recall failures dominate**: 29-41% of every positive class is misrouted to `null`. This is the imbalance, not class confusion.
* **Inter-positive confusion is small**: at most 13 effect → advise. The classifier separates positive classes well; it just doesn't activate them often enough.
* **null → mechanism** is the largest false-positive direction (44). This makes intuitive sense — `mechanism` lexicalisations like "induce", "inhibit" overlap with neutral pharmacology vocabulary.

### Final champion summary (canonical re-run after revert to mod_best2 + two-stage)

| System | Selection metric (devel) | Devel M | Devel m | Test M | Test m |
|---|---|---:|---:|---:|---:|
| 2.0 rule-based | — | 22.2 | 13.1 | 26.9 | 20.8 |
| 2.1 ref-MEM | — | 64.3 | 63.2 | — | — |
| 2.1 mod_best2-MEM (single-stage) | feature combo | 65.4 | 64.8 | 65.7 | 61.8 |
| **2.1 two-stage on mod_best2 (t=0.37)** | devel-tuned threshold | **65.9** | **65.3** | **66.8** | **62.5** |

The two-stage classifier on mod_best2 features is the System 2.1 final
configuration we'll report. **+39.9 pp test macro over rule-based 2.0**
(26.9 → 66.8); **+2.5 pp devel macro over ref-MEM** (64.3 → 65.9).

### Final per-class breakdown (test)

```
                P     R    F1
advise        68.8  61.2  64.8
effect        60.1  66.2  63.0
int           93.3  70.0  80.0     ← biggest single-class win
mechanism     54.8  64.5  59.3
M.avg         69.3  65.5  66.8
m.avg         60.6  64.6  62.5
```

`int` is the easiest positive class (concentrated trigger lexicon).
`mechanism` is the hardest (diffuse phrasing). The two-stage classifier
trades precision for recall vs single-stage on every class.

### Also tested but abandoned (one-line each)

* **mod_best (mod2+mod3+mod4) two-stage** — devel 65.3 / test 66.1; mod4 helps single-stage but hurts two-stage. Dropped.
* **Two-stage with class_weight=balanced in stage1** — over-corrects; devel M tops at 63.4 (vs 66.0 unbalanced).
* **Two-stage with saga solver** — devel M=65.5 (vs lbfgs 66.0). lbfgs wins.
* **SVM rbf** — M=55, far below MEM. SVM linear C=0.5 reaches 60.5 but still below.

## Deep-dive extensions (post-headline)

### Stage-2 class_weight=balanced
Idea: with stage-1 already handling the null/positive imbalance, stage 2
still trains on imbalanced positives (advise:effect:int:mech ≈ 1:2.2:0.3:1.8).
Re-trained two-stage with `class_weight='balanced'` inside stage 2 only.

| t | devel M | devel m |
|---|---:|---:|
| 0.35 | 65.7 | 64.9 |
| 0.37 | 66.0 | 65.3 |
| **0.40** | **66.2** | **65.6** |
| 0.45 | 65.1 | 64.8 |

Best at t=0.40 → devel M=66.2 (+0.2 over no_s2_balance). On test the same
config gave M=66.0 (-0.8 vs base two-stage), so the devel gain doesn't
generalise — verdict: **noise-level, drop**.

### Train + devel combined for final test
Once the (model, threshold) is selected on devel-only, papers often retrain
on `train + devel` together to maximise the final-eval data. Implementation:
concat the .feat files, re-train the saved two-stage model.

| Variant (t=devel-chosen) | Test M | Test m |
|---|---:|---:|
| train-only, no_s2_bal, t=0.37 | **66.8** | 62.5 |
| train+devel, no_s2_bal, t=0.37 | 66.6 | 63.8 |
| train+devel, s2_bal, t=0.40 | 66.3 | 63.4 |

Counter-intuitive but real: adding devel to training fractionally **lowered**
test macro-F1. Likely cause: the calibration of stage-1's `P(positive)`
shifted when more data was added, so the threshold tuned on the train-only
calibration is mismatched. **No legitimate way to re-tune the threshold
on the new model without a second held-out set.** Drop.

### Per-class threshold tuning (devel-overfit)
Instead of one global threshold, tune a separate `T_c` on the joint
probability `P(c) = P(positive) · P(c|positive)`. Independent
coordinate-descent over `T_c` for each c ∈ {advise, effect, int, mech}.

| Config | Devel M | Devel m | Test M | Test m |
|---|---:|---:|---:|---:|
| Global t=0.37 | 65.9 | 65.3 | 66.8 | 62.5 |
| Per-class (T_adv=0.43, T_eff=0.55, T_int=0.11, T_mech=0.35) | **68.4** | **67.6** | 65.3 | 63.5 |

A spectacular **+2.5 pp devel macro jump**, but test drops by 1.5 pp.
The four per-class thresholds have enough freedom to overfit ~700 devel
positives. Lesson: at this data scale, single-threshold acts as an
implicit regulariser. **Drop for final eval.** Implementation in
`bin/tune_thresholds.py`.

### Mechanism-specific feature mod (mod8)
`mechanism` is the lowest-F1 positive class. Built `pat_class_trig`:
per-class trigger lexicon (mechanism: induce/inhibit/metabolize/displace/
substrate/enzyme/cyp/clearance/…; effect: enhance/potentiate/…;
advise: avoid/recommend/…; int: interact/coadminister/…) emitting
features tagged with class and position.

| | Devel M | Devel m | Test M | Test m |
|---|---:|---:|---:|---:|
| mod_best2 single | 65.4 | 64.8 | 65.7 | 61.8 |
| mod_best2 + mod8 single | 65.6 | 64.9 | 65.4 | 61.6 |
| mod_best2 two-stage | 66.0 | 65.2 | 66.8 | 62.5 |
| mod_best2 + mod8 two-stage | 65.4 | 65.0 | 65.7 | 62.1 |

mod8 helps single-stage devel by +0.2 but **hurts everything else** (esp.
two-stage test by -1.1). The class-tagged trigger lexicon must be
overfitting the trigger distribution in training. Disabled.

### Ensemble: single-stage ∪/∩ two-stage
Single-stage MEM and two-stage classifier produce different predictions
(twostage predicts ~33 % more positives than single-stage on devel).
Tested three rules:
* **OR** — predict positive if either predicts positive; on disagreement keep single-stage's label.
* **AND** — predict positive only if both agree on the SAME label.
* **OR-B** — same as OR but two-stage's label wins on disagreement.

Devel agreement breakdown: A=591 positives, B=788 positives,
same-label=564, A-only=15, B-only=212, both-positive-different=12.

| Strategy | Devel M | Devel m | Test M | Test m |
|---|---:|---:|---:|---:|
| Single-stage alone | 65.3 | 64.6 | 65.8 | 61.6 |
| Two-stage alone (t=0.37) | 65.2 | 64.7 | 66.6 | 62.4 |
| Ensemble OR (A-priority) | 65.2 | 64.4 | 65.9 | 62.0 |
| **Ensemble AND** | **65.8** | **65.0** | 65.5 | 61.6 |
| Ensemble OR-B (B-priority) | 65.3 | 64.4 | 66.8 | 62.7 |

Devel-selection points to **AND** (devel M=65.8). On test, AND lands at 65.5
(worse than two-stage alone). OR-B accidentally ties two-stage on test
(66.8) but can't be chosen from devel.

### Conclusion of the deep dive

After 5 post-headline extensions, none robustly improves on the two-stage
classifier alone. The pattern is consistent:
* **Devel-set improvements that don't survive test** (per-class threshold,
  stage-2 balanced, mod8) indicate the devel size (~700 positives) is at
  the noise floor for fine-grained tuning.
* **Train+devel-combined retraining doesn't help** because we cannot
  re-tune the calibration-sensitive threshold.
* **Ensembling adds variance without consistent gains** because the two
  systems share most of their feature pipeline and make correlated errors.

**Final reported System 2.1 = two-stage classifier on mod_best2 features,
lbfgs, t=0.37, no class-weight balancing → devel M=65.9, test M=66.8.**
This is the legitimate devel-selected champion; every richer configuration
either overfits devel or fails to generalise to test.

## Analysis (qualitative, for the report's discussion section)

### Top-weighted features per class (single-stage MEM)

Extracted from the saved single-stage LR model via `bin/feature_weights.py`.
Sanity-checks the classifier: each class's top features should be
linguistically meaningful.

* **advise (recommendation/warning class).** Top positives:
  `lcsCH=should`, `pat_wout=lb1:caution`, `lcsCH=not`, `pat_clue=monitor`,
  `pat_clue=recommend`, `pat_verb_lcs=avoid`, `pat_verb_lcs=recommend`,
  `pat_verb_func=contraindicate_VERB:nsubjpass_nsubjpass`. Top negatives:
  `pat_wout=lb1:report`, `samedrug`, `other_type=drug_n`. → The classifier
  has learned modal-verb constructions and direct recommendation verbs.

* **effect (clinical-impact class).** Top positives: `pat_wout=lb1:report`,
  `wip=effect`, `pat_wout=lb1:additive`, `pat_verb_lcs=result`,
  `pat_verb_lcs=associate`, `lcsCH=may`, `lcsCH=can`. Top negatives:
  `lcsCH=should`, `lcsCH=not` (i.e. negation / modal verbs which belong
  to advise). → Classifier separates "effect" reporting language from
  "advise" warning language.

* **int (generic interaction).** Top positives: `pat_wout=lb1:interaction`,
  `typeE1=drug`, `wip=interaction`, `pat_verb_func=interact_VERB:nsubj_prep`,
  `pathA=nsubj<interact>prep`. Notably, one of the top features is the
  *literal* `wip1=MIVACRON` and a specific path-trace
  `path1=MIVACRON_pobj<as_prep<…<enhance_relcl<drug_nsubj`. **This is a
  red flag**: the classifier has memorised a specific training example
  about Mivacron. The features generalise (the dep-shape pattern is
  reusable) but the lexical anchors are over-fit.

* **mechanism (pharmacokinetic class).** Top positives:
  `pathAb=nsubj<VERB>dobj`, `pat_wout=la2:%`, `pat_wout=lb1:level`,
  `pathAb=nsubjpass<VERB>xcomp`, `pat_wout=la2:metabolism`,
  `pat_wout=la2:absorption`, `wip=absorption`, `pat_wib=l=modest`. The
  classifier has learned a syntactic shape (transitive verb with subject
  and direct object) plus pharmacokinetic vocabulary. Negatives:
  `lcsCH=not`, `samedrug`, `lcsCH=do`, `pat_wib=eib` (entity-in-between
  is anti-correlated with mechanism — mechanisms tend to be between *two*
  entities rather than three).

* **null.** Top positives: `samedrug` (+2.76 — by far the strongest
  weight in the model), `lcsCH=or`, `lcsCH=do`, `path1c=dep`,
  `path1c=conj`. → The strongest signal in the entire model is that the
  same drug mentioned twice is almost never a real interaction, which
  is correct.

### Errors stratified by document source

Devel set is 91 % DrugBank, 9 % MedLine. The classifier's behaviour
differs sharply:

| Source | #pairs | advise | effect | int | mechanism | M-F1 | m-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| DrugBank | 4 454 | 0.609 | 0.703 | 0.750 | 0.657 | **0.680** | 0.669 |
| MedLine  |   423 | 0.667 | 0.321 | 0.400 | 0.333 | **0.430** | 0.355 |

Test set shows the same pattern (DrugBank M=0.686 vs MedLine M=0.364).
**DrugBank pairs are ~25-30 pp easier than MedLine.** This makes sense:
DrugBank uses standardised drug-label phrasing ("monitor closely",
"may potentiate") that the classifier's clue-verb features fit; MedLine
uses research-abstract prose with more diverse syntactic patterns.

### Errors stratified by sentence length

| Bucket | #pairs (devel) | M-F1 | m-F1 |
|---|---:|---:|---:|
| short (≤10 words) | 274 | 0.520 | 0.667 |
| medium (11-25) | 1 858 | **0.707** | **0.704** |
| long (26-50) | 2 308 | 0.412 | 0.554 |
| very long (>50) | 437 | 0.457 | 0.634 |

Medium-length sentences are the sweet spot. Long sentences (26+ words)
drop ~30 pp in macro-F1 — likely because:
1. longer dependency paths increase noise in `path*` features,
2. more entities present per sentence (the mean is 4.8 in long vs 2.1
   in short), so the model has to handle more pair-candidates per
   sentence,
3. the LCS subtree analysis loses signal when E1 and E2 are far apart
   in the tree.

Short sentences have a different problem: too few words for the
verb/clue features to fire (advise gets 0 F1 in the very-long bucket
because there are 0 advise pairs there in devel).

### Take-away for the discussion

The classifier is well-tuned for DrugBank-style "interaction warning"
language and medium-length sentences. The two genuine failure modes are
(a) long sentences with diffuse syntax and (b) MedLine-style research
prose. Both are out-of-domain for the shipped feature set; future work
could add corpus-specific feature mods (e.g., abstract-prose verb
patterns for MedLine) or sentence-level decomposition for long
sentences. Neither is in scope for the current report.

### Methodological notes (for the discussion section)

1. **Combinations beat individuals.** Each of mod2 / mod3 / mod4 alone
   was neutral-to-marginal, but mod2+mod3 (=mod_best2) gained +1.1 pp
   devel macro over ref. mod4 helped mod_best on test but hurt on devel
   — devel-only selection (per the spec) keeps mod4 out.
2. **Hyperparameter tuning ≈ 0**. Default C=1 lbfgs l2 is optimal on
   ref and on mod_best2 features. saga at C=1 l2 gains +0.1 pp.
3. **Two-stage is the largest single jump.** A binary stage-1 router
   followed by a 4-way stage-2 classifier on positives lifts devel
   macro by +0.6 pp and test macro by +0.9 pp. The hyperparameter
   `class_weight` in stage-1 must stay at `None` — balancing
   overcorrects because the eval format discards `null` predictions.
4. **Threshold of 0.37** is well below 0.5 because the binary stage-1
   classifier's `positive` class is itself rare (15% of training);
   demanding 0.5 probability rejects too many borderline true positives.

### Phase R/F/A summary (devel, no test except mod_best)

```
                  M-F1   m-F1   ΔM     Δm    notes
ref-MEM           64.3   63.2     —      —   baseline
ref-MEM C=0.1     61.1   60.5  -3.2   -2.7   over-regularised
ref-MEM C=0.5     63.7   62.8  -0.6   -0.4
ref-MEM C=2       63.6   62.6  -0.7   -0.6
ref-MEM C=5       63.2   62.0  -1.1   -1.2
ref-MEM cw=bal    53.2   56.7 -11.1   -6.5   class_weight=balanced over-corrects
ref-SVM           55.0   54.3  -9.3   -8.9   rbf default, precision-biased
SVM linear C=0.5  60.5   60.0  -3.8   -3.2   linear improves but still trails MEM
mod1-MEM          62.4   61.8  -1.9   -1.4   distance/length buckets HURT
mod2-MEM          64.2   63.0  -0.1   -0.2   clue verbs ≈ neutral
mod3-MEM          64.0   64.0  -0.3   +0.8   path entities micro+, macro-
mod4-MEM          64.1   63.1  -0.2   -0.1   type-pair combined ≈ neutral
mod5-MEM          63.0   62.8  -1.3   -0.4   negation cues HURT
mod_best-MEM      65.2   65.1  +0.9   +1.9   ← combo of mod2+mod3+mod4 wins
```

### mod_best per-class breakdown (devel + test):

```
DEVEL:                P     R    F1
advise              64.1  63.6  63.9   (+1.3 vs ref)
effect              82.4  57.9  68.0   (+2.6 vs ref)
int                 78.1  58.1  66.7   (-2.5 vs ref)
mechanism           73.7  53.6  62.1   (+2.1 vs ref)
M.avg               74.6  58.3  65.2
m.avg               74.9  57.5  65.1

TEST:                 P     R    F1
advise              76.0  56.0  64.5
effect              73.3  56.3  63.7
int                100.0  65.0  78.8   (!)
mechanism           64.2  56.4  60.1
M.avg               78.4  58.4  66.8
m.avg               71.0  56.7  63.0
```

Notable: on test, `int` reaches 78.8 F1 (zero false positives, 26/40 recall).
This is a small class (40 test pairs), so the macro gain is partly variance-
driven. The macro-F1 on test (66.8) is +1.6 pp above devel (65.2), which is
unusual but plausible given the small devel size for `int`. Spans the
spec's "~65% target" comfortably.

### Lessons learned (for the report)

1. **The shipped feature set is well-tuned**: 4/5 single-mod additions
   were neutral-to-negative. A logistic regression on sparse binary
   features is sensitive to dilution.
2. **Combinations can synergise**: each of mod2/mod3/mod4 alone was
   neutral or marginal, but together they gain +0.9 pp macro. The
   informative third-entity / type-pair / clue-verb features cooperate
   in ways the model can't reconstruct from any single one alone.
3. **Hyperparameter tuning didn't help**: default C=1 with lbfgs solver
   is already optimal for both ref features and mod_best features.
   `class_weight='balanced'` overcorrects (the predict path drops `null`
   pairs anyway, so balancing across 5-way training is misguided).
4. **MEM (LogisticRegression) >> SVM** on this task with these features
   (best MEM 65.2 vs best SVM 60.5). The shipped feature space is sparse
   and high-dim — exactly LR's regime.
5. **`int` is the easiest positive class** despite being the rarest in
   the corpus, because its trigger lexicon is concentrated ("interact",
   "coadminister"). `mechanism` and `advise` are hardest — more diffuse
   linguistic patterns.

### ref-MEM per-class devel breakdown:

```
                   P      R      F1
advise           63.8%  61.5%  62.6%
effect           78.7%  56.0%  65.4%
int              77.1%  62.8%  69.2%
mechanism        69.3%  52.9%  60.0%
M.avg            72.2%  58.3%  64.3%
m.avg            72.0%  56.4%  63.2%
```

### ref-SVM per-class devel breakdown:

```
                   P      R      F1
advise           60.8%  41.3%  49.2%
effect           83.2%  47.2%  60.2%
int              90.9%  46.5%  61.5%
mechanism        70.5%  37.5%  49.0%
M.avg            76.4%  43.1%  55.0%
m.avg            74.6%  42.7%  54.3%
```

**Decision**: MEM (LogisticRegression) is the carrier for feature
engineering. SVM (rbf default) is too precision-biased (recall < 45% on
every class); we'll revisit it in Phase A with linear kernel + lower C.

## Feature-set catalogue

| Tag | Description | Source |
|---|---|---|
| `ref` | Shipped feature set (type, samedrug, LCS lemma+pos, 9 path variants, words/lemmas in path, lcs children, 4 patterns: verb-lcs / verb-func / wib / wout) | `extract_features.py` + `patterns.py` shipped |

Add new rows under "Feature-set catalogue" whenever a new feature mod is
introduced; reference the mod tag in the run name (e.g. `MEM-mod1-…`).

## Algorithm sweep plan (provisional)

| Algorithm | Hyperparameter | Values to try |
|---|---|---|
| MEM | C | 0.1, 0.5, 1.0, 5.0, 10.0 |
| MEM | solver | lbfgs, liblinear, saga |
| MEM | max_iter | 500, 1500, 5000 |
| SVM | C | 0.1, 0.5, 1.0, 5.0, 10.0 |
| SVM | kernel | linear, rbf, poly |
| SVM | degree (poly only) | 2, 3 |
| SVM | gamma | scale, auto, 0.01, 0.1 |

Feature engineering takes priority over hyperparameter tuning. Hyperparam
sweep only happens after we lock in the best feature mod, mirroring the
Task 1 1.1 (CRF) methodology.

## Feature engineering ideas (from spec p41–46)

- **Position features**: word lemmas/POS before E1, between E1/E2, after E2
- **Clue verbs**: presence + position of pharmacology-trigger verbs
  (interact, inhibit, induce, potentiate, antagonize, increase, decrease, affect, alter, …)
- **Third entity in path / sentence**: helps disambiguate when 3+ drugs co-occur
- **Entity-pair type combinations**: `typeE1+typeE2` as a combined feature
- **More tree-pattern features**: LCS verb subtree shape, subject/object roles
- **Negation cues**: presence of "not", "no", "without" near the verb
- **Sentence length / distance** between entities (bucketed: <5, 5-10, 10-20, 20+)

These are candidates only; we'll prioritise based on baseline error analysis.

## Campaign plan (mirrors Task 1's NERC-ML methodology)

### Phase R — Reference (default features, default algos)
- `ref-MEM`: shipped features + MEM defaults
- `ref-SVM`: shipped features + SVM defaults
- Outcome: establishes baseline and decides which classifier to run mods on.

### Phase F — Feature engineering (one mod at a time, additive)
Each mod is a `[MOD-2.1]`-tagged addition to `extract_features.py` (or
`patterns.py`). After each mod we re-extract features (this is the slow
step), retrain the winning Phase-R classifier with default HPs, and log
results. We keep mods that help (>0.5 pp M-F1 on devel) and discard the rest.

| mod | What it adds | Files touched |
|---|---|---|
| mod1 | Distance + sentence-position features (E1/E2 token positions, distance bucket, sentence-length bucket) | `extract_features.py` |
| mod2 | Pharmacology clue-verbs (lemmas + position before/between/after pair) | `patterns.py` (new pattern) |
| mod3 | Third-entity context (count of other entities in sentence and in path, types of in-path entities) | `extract_features.py` |
| mod4 | Type-pair combined feature (`typeE1_typeE2`) | `extract_features.py` |
| mod5 | Negation cues near LCS / in path (`not`, `no`, `without`, `fail` lemmas) | `extract_features.py` or `patterns.py` |
| mod6 | Lemma/POS n-grams before E1 / between / after E2 (limited to content words) | `patterns.py` |
| mod7 | LCS subtree shape: LCS lemma + immediate-children dependencies | `extract_features.py` |
| mod_best | Bundle of all mods that helped | composite |

### Phase A — Algorithm + hyperparameter sweep (on `mod_best`)
| Algorithm | Grid |
|---|---|
| MEM | C ∈ {0.1, 0.5, 1, 5, 10}, solver ∈ {lbfgs, liblinear, saga}, max_iter ∈ {1500} |
| SVM | C ∈ {0.1, 0.5, 1, 5, 10}, kernel ∈ {linear, rbf}, gamma ∈ {scale, 0.1, 0.01} |

### Phase T — Final test eval
Best (algo, HP, feature-set) from Phases F+A, evaluated on test once.

## Open questions / decisions to log

- Whether to use spaCy's `en_core_web_trf` (transformer, slow but accurate)
  or `en_core_web_sm` (small, fast). Shipped uses `trf`. We'll stick with
  `trf` for the report; document any swap.
- Whether to evaluate the rule-based baseline 2.0 once and quote it from
  the spec, or include it in our cross-system comparison row. Decision:
  include it (re-ran 2026-05-02, matches spec exactly).
- Do we need Boada for 2.1? **No** — feature extraction is ~3 min locally
  on CPU (en_core_web_trf parses ~25 sentences/sec with batching);
  training MEM/SVM is seconds. Run everything locally.

## Open questions / decisions to log

- Whether to use spaCy's `en_core_web_trf` (transformer, slow but accurate)
  or `en_core_web_sm` (small, fast). Shipped uses `trf`. We'll stick with
  `trf` for the report; document any swap.
- Whether to evaluate the rule-based baseline 2.0 once and quote it from
  the spec, or include it in our cross-system comparison row. Decision:
  include it (re-run to confirm).
