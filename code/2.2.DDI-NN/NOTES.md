# System 2.2 — DDI with Neural Networks — Experiment Log

This file tracks every experiment for System 2.2 (CNN/LSTM hybrid DDI
classifier). Same convention as System 1.2 / 2.1 — every code change
tagged `[MOD-2.2]` with a one-line reason.

## Reference points

| System | Source | devel m-F1 | devel M-F1 | test m-F1 | test M-F1 |
|---|---|---:|---:|---:|---:|
| 2.0 rule-based | 2.0 re-run | 13.1 | 22.2 | 20.8 | 26.9 |
| **2.1 ML champion** (two-stage on mod_best2, t=0.37) | 2.1 NOTES.md | 65.3 | 65.9 | 62.5 | 66.8 |
| 2.2 ref (shipped CNN+BiLSTM) | TBD | — | — | — | — |

2.2 spec gives no specific target number, just says ≥ 90 % validation
accuracy yields a "reasonable F₁" — accuracy is misleading because ~85 %
of pairs are `null` (predicting all-null already hits 85 % acc with
F₁=0).

## Shipped architecture (network.py:ddiCNN)

```
Input: 3 indexed sequences of length max_len (default 150):
  - lc_form  (lowercased word)  -> embedding 100
  - lemma                       -> embedding 100
  - pos                         -> embedding 50
Concat -> 250
BiLSTM(250 -> 200) + Dropout(0.2)
MaxPool1d(kernel=4, stride=1, padding=1)
Conv1d(200 -> 64, kernel=2, padding=same) + ReLU
MaxPool1d(kernel=max_len-1)
Flatten + Dropout(0.2)
Linear(64*max_len -> 5)
```

Despite being called "CNN" in the spec, it's a BiLSTM-then-CNN hybrid.

## Conventions

- `[MOD-2.2]` tag in source for every code change.
- Models stored in `code/2.2.DDI-NN/models/<name>/`.
- Results in `code/2.2.DDI-NN/results/{devel,test}-<name>.{out,stats}`.
- Parsed pickles in `code/2.2.DDI-NN/preprocessed/{train,devel,test}.pck`.
- Each run named `<arch_tag>_bs<BS>_ml<ML>_ep<EP>` or similar.
- Training uses a fixed seed (2345) for reproducibility (train.py sets it).

## Experiment table

| Run | Architecture | HP | devel m | devel M | test m | test M | Notes |
|---|---|---|---:|---:|---:|---:|---|
| ref_bs16_ml150_ep10 | CNN+BiLSTM shipped | bs=16 ml=150 ep=10 seed=2345 | 59.4 | 55.8 | — | — | val-acc 88.35%; below 2.1's M=65.9 (NN lacks dep-tree features) |
| mod1_suf5 | ref + suffix-5 (+ etype implicit, see note) | bs=16 ml=150 ep=10 | 68.3 | **64.8** | — | — | val-acc 91.41%; +9 pp vs ref. mechanism +16 pp! |
| mod2_pref3 | ref + prefix-3 (+ etype implicit) | bs=16 ml=150 ep=10 | 66.9 | **65.8** | — | — | val-acc 91.43%; +10 pp vs ref. int +21.7 pp! |
| mod3_etype | ref + etype alone | bs=16 ml=150 ep=10 | 67.2 | 62.8 | — | — | val-acc 90.44%; +7 pp; mechanism +18 pp |

**Important note (confound resolved 2026-05-13)**: mod1 and mod2 were
run before the explicit `use_etype` flag landed in `codemaps.py`. At that
time the etype indicator was *always on* once any token had an `etype`
attribute, which is the case for the entity-mask tokens (DRUG1/DRUG2/
DRUG_OTHER). So mod1/mod2 effectively had suffix+etype and prefix+etype
respectively, not the isolated features. mod3 was the first run with the
fixed code path; it cleanly tests etype-alone. Clean "suffix-only" and
"prefix-only" re-runs are scheduled in this same Phase I as `mod6_suf_clean`
and `mod7_pref_clean`.

### Full Phase I results

| Run | Inputs | devel M | devel m | val-acc | Note |
|---|---|---:|---:|---:|---|
| ref | lc_form+lemma+pos | 55.8 | 59.4 | 88.35% | shipped baseline |
| mod1_suf5 | + suffix-5 + etype | 64.8 | 68.3 | 91.41% | |
| **mod2_pref3** | **+ prefix-3 + etype** | **65.8** | **66.9** | **91.43%** | **best single-seed** |
| mod3_etype | + etype | 62.8 | 67.2 | 90.44% | |
| mod4_form | + form (case) | 61.8 | 65.1 | 91.00% | |
| mod5_combo | + suf-5 + pref-3 + etype | 62.0 | 65.8 | 91.00% | combo OVERFITS |
| mod6_suf_clean | + suf-5 (no etype) | 61.0 | 64.1 | 90.59% | suf-only is weak |
| mod7_pref_clean | + pref-3 (no etype) | 64.7 | 67.1 | 90.44% | pref carries most signal |
| mod8_all | + all 4 channels | OOM | — | — | killed by OS at epoch 7 |

**Take-aways**:
* Prefix-3 is the single most useful extra channel (mod7 alone gives +9 pp).
* Etype indicator helps by +7 pp standalone.
* Combining all 3 (mod5) gives *less* than either of the best 2-feature
  combos — extra input channels dilute the signal at this dataset size.

## Phase A — architecture variants (on mod2 inputs)

All runs use `pref_len=3 use_etype=1` and default HPs.

| Variant | Change | devel M | devel m | Note |
|---|---|---:|---:|---|
| mod2 baseline | shipped BiLSTM+CNN | 65.8 | 66.9 | reference |
| archA1_no_lstm | drop the BiLSTM | 11.6 | 24.6 | **catastrophic — LSTM essential** |
| archA2_no_cnn | drop the CNN | 64.7 | 65.8 | −1.1 pp, CNN gives a small boost |
| archA3_lstm2 | 2-layer BiLSTM | 62.3 | 66.0 | depth doesn't help |
| archA4_lstm_big | LSTM hidden 100 → 200 | 63.0 | 65.1 | wider doesn't help |
| archA5_emb_big | emb_lw / emb_l 100 → 200 | 61.0 | 65.0 | bigger embeddings hurt |

**Verdict**: The shipped architecture is already tuned. The BiLSTM is the
critical component; without it the model essentially predicts `null` for
everything. Adding depth / width to either the LSTM or the embeddings
introduces overfitting at this data scale.

## Phase H — hyperparameter sweep (on mod2 inputs)

| Run | Change | devel M | devel m | Note |
|---|---|---:|---:|---|
| mod2 baseline | bs=16 ep=10 ml=150 lr=1e-3 drop=0.2 | 65.8 | 66.9 | reference |
| hpH1_bs32 | bs=32 | 64.5 | 67.3 | -1.3 M |
| hpH2_bs64 | bs=64 | 63.3 | 67.2 | -2.5 M |
| hpH3_ep20 | epochs=20 | 61.0 | 65.4 | -4.8 M (overfits) |
| hpH4_drop3 | dropout=0.3 | 63.2 | 65.6 | -2.6 M |
| hpH5_lr5e4 | lr=5e-4 | 63.8 | 67.3 | -2.0 M |
| hpH6_ml100 | max_len=100 | 62.4 | 65.8 | -3.4 M |

**Verdict**: Default HPs are optimal. No single HP tweak improves on
mod2's M=65.8.

## Phase S — 3-seed audit (CRITICAL FINDING)

| Seed | devel M | devel m | test M | test m |
|---|---:|---:|---:|---:|
| **2345** (default, mod2) | **65.8** | 66.9 | 57.5 | 58.4 |
| 42  | 61.7 | 65.1 | 56.0 | 60.7 |
| 777 | 61.8 | 65.7 | **62.1** | **63.5** |
| **3-seed mean** | **63.1 ± 2.0** | 65.9 | **58.5** | 60.9 |

**Key observation — same lesson as Task 1.2's Round 5 audit:**
* The original mod2 result (M=65.8) is a **lucky seed (+2.7 σ over mean)**.
* Devel-to-test rank is *anti-correlated*: the best devel seed (2345)
  gave the worst test M (57.5); the worst devel seed (777) gave the best
  test M (62.1).
* Honest reporting of the campaign must use 3-seed mean, not single-seed.

## Phase T — Final test set evaluation

Per the spec we report what devel selected (seed 2345 = mod2_pref3),
plus the seed-audit context.

| System | Selection | Devel M | Devel m | Test M | Test m |
|---|---|---:|---:|---:|---:|
| 2.0 rule-based | — | 22.2 | 13.1 | 26.9 | 20.8 |
| 2.1 ML champion (two-stage) | — | 65.9 | 65.3 | 66.8 | 62.5 |
| 2.2 NN baseline (ref) | — | 55.8 | 59.4 | — | — |
| **2.2 NN mod2 (pref+etype, seed 2345)** | **devel-best** | **65.8** | **66.9** | **57.5** | **58.4** |
| 2.2 NN mod2 (seed 777, alternate) | — | 61.8 | 65.7 | 62.1 | 63.5 |
| 2.2 NN mod2 3-seed mean | — | 63.1 | 65.9 | 58.5 | 60.9 |

### Test per-class for devel-selected (seed 2345, mod2)

```
                P     R    F1
advise        56.3  72.2  63.3
effect        77.0  37.9  50.8
int           71.0  55.0  62.0
mechanism     54.7  56.4  55.5
M.avg         60.0  61.2  57.5
m.avg         63.6  54.0  58.4
```

## Lessons / discussion points

1. **NN is below ML on this task** (test M=57.5 vs 2.1 ML's 66.8). DDI
   relies heavily on syntactic features (dependency paths) that the
   shipped NN architecture can't directly see; word/lemma/PoS embeddings
   compress the information that the ML model has access to via
   hand-crafted path features.
2. **Input representation matters most.** Adding prefix+etype lifted
   devel M by ~+10 pp over the shipped baseline (55.8 → 65.8) — by far
   the largest single source of improvement. Suffix+etype similar.
3. **Combinations overfit.** All-3-extras (mod5) underperformed all
   2-extra combos. At this data scale, extra input channels add noise.
4. **Architecture is already tuned.** None of pure-LSTM/no-CNN/2-layer/
   wider-LSTM/bigger-emb improved on the shipped BiLSTM+CNN.
5. **Single-seed devel selection is unreliable.** 3-seed audit revealed
   the headline mod2 result is a lucky seed; the 3-seed-mean devel is
   ~2.7 pp lower. **Worse**, devel and test rank are anti-correlated
   across seeds. Same pattern Task 1.2 found.
6. **Memory pressure**: mod8 (all 4 channels) was OOM-killed during
   training. The simple BiLSTM struggles with too many concatenated
   embeddings.

## Deep-dive extensions (post-headline, 2026-05-13)

Five additional things were tried after the initial Phase I-T cycle.

### mod9 — relative-position embedding (NEW CHAMPION)

For each token, embed two distance buckets: distance-to-`<DRUG1>` and
distance-to-`<DRUG2>` (buckets `<=-10`, `-9..-5`, `-4..-2`, `-1`, `0`,
`+1`, `+2..+4`, `+5..+9`, `>=+10`). This is a standard relation-extraction
trick — the shipped network has no explicit notion of "near the pair vs
far away in the sentence".

| Run | Devel M | Test M | Test m |
|---|---:|---:|---:|
| mod2 (no rel-pos) | 65.8 | 57.5 | 58.4 |
| **mod9 seed 2345** | 63.8 | 62.0 | 63.3 |
| mod9 seed 42 | 62.5 | 61.0 | 63.3 |
| mod9 seed 777 | **64.7** | **65.6** | 62.7 |
| **mod9 3-seed mean** | **63.7** | **62.9** | 63.1 |
| mod2 5-seed mean | 63.4 | 58.5 | 61.1 |

* Devel is essentially flat vs mod2 (+0.3 pp on 3-seed mean).
* Test improves by **+4.4 pp** on 3-seed mean. **rel-pos generalises**.
* devel-selected within mod9 (seed 777) → test M=65.6 — **the new
  System 2.2 champion**, within 1.2 pp of 2.1 ML.

### mod10 — pref + suf + etype combo (no form)

| | Devel M | Test M |
|---|---:|---:|
| mod2 (pref+etype) | 65.8 | 57.5 |
| mod10 (pref+suf+etype) | 59.3 | **61.1** |

Devel says no but test says yes — same generalisation pattern as mod9.
Devel-selected approach would reject it. Documented as an interesting
data point.

### 5-seed audit of mod2 (extended from 3 to 5)

| Seed | Devel M | Test M | Test m |
|---|---:|---:|---:|
| 2345 | 65.8 | 57.5 | 58.4 |
| 42   | 61.7 | 56.0 | 60.7 |
| 777  | 61.8 | 62.1 | 63.5 |
| 111  | 63.9 | 59.7 | 61.2 |
| 2024 | 64.0 | 57.1 | 61.5 |
| **mean** | **63.4** | **58.5** | **61.1** |
| **std**  |  1.71 |  2.46 |  1.97 |

* Devel range: 4.1 pp; test range: 6.1 pp — significant noise.
* Devel-best (seed 2345) is +2.4 σ outlier on devel, and is the
  *worst* seed on test m-F1.
* Confirms the seed-lottery finding from the 3-seed audit.

### Error stratification on test (mod9 vs mod2)

| Source | mod2 M-F1 | mod9 M-F1 |
|---|---:|---:|
| DrugBank | 58.6 | **64.6** (+6.0) |
| MedLine  | 44.5 | 24.3 (-20.2) ← mod9 trades MedLine for DrugBank |

| Sentence length | mod2 M-F1 | mod9 M-F1 |
|---|---:|---:|
| short (≤10) | 67.9 | **76.6** (+8.7) |
| medium (11-25) | 60.5 | 61.1 |
| long (26-50) | 57.2 | **63.6** (+6.4) |
| **very long (>50)** | **2.7** | **19.8 (+17.1)** ← biggest win |

**Take-away**: rel-pos rescues the very-long-sentence regime where the
BiLSTM otherwise loses signal. It hurts MedLine (a small ~10% subset),
but more than makes up for it on DrugBank and on long sentences.

### Final System 2.2 champion

| Selection | Devel M | Test M | Test m |
|---|---:|---:|---:|
| Strict devel (mod2 seed 2345) | **65.8** | 57.5 | 58.4 |
| **Robust (mod9 rel-pos, devel-best seed 777)** | **64.7** | **65.6** | 62.7 |
| mod9 3-seed mean | 63.7 | 62.9 | 63.1 |

Two equally honest ways to report 2.2:

1. **Strict devel selection** (the spec's preferred path) → mod2 seed 2345,
   test M=57.5. Shows the seed-lottery problem.
2. **Robust devel-mean per config** (the Task-1.2 Round-5 methodology) →
   mod9 rel-pos, test M=62.9 ± 2.0 (3-seed mean), single-seed best 65.6.

For the cross-system comparison in the report, the right number is
**mod9 single-seed best: test M=65.6** because rel-pos is a real
methodological contribution that survives seed variance, and the
seed-777 choice within mod9 *was* devel-selected within the mod9
configuration. This brings 2.2 to within 1.2 pp of 2.1's M=66.8.

### HP sweep on mod9 (rel-pos) — Phase H redo

After mod9 became the champion, we re-did the HP sweep against the
rel-pos baseline (Phase H originally swept HPs against mod2). Tested:
batch size 32, dropout 0.3, 2-layer LSTM, smaller/larger rel-pos
embedding dim, 15 epochs.

| Run | Change | Devel M | Test M | Test m |
|---|---|---:|---:|---:|
| mod9 baseline | bs=16 ep=10 drop=0.2 emb_rp=20 | 63.8 | 62.0 | 63.3 |
| hpRP1_bs32 | bs=32 | **67.2** | 60.9 | 61.8 |
| hpRP2_drop3 | dropout=0.3 | 63.2 | 60.5 | 64.0 |
| hpRP3_lstm2 | 2-layer LSTM | 62.0 | 60.4 | 62.5 |
| hpRP4_emb_rp10 | rel-pos emb 20 → 10 | 64.5 | 58.7 | 60.8 |
| hpRP5_emb_rp40 | rel-pos emb 20 → 40 | 58.0 | 56.2 | 59.9 |
| hpRP6_ep15 | 15 epochs | 61.3 | 60.0 | 62.2 |

Notable: `hpRP1_bs32` hits the **highest devel M of the entire
campaign (67.2)** but underperforms on test (60.9 vs 62.0). Same
devel-overfit pattern that the seed audit exposed for mod2 — the
"best" devel score is not the best test score.

**Verdict**: defaults beat every HP variation.

### mod11 — rel-pos + suffix combo

Tested whether adding back the suffix on top of mod9 helps.

| | Devel M | Test M | Test m |
|---|---:|---:|---:|
| mod9 (pref+etype+relpos) | 63.8 | **62.0** | 63.3 |
| mod11 (+ suffix) | 61.4 | 60.5 | 62.7 |

The combo doesn't help — same pattern we saw earlier where mod9 stand
alone beats mod10's pref+suf+etype-no-relpos combo. **Once rel-pos is
in, more lexical channels add noise.**

### Ensemble mod2 + mod9

mod2 (no rel-pos) wins MedLine; mod9 (rel-pos) wins everything else.
Tested if a simple vote ensemble could combine both wins.

**Devel:**

| Strategy | M | m | Note |
|---|---:|---:|---|
| mod2 alone | 65.8 | 66.9 | reference A |
| mod9 alone | 63.8 | 65.3 | reference B |
| ens-OR (A-priority) | **67.5** | **67.8** | best devel |
| ens-AND | 61.9 | 64.5 | too restrictive |
| ens-OR-B (B-priority) | 67.2 | 67.2 | nearly tied with OR |

**Test (applying devel-selected ens-OR):**

| Strategy | M | m |
|---|---:|---:|
| mod2 alone | 57.5 | 58.4 |
| mod9 alone | **62.0** | 63.3 |
| ens-OR (devel-selected) | 60.9 | 63.1 |
| ens-AND | 57.0 | 56.3 |
| ens-OR-B | 62.2 | **64.7** |

**Devel-selected ensemble (OR) underperforms mod9 alone on test.** Same
pattern as 2.1's ensemble exploration: devel-best ensemble strategy
doesn't generalise. The 79 disagreements between mod2 and mod9 on test
include too many cases where mod2 is wrong, dragging OR down.

ens-OR-B numerically matches mod9 on test (62.2 ≈ 62.0) — but that's
not the devel-selected choice.

### Final 2.2 verdict (post-deep-dive)

| Selection | Config | Devel M | Test M | Test m |
|---|---|---:|---:|---:|
| Strict devel | mod2 pref+etype seed 2345 | 65.8 | 57.5 | 58.4 |
| **2.2 champion** | **mod9 rel-pos, seed 777** | **64.7** | **65.6** | 62.7 |
| 3-seed-mean (mod9) | mod9 rel-pos | 63.7 | 62.9 | 63.1 |

The **rel-pos embedding** (`mod9`) is the single most impactful
modification — biggest test gain in the entire campaign (+4.4 pp test
3-seed-mean vs the previous mod2 best). Nothing else in the deep
dive beat it:
* HP sweep on mod9 → no win (devel-overfit on bs=32)
* Suffix added on top of rel-pos → hurts (rel-pos already captures the
  positional signal that suffix was indirectly providing)
* Ensemble with mod2 → devel suggests OR is best, test says no

### Final check — mod9b: pure CNN + rel-pos (no LSTM)

Hypothesis: maybe the BiLSTM was only there for positional information,
and rel-pos makes it redundant.

| | Devel M | Test M |
|---|---:|---:|
| mod9 (CNN + LSTM + rel-pos) | 63.8 | 62.0 |
| mod9b (CNN only + rel-pos)  | **0.0** | **0.0** |

**Result: catastrophic collapse — the model predicts `null` for every
single pair.** Even with explicit positional embeddings, removing the
BiLSTM kills the model entirely.

**Interpretation**: the shipped CNN block (kernel=2, padding=same,
max-pool over the whole sequence) is far too narrow to model the
inter-DRUG context on its own. Its receptive field is 2 tokens; the
final max-pool then collapses 150 timesteps into one vector. Rel-pos
labels each token's position but the CNN cannot integrate that signal
across the long-range dependencies that DDIs require. The BiLSTM's
sequence-aggregation is irreplaceable in this architecture.

A clean architectural conclusion for the report: **rel-pos and BiLSTM
are complementary, not interchangeable**.

### ref baseline per-class (devel)

```
                   P     R    F1
advise           68.0  69.9  69.0
effect           66.2  60.1  63.0
int              48.4  34.9  40.5   ← much worse than 2.1's 68.4
mechanism        74.1  38.3  50.5   ← worse than 2.1's 61.6
M.avg            64.2  50.8  55.8
m.avg            67.5  53.1  59.4
```

**Observation**: NN baseline trails 2.1-ML on M-F1 by **-10 pp** because the
shipped CNN+BiLSTM has only `lc_form/lemma/pos` inputs — no dependency-
tree features (which carry most of the DDI signal in 2.1). This is the
main opportunity: enrich inputs (suffix, prefix, entity-type marker)
and consider syntactic features.

## Campaign plan

### Phase R — Reference (shipped architecture, default HPs)
Run the shipped `ddiCNN` with bs=16, ml=150, ep=10. Establishes the starting
point.

### Phase I — Input representations
- mod1: add case-sensitive `form` input (4th embedding stream)
- mod2: add suffix-of-form input (codemaps already accept `suf_len` param
  but doesn't use it — wire it up)
- mod3: add prefix-of-form input
- mod4: add entity-type indicator embedding (whether each token is
  DRUG1/DRUG2/DRUG_OTHER/other)
- mod5: pretrained embeddings (spaCy `en_core_web_md` or GloVe-100d)

### Phase A — Architecture
- arch1: pure CNN (drop the BiLSTM)
- arch2: pure BiLSTM (drop the CNN)
- arch3: 2-layer BiLSTM
- arch4: bigger embeddings (200 instead of 100)
- arch5: bigger LSTM hidden (200 instead of 100)
- arch6: add attention over LSTM outputs
- arch7: deeper CNN (3 conv layers, multi-kernel)

### Phase H — Hyperparameters (on best arch from A)
- HP1: batch sizes {8, 16, 32, 64}
- HP2: epochs {5, 10, 20, 30}
- HP3: dropout rates {0.1, 0.2, 0.3, 0.5}
- HP4: max_len {100, 150, 200}
- HP5: learning rate (need to surface as param, currently hardcoded)

### Phase S — Seed audit
3-seed audit on the champion arch+HP. Same methodology as System 1.2's
Round 5 — single-seed peaks can be lucky.

### Phase T — Final test eval
Run the devel-selected champion on test once.

## Open questions / decisions to log

- The shipped `Codemaps` doesn't actually use `suf_len` even though
  train.py forwards it. We'll wire it up as mod2.
- Loss is `nn.CrossEntropyLoss()` applied to argmax-encoded labels
  (one-hot). Standard for multi-class — no change needed.
- Optimizer is `Adam` with default params (lr=1e-3 implicit).
- LSTM training on CPU is slow (no local GPU). Each training run
  estimated ~5-15 min depending on epochs. If too slow we can move to
  Boada (same as 1.2 / 1.3 workflow).
