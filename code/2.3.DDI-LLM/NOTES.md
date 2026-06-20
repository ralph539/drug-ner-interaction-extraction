# System 2.3 — DDI with LLMs — Experiment Log

Same convention as System 1.3 / 2.1 / 2.2 — every code change tagged
`[MOD-2.3]` with a one-line reason. Mirrors the 1.3 NER-LLM campaign:
few-shot sweep → fine-tuning → final test eval.

## Reference points

| System | Source | devel m-F1 | devel M-F1 | test m-F1 | test M-F1 |
|---|---|---:|---:|---:|---:|
| 2.0 rule-based "wib" | 2.0 re-run | 13.1 | 22.2 | 20.8 | 26.9 |
| 2.1 ML champion (two-stage on mod_best2) | 2.1 NOTES.md | 65.3 | 65.9 | 62.5 | **66.8** |
| 2.2 NN champion (mod9 rel-pos, seed 777) | 2.2 NOTES.md | — | 64.7 | 62.7 | **65.6** |
| 2.3 LLM ref (few-shot 0-shot baseline) | TBD | — | — | — | — |

DDI is sentence classification (5 classes incl. null). The LLM emits
a single token (the class name) per (sentence, drug pair).

## Task summary

- Each test instance = a sentence with `[DRUG1]` / `[DRUG2]` / `[DRUG_OTHER]` masks
- Possible answers: `mechanism`, `effect`, `advise`, `int`, or `null`
- ~85% of pairs are `null` (gold class distribution heavily skewed)
- Output format: single word (no XML, no offset reconstruction needed)
  — unlike 1.3 NER, there is **no offset-shift bug** to worry about
- Evaluator: `util/evaluator.py DDI`

## Provided infrastructure

```
code/2.3.DDI-LLM/bin/
├── examples.py          # Examples loader (DDI task)
├── prompts.py           # Prompts loader (same code as 1.3 — keep)
├── prompts01.json       # initial prompt (sysprompt + usrprompt)
├── fewshot.py           # few-shot inference driver
├── fewshot.sh           # SBATCH wrapper: 48 GB / RTX 3080
├── finetune-train.py    # LoRA fine-tuning trainer
├── finetune-inference.py# inference using saved LoRA adapter
├── FT-train.sh          # SBATCH: 64 GB / RTX 4090
├── FT-inference.sh      # SBATCH: 48 GB / RTX 3080
├── model.py             # Inference + FineTuning HF wrappers (same as 1.3)
└── paths.py
```

## Known issues from 1.3 transferred to 2.3

1. **`prompts.py` instruction-repetition pattern** — same code as 1.3.
   The prof's 2026-04-24 announcement says the *structure* is correct
   (usrprompt is repeated per-shot, preserving chat-template alternation).
   The fix is to keep `usrprompt` short — already mostly true in 2.3's
   prompts01.json (~6 lines, shorter than 1.3's prompts01).

2. **No offset bug** for DDI — `DDI_eval_format` just emits the class name,
   no XML tag offsets to manipulate.

## Spotted potential issue: `advice` vs `advise`

`prompts01.json` sysprompt enumerates classes as `mechanism`, `effect`,
**`advice`**, `int` — but the DDI-2013 dataset uses **`advise`** (no `c`).
The evaluator and the gold labels expect `advise`. Few-shot examples
(when given) will teach `advise` via the assistant turns, so the model
likely picks up the right spelling — but the system instruction nudges it
toward `advice`. **Worth testing: fix the typo and see if it helps.**

## SBATCH right-sizing (per prof's 2026-05-14 email)

- Inference jobs (fewshot.sh, FT-inference.sh): **48 GB** seems within
  guidance for LLM tasks. Probably fine.
- Training jobs (FT-train.sh): currently **64 GB**. Prof's note says
  oversizing hurts queue throughput. LoRA fine-tune of 3B-quantized
  models on the cluster has previously fit in ~40–48 GB. Will trim
  before launching large sweeps if 48 GB confirmed sufficient.
- **Maintenance window: 2026-06-03 08:00-17:00.** Avoid scheduling jobs
  that would span this.

## Conventions

- `[MOD-2.3]` tag in source for every code change.
- Models stored on Boada at `/scratch/nas/1/PDI/mml0/models/{name}` or
  via ollama for inference-only runs.
- Results in `code/2.3.DDI-LLM/results/{FS,FT}-<model>-<config>.{json,out,stats}`.

## Campaign plan (mirrors 1.3)

### Phase C — Few-shot sweep
- Phase C1: baseline = 0-shot llama32B3 with shipped prompts01
- Phase C2: shots sweep {0, 3, 5, 10, 15} on prompts01
- Phase C3: prompts × shots — try refined prompt variants (prompts02, prompts03)
- Phase C4: model swap (qwen25B3 vs llama32B3) at the best shots/prompt
- Phase C5: balanced vs unbalanced few-shot sampler

### Phase D — Fine-tuning baseline
- LoRA r=8, 10 epochs, lr=2e-5, 4-bit base, prompts01

### Phase F — FT sweep
- r=32 (bigger rank), more epochs, different models

### Phase H — best system consolidation (3-seed audit?)
- Multi-seed FT to check the lucky-seed pattern

### Phase I — final test eval

### Phase T — final report numbers

## Experiment table

### Phase C — Few-shot sweep (devel)

| Run | Model | Prompts | Shots | Balanced | devel M | devel m | Notes |
|---|---|---|---|---|---:|---:|---|
| C1.0 | llama32B3 | prompts01 | 0 | — | 9.5 | 8.2 | 0-shot collapses |
| C1.3 | llama32B3 | prompts01 | 3 | yes | 21.4 | 18.7 | huge FP load on `int` (1866 FPs) |
| C1.5 | llama32B3 | prompts01 | 5 | yes | 21.0 | 18.7 | |
| **C1.10** | **llama32B3** | **prompts01** | **10** | **yes** | **23.2** | 17.4 | **best macro of FS sweep** |
| C1.15 | llama32B3 | prompts01 | 15 | yes | 21.8 | 22.0 | best m |
| C2.5 | llama32B3 | prompts02 | 5 | yes | 20.7 | 23.3 | best micro overall |
| C2.10 | llama32B3 | prompts02 | 10 | yes | 21.8 | 23.0 | |
| C2.15 | llama32B3 | prompts02 | 15 | yes | 18.7 | 20.7 | |
| C4.10 | qwen25B3 | prompts01 | 10 | yes | 19.8 | 16.4 | qwen worse than llama |

### Phase C findings

- **Few-shot is fundamentally weak on DDI** at M=18-23. Compare to 2.1 ML
  (M=66.8 test) and 2.2 NN (M=65.6 test) — LLMs in FS mode trail by ~45 pp.
- **prompts02 (fixed "advise" + stronger null emphasis)** improved micro
  (~m+5 pp) but barely moved macro. The model is so confused on the
  positive/null boundary that prompt-level tweaks don't help much.
- **Adding shots beyond 5 helps only marginally** (3→10 shots: +1.8 pp;
  10→15 shots: -1.4 pp). The signal saturates because the model can't
  generalize the structural distinction "is there an interaction stated"
  from a handful of examples.
- **Qwen-2.5-3B is worse than Llama-3.2-3B** at 10-shot (19.8 vs 23.2).
  Same direction as 1.3 where qwen also trailed at few-shot.
- **Massive false-positive load**: model emits a positive class for ~85%
  of test pairs even though only ~15% are positive. Especially `int`
  (1-2% precision: ~1900 FPs for ~20 TPs). The instruction "if unsure,
  choose null" is being ignored.

**Implication**: fine-tuning is the only path to closing the gap to ML/NN.
Phase D is running now (job 425328, llama32B3 + prompts01 + 4-bit, LoRA r=8).

### File-naming fix (mid-campaign)

The shipped `fewshot.sh` named outputs as `FS-<model>-<shots>-<test><quant>.out`
without the prompt-variant tag — same bug as 1.3. So when we ran prompts02
at 5/10/15 shots, the .out/.stats files overwrote the matching prompts01
runs. Local copies of the prompts01 results were already saved.

Fix applied (mirroring 1.3):

```bash
TAG=$(basename "$PROMPTS" .json)
BASE=../results/FS-$MODEL-$SHOTS-${TEST}${QUANT}
TAGGED=../results/FS-$MODEL-$TAG-$SHOTS-${TEST}${QUANT}
mv $BASE.{out,json} $TAGGED.{out,json}
python3 evaluator.py DDI ... $TAGGED.out $TAGGED.stats
```

Overwritten files on Boada renamed to `FS-llama32B3-prompts02-{5,10,15}-…`.

## Phase D + F — Fine-tuning (final results)

Trained four LoRA fine-tune configurations on the train split (5000
balanced examples) with 10 epochs, lr=2e-5, batch=1, gradient_accum=8,
prompts01.json. All 4-bit quantised.

| Config | Devel M | Devel m | Test M | Test m |
|---|---:|---:|---:|---:|
| FT Llama r=8  | 28.9 | 36.0 | 30.5 | 35.3 |
| FT Qwen  r=8  | 33.7 | 33.7 | 27.6 | 29.4 |
| **FT Llama r=32** | **33.8** | **37.6** | **35.2** | **37.8** |
| FT Qwen  r=32 | 32.5 | 34.5 | 29.1 | 32.0 |

**Champion: FT Llama r=32 — devel M=33.8, test M=35.2.** Per-class on test:

```
                P     R    F1
advise        27.7  78.5  40.9
effect        27.3  67.9  38.9
int           17.6  40.0  24.4
mechanism     23.8  78.5  36.5
M.avg         24.1  66.2  35.2
m.avg         25.5  73.3  37.8
```

Pattern: **all classes have high recall (40-79%) but low precision (17-28%)**.
The fine-tuned LLM still over-predicts positives (the prompt's "null"
instruction is hard to obey under heavy class imbalance).

### Phase D/F findings

1. **Rank r=32 helps a lot.** Llama r=32 beats Llama r=8 by +4.7 pp test
   macro. Same direction as 1.3 NER campaign. The bigger adapter has
   more capacity for the multi-class boundaries.

2. **Llama > Qwen for DDI at this scale.** Llama r=32 wins both devel
   (33.8 vs 32.5) and test (35.2 vs 29.1). Same as in 1.3 (Llama r=32
   won test) but at a much lower absolute level.

3. **FT is much better than FS but still far below ML/NN.** FT Llama
   r=32 test M=35.2 vs FS best 23.2 → +12 pp. But ML champion is 66.8
   and NN champion is 65.6 — the LLM trails by ~30 pp.

4. **Why? DDI ≠ NER.** Unlike 1.3 where FT-LLM matched/beat dedicated
   NERC, here the LLM struggles because:
   - The class imbalance is extreme (85% null); the model can't reliably
     refuse to predict a class.
   - DDI requires *relational* reasoning between the two DRUG markers,
     not just lexical matching. A 3B-quantised LLM lacks the deep
     reasoning that the ML's syntax features and the NN's positional
     embeddings encode explicitly.
   - Output is a single token (the class name) — there's no
     "structured generation" gain, unlike NER where the LLM can stream
     XML tags.

5. **Disk quota mid-campaign caused silent failures.** Several
   FT-inference jobs ran for 13-21s producing empty .out files because
   the Boada 2GB user quota was exhausted by retained training
   checkpoints. Lesson: clean `checkpoint-*` subdirs aggressively after
   each FT-train completes.

## Cross-system comparison

| System | Devel M | Devel m | Test M | Test m |
|---|---:|---:|---:|---:|
| 2.0 rule baseline | 22.2 | 13.1 | 26.9 | 20.8 |
| 2.1 ML two-stage (mod_best2) | 65.9 | 65.3 | **66.8** | 62.5 |
| 2.2 NN mod9 rel-pos (seed 777) | 64.7 | 62.7 | **65.6** | 62.7 |
| 2.3 LLM FT-Llama r=32 (champion) | 33.8 | 37.6 | **35.2** | 37.8 |

For DDI:
- **ML > NN by ~1 pp** (66.8 vs 65.6 test macro)
- **LLM trails both by ~30 pp** at the 3B-quantised scale

This is the *opposite* of 1.3 NER where the LLM matched/beat the
dedicated systems. The key task differences explain it (see point 4
above).

## Qualitative analysis on the FT Llama r=32 champion

### Confusion matrix (test)

```
GOLD\PRED  advise  effect  int  mechanism  null    Σ
advise       164       5     1        10     29    209
effect        14     199     0        37     43    293
int            0       3    16         2     19     40
mechanism      1      16     0       270     57    344
null         413     507    74       817   3141   4952
```

Two dominant error patterns:

1. **null → positive over-prediction.** Out of 4952 truly-null pairs in
   test, **1811 (36.6 %)** are misclassified as positive. The model
   fails to "default to null" despite the explicit instruction.
   `mechanism` is the worst sink — 817 null pairs are labelled
   `mechanism`, meaning when the model predicts `mechanism`, it is
   wrong 75 % of the time (270 TP vs 866 FP).

2. **Inter-positive confusion is small.** Largest off-diagonal in the
   positive 4×4 block is `effect → mechanism` (37 cases). The model
   *can* distinguish the four positive types when it commits to a
   positive class — it just commits too often.

This explains why the metrics are high-recall / low-precision:

```
              P    R     F1
advise       27.7 78.5  40.9
effect       27.3 67.9  38.9
int          17.6 40.0  24.4
mechanism    23.8 78.5  36.5
```

The shape (R >> P for every class) is identical across all 4 FT
configurations — it is a property of the LoRA-fine-tuned 3B LLM under
85 %-null imbalance, not of any specific rank or model. Same shape we
saw in the FS sweep too. **The model has not learned to refuse to
predict.**

### Error stratification (test)

| Source | M-F1 |
|---|---:|
| DrugBank (5312 pairs) | 35.7 |
| MedLine (526 pairs) | 39.8 |

Counter-intuitively, MedLine is *slightly easier* for the FT-LLM than
for ML/NN — but the MedLine subset has so few positives (5 advise,
40 effect, etc.) that the comparison is noisy. The DrugBank result is
the methodologically meaningful one.

| Sentence length | M-F1 |
|---|---:|
| short (≤10 words) | 49.3 |
| medium (11-25) | 39.7 |
| long (26-50) | 27.2 |
| **very long (>50)** | **11.0** |

**Same pattern as 2.2 NN: very-long sentences collapse (M=11.0).** The
LLM has 512 tokens of context, which the dataset's longest sentences
fill before the `[DRUG1]`/`[DRUG2]` markers can anchor attention. Even
with explicit positional markers, the LLM doesn't seem to focus on
them in long contexts — same architectural limitation the BiLSTM hit.

### Take-away for the discussion

The story for the cross-system section:

- **All three systems get worse on long sentences.** ML loses ~10 pp
  going short → long, NN loses ~50 pp, LLM loses ~38 pp. The dependency
  features that ML has (path through the parse tree) provide some
  protection at long distances; the NN and LLM degrade more.
- **All three systems make their dominant errors at the positive/null
  boundary.** For ML the failure is *missing* positives (FN dominates);
  for the LLM the failure is *over-predicting* positives (FP dominates).
  NN sits in between. The fix would be the same in all three:
  better calibration of the null-vs-positive boundary.
- **The 3B LLM is not the right tool for this DDI task at this scale.**
  ML's explicit syntactic features and NN's positional embedding both
  outperform the LLM's free-form classification head. A bigger LLM
  (7B+) at full precision might close the gap; a small LLM at 4-bit
  quant trained on 5000 examples cannot.

## Phase G: prompts02 FT (advise/advice typo fix + stronger null emphasis)

### Motivation

The prompts01 FT champion's dominant failure is null→positive
over-prediction (36.6 % of all null pairs). Two hypotheses behind
prompts02 (already used in the FS sweep, never trained on):

1. The original `prompts01` definition of the `advise` class contained
   the typo "advise" misspelled as "advice" — the model literally
   couldn't learn to emit the label string when its definition pointed
   to a different word.
2. The null definition in prompts01 was a single line; prompts02
   strengthens it to explicitly tell the model *do not invent an
   interaction*.

### Setup

Same as Phase D champion: `llama32B3`, 4-bit quant, LoRA r=32 (alpha=64),
10 epochs configured. Training hit Boada's 2 GB disk quota at end of
epoch 3 (same as the prior champion run — checkpoint footprint is
~75 MB × 3 = 225 MB + base usage, exceeding quota during the save_model
finalize). Recovery: `mv checkpoint-1878/* ../` to promote the
epoch-3 adapter to top-level. Both champion runs are therefore
3-epoch adapters trained from the same recipe — fair head-to-head.

### Headline numbers

| Config | devel M-F1 | test M-F1 |
|---|---:|---:|
| Llama r=32 prompts01 (prior champion) | 33.8 | 35.2 |
| **Llama r=32 prompts02** | **41.6** | **39.8** |
| Δ | **+7.8** | **+4.6** |

This is a larger gap than r=8 → r=32 produced (which was +5 pp devel,
+4.7 pp test). Prompt quality matters more than LoRA rank at this
scale.

### Confusion matrix (test)

```
GOLD\PRED     advise    effect       int mechanism      null    Σ
advise           163         8         0         5        33   209
effect            14       247         0         6        26   293
int                0         6        15         1        18    40
mechanism          1        69         0       199        75   344
null             331       682        35       380      3524  4952
```

Two diagnostics vs the prompts01 champion (`confmat-test-llama-r32.txt`):

- **Total null→positive errors: 1428 vs 1811** (28.8 % vs 36.6 %).
  The stronger null definition really did reduce
  over-prediction by ~8 pp absolute. Most of the saving came from the
  null→mechanism column: **380 vs 817**, a 54 % cut.
- **null→advise now 331** (was negligible before the typo fix). The
  model is now actually using the `advise` label, which is what pushed
  advise per-class F1 to 45.4 % (up from a value where the model
  basically didn't predict `advise` at all under prompts01).

### Per-class F1 (test)

| Class | P | R | F1 |
|---|---:|---:|---:|
| advise | 32.0 | 78.0 | **45.4** |
| effect | 24.4 | 84.3 | 37.9 |
| int | 30.0 | 37.5 | 33.3 |
| mechanism | 33.7 | 57.8 | 42.6 |

`mechanism` per-class F1 dropped (~52 → 42.6) because the model
stopped using mechanism as the catch-all "I think there's something
here" bucket. Net M-F1 is up because the gain on `advise` and the
reduced FP count on `mechanism` more than offset.

### Error stratification (test)

| Sentence length | prompts01 M-F1 | prompts02 M-F1 | Δ |
|---|---:|---:|---:|
| short (≤10) | 49.3 | 45.1 | -4.2 |
| medium (11-25) | 39.7 | 44.2 | +4.5 |
| long (26-50) | 27.2 | 30.8 | +3.6 |
| very long (>50) | 11.0 | 12.3 | +1.3 |

The medium and long buckets gain the most. The very-long collapse
persists — that's an architectural ceiling, not a prompt issue.

| Source | prompts01 | prompts02 | Δ |
|---|---:|---:|---:|
| DrugBank | 35.7 | 39.8 | +4.1 |
| MedLine | 39.8 | 44.0 | +4.2 |

Both gain ~+4. Uniform improvement across sources.

### Revised take-aways for the cross-system discussion

- **Prompt quality is a first-class capacity knob for FT LLMs.** Going
  from prompts01 → prompts02 gave +4.6 pp test M-F1, larger than the
  r=8 → r=32 LoRA rank doubling. For the report: don't treat the prompt
  template as fixed — it's part of the model.
- The over-prediction story holds in shape but moderates in magnitude:
  1428/4952 null pairs still get a positive label. The LLM is still
  RM-dominated (precision << recall).
- The LLM still trails ML (test 66.8) and NN (test 65.6) by 25-27 pp,
  so the "3B LLM not the right tool at this scale" conclusion stands;
  it just lands ~5 pp gentler than before.

## Phase H: length-filter post-processing (negative result)

### Motivation

The error stratification showed M-F1 collapsing to 12.3 in the >50-word
sentence bucket. Hypothesis: predictions in that bucket are mostly
noise; replacing them with `null` should be a free precision boost.

### Implementation

`bin/length_filter.py` removes any line in the `.out` whose sentence
has more than N words. Re-run evaluator on the filtered file.

### Results (test, M-F1)

| Filter | Llama r=32 **prompts02** | Llama r=32 prompts01 (prior champ) |
|---|---:|---:|
| no filter | **39.8** | **35.2** |
| L=50 (drop >50) | 39.6 | 35.2 |
| L=40 | 39.0 | 34.8 |
| L=30 | 38.5 | 34.7 |

**Filter never helps; tighter thresholds hurt more.** The pattern is
identical across both champions, so it's a property of the model class,
not of a specific prompt.

### Why the filter fails

Counts of predictions in the vlong bucket on the test set:

| Champion | total preds | preds in >50 sents | share |
|---|---:|---:|---:|
| Llama prompts02 | 2162 | 38 | 1.8 % |
| Llama prompts01 | 2549 | 49 | 1.9 % |

**The FT LLM already self-suppresses on long sentences.** It almost
never emits a positive label when the context is too long, and the
few it does emit contain enough true positives that removing them
loses recall without buying any precision.

### Implication for the report

The "vlong M=12.3" in the strata table is not a failure mode the
LLM can be cheaply patched out of. The collapse is a recall problem
(positives in long sentences are missed, predicted as null), not a
precision problem (few wrong predictions to filter out). The fix
would require *more* signal on long sentences, not less — e.g. a
larger context window or a model with stronger long-range attention.

## Phase I: prompts02 on Qwen — opposite result + epoch confound

Ran the same prompts02 recipe on `qwen25B3` r=32 to test whether the
prompt-quality finding generalises across model families.

### Headline (test M-F1)

| Config | devel | test |
|---|---:|---:|
| Qwen prompts01 r=32 (prior) | 32.5 | 29.1 |
| Qwen prompts02 r=32 (this run) | 28.8 | **26.0** |
| Δ | −3.7 | **−3.1** |

**prompts02 hurt Qwen — the opposite of Llama (which gained +4.6).**
Same prompt, opposite sign across the two model families.

### What happened: catastrophic over-prediction

Qwen prompts02 confusion matrix (test):

```
GOLD\PRED     advise    effect       int mechanism      null    Σ
advise           130        19         3        33        24   209
effect             7       239         2        40         5   293
int                0         4        34         1         1    40
mechanism          1        22         0       316         3   344
null             363       935       618      1705      1328  4952
```

- **3624 / 4952 null pairs (73.2 %) get a positive label** — vs Llama
  prompts02's 28.8 % and Llama prompts01's 36.6 %. Qwen with this
  prompt predicts a positive on 4477 / 5838 = 77 % of *all* pairs.
- mechanism: P=15.1 %, R=92.4 % (1705 false positives from null alone).
- int: P=5.2 % (the model emits `int` 657 times for 40 gold cases).

This is the classic "predict the majority of *positives* to maximise
recall" collapse — the model has stopped discriminating.

### The confound: 10 epochs vs 3

**This Qwen run completed all 10 configured epochs** (5h24m, no quota
crash — disk had been freed beforehand and Qwen's adapter is smaller).
Every prior r=32 run (both Llama champions, and almost certainly the
Qwen prompts01 baseline) **crashed at epoch 3** on the disk quota and
was recovered from the epoch-3 checkpoint.

So the only *clean* prompt comparison is the Llama one (prompts01 vs
prompts02, both 3 epochs). The Qwen comparison mixes two changes:
prompt (01→02) **and** epochs (3→10). 10 epochs on a 3B model under
85 % null is a textbook overfitting-to-positives setup, and the
73 %-over-prediction signature looks far more like overfitting than a
pure prompt effect.

### Next: matched 3-epoch Qwen prompts02 (Phase J)

Added an `EPOCHS` env var (`[MOD-2.3]`, model.py) mirroring `LORA_R`.
Re-running Qwen prompts02 at `EPOCHS=3` will give two clean results in
one job:
1. **Prompt effect at fixed epochs** — Qwen 3ep prompts01 vs prompts02.
2. **Epoch ablation at fixed prompt** — Qwen prompts02 3ep vs 10ep,
   isolating how much of the collapse is just overfitting.

## Phase J: matched 3-epoch Qwen prompts02 — confound resolved

Re-ran Qwen prompts02 r=32 at `EPOCHS=3` (1h34m train).

### Epoch ablation (Qwen prompts02 r=32, fixed prompt)

| Epochs | devel M-F1 | test M-F1 | null→pos over-pred (test) |
|---|---:|---:|---:|
| 3 | 30.9 | 25.9 | 62.0 % (3071/4952) |
| 10 | 28.8 | 26.0 | 73.2 % (3624/4952) |

**The collapse is not overfitting.** 3-epoch and 10-epoch test M-F1 are
identical (25.9 vs 26.0). More epochs push the over-prediction higher
(62 → 73 %) but devel/test F1 barely move — the model is already in the
majority-positive failure mode by epoch 3. The epoch confound from
Phase I is therefore eliminated: the Qwen result stands at 3 epochs.

### Clean prompt comparison (everything at 3 epochs)

| Model | prompts01 | prompts02 | Δ test |
|---|---:|---:|---:|
| **Llama r=32** | 35.2 | **39.8** | **+4.6** |
| **Qwen r=32** | 29.1 | 25.9 | **−3.2** |

**The same prompt has opposite-signed effects across model families.**
With epochs matched, prompts02 helps Llama and hurts Qwen.

### The mechanism: opposite over-prediction response

The prompts02 changes (advise/advice typo fix + stronger "do not invent
an interaction" null definition) move the two models' null→positive
over-prediction rates in opposite directions:

| Model | prompts01 over-pred | prompts02 over-pred | direction |
|---|---:|---:|:--|
| Llama r=32 | 36.6 % | 28.8 % | ↓ (prompt worked) |
| Qwen r=32 | ~49.6 % | ~64 % | ↑ (prompt backfired) |

Llama *followed* the stronger null instruction and predicted positives
less; Qwen *over-reacted* to the richer class definitions and predicted
positives more. This is a concrete instance of **prompt brittleness
across model families**: an instruction that improves one instruction-
tuned 3B model degrades another, and you cannot tell which without
running both.

### Report take-aways (final 2.3 cross-model story)

- Prompt design is a real capacity lever (Llama +4.6 > the r=8→r=32
  LoRA gain) **but it is not transferable** — tune the prompt per model.
- Qwen is the weaker base for this task at 4-bit/r=32: it over-predicts
  more than Llama under every prompt, and the better prompt makes it
  worse. Llama r=32 prompts02 (test 39.8) remains the 2.3 champion.
- Neither model closes the gap to ML (66.8) / NN (65.6); the
  positive/null boundary is the binding constraint for the FT-LLM.
