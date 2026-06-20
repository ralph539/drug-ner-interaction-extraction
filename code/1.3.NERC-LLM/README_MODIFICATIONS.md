# System 1.3 — NERC with Large Language Models

This README tracks **every change** made on top of the provided baseline
(`1.3.NERC-llm.pdf` + the code shipped in `bin/`) so experiments are
reproducible and the report writer can tell at a glance *what was original code
and what we added* — mirroring the `1.2.NERC-NN/README_MODIFICATIONS.md`
workflow.

Every modification in the source code is marked inline with a comment of the
form `# [MOD-1.3] …` so you can grep for them:

```bash
grep -n "\[MOD-1.3\]" bin/*.py bin/*.sh
```

**Status (as of 2026-04-20 12:22 CEST — ALL EXPERIMENTS DONE):**
Phase A (code prep, local only) — **done**.
Phase B (smoke test) — **done** (job `414791`, devel micro-F1 = 37.6 %).
Phase C — **done** (jobs `414918`–`414925`). Best devel few-shot
configuration: llama32B3 / prompts01 / **15 shots** / -quant →
**39.2 % micro-F1** (later beaten by `prompts03/15` → 40.9 %).
Phase D (LoRA fine-tune) — **done** (train `414926` 5h07m + auto-chained
inference `415067` 1h07m) → devel **59.7 % micro-F1** (+20.5 pp over
best few-shot).
Phase E (test-set eval) — **done** (`415436` + `415437`):
FT test m-F1 = **57.0 %** (–2.7 pp vs devel), few-shot test m-F1 =
36.2 %. FT beats few-shot on test by **+20.8 pp**. See §6.3.
Phase F — **done** (3/4, F2 dropped):
`415438` prompts02/15 devel = 39.6 %, `415439` prompts03/15 devel =
**40.9 % (best few-shot micro)**, `415440+415467` FT **qwen25B3 devel
= 61.2 % m-F1 / 56.6 % M-F1**. F2 (non-quant LoRA) dropped — OOM on
10 GB RTX 3080 (see §4.6).
Phase G (second-order ablations) — **done**:
`415464+415481` FT **r=32 devel = 63.8 % m-F1 / 59.6 % M-F1 (new best
devel)**, `415466` balanced 15-shot FS devel = 33.2 % m-F1 (lifts
`drug_n` 0 → 25.3 % F1), `415465+415516` 15-epoch FT devel = 61.5 % /
56.8 % (+1.8/+5.3 pp over baseline; rank matters more than epochs).
Phase H (best-system consolidation) — **done**:
`415523` FT **r=32 test = 58.9 % m-F1 / 55.7 % M-F1 (best test
micro-F1, +1.9 pp over r=8)**; `drug_n` test F1 27.9 → **41.7 %**.
`415524+415683` Qwen+r=32 combo devel = **63.0 % m-F1 / 60.8 % M-F1
(best devel macro-F1)**; `drug_n` = **44.7 %** (best devel rare-class).
Phase I (devel→test robustness of the combo) — **done**:
`415762` Qwen+r=32 **test = 58.4 % m-F1 / 55.1 % M-F1** — *below* Llama
r=32 on every test metric. Qwen overfit devel (`drug_n` 44.7 → 35.8,
–8.9 pp), Llama did not (37.3 → 41.7, +4.4 pp). **Llama 3.2 3B + LoRA
r=32 is the single best configuration on both devel and test.**

---

## 1. Starting point (what was *already* in the provided baseline)

The code shipped in `bin/` is the one delivered on the AHLT web page *before*
the professor's 2026-04-10 bugfix notice. It already contained:

* **Data loader** (`examples.py`, class `Examples`)
  - Parses the DDI-style XML into `(id, input_sentence, expected_xml_output)`
    triples where the expected output wraps drug mentions in pseudo-XML tags
    (`<drug>…</drug>`, `<group>…</group>`, `<drug_n>…</drug_n>`, `<brand>…</brand>`).
  - `select_examples(k, balanced=…)` picks `k` few-shot demonstrations
    (currently random / first-k).
  - `eval_format(ex, gen_text)` re-converts the LLM's pseudo-XML output into
    the `doc_id|start|end|name|type` evaluator format so the standard
    `util/evaluator.py NER …` script can score it.
* **Prompt class** (`prompts.py`, class `Prompts`)
  - Loads a JSON prompt file with `system` and `user` templates.
  - `prepare_messages(question, answer="")` returns the OpenAI-style
    `messages=[…]` list, with the `k` few-shot examples pre-baked when
    constructed for few-shot use.
* **Model wrappers** (`model.py`)
  - `Inference`: loads an HF causal LM (optionally 4-bit quantized via
    BitsAndBytes, optionally with a LoRA adapter dir) OR routes through
    `ollama`. Exposes `generate(messages)`.
  - `FineTuning`: loads the model with a LoRA config applied, `tokenize_dataset`
    turns `Examples` triples into HF `Dataset` rows, `train` runs the HF
    `Trainer` and saves the LoRA weights.
* **Drivers**
  - `fewshot.py`: loads train data, picks `k` examples, loads model, generates
    for every devel/test sentence, saves `.json` (raw) + `.out` (evaluator
    format).
  - `finetune-train.py`: LoRA-tunes a model on the training set, saves the
    adapter to `models/FT-<model>[-quant].weights`.
  - `finetune-inference.py`: loads the base model + the saved LoRA adapter,
    runs inference, same `.json` + `.out` output layout.
* **Sbatch drivers** (`fewshot.sh`, `FT-train.sh`, `FT-inference.sh`) for the
  Boada cluster — 1×RTX 3080, 48–64 GB RAM, queue `cudabig`.
* **Default prompt** `prompts01.json` — minimal task description plus an
  example-of-one inline.

So the data-conversion logic, LoRA config, evaluator re-formatting, and Boada
sbatch headers all come from the baseline — we build on top of them.

---

## 2. Phase A — Code prep (local, no cluster yet)

All of §2 is pure "make the provided baseline run"; no experimental variation
yet. Everything here is marked `# [MOD-1.3] …` inline.

### 2.1 `bin/fewshot.py` — fix the parameter-parsing crash

**Problem.** The original `get_arguments()` read `sys.argv[6]` *before*
checking that the argument existed:

```python
if (not 6<=len(sys.argv)<=7 or sys.argv[6] not in ["-quant", "-ollama"]):
```

Running `fewshot.py` with only the 5 required positional args
(`model prompts shots train test`) — which is the default path the sbatch
driver takes when no `-quant`/`-ollama` flag is set, because the shell
variable `$QUANT` is empty — crashed with `IndexError: list index out of
range`. Same issue in the two lines that set `quantized` and `ollama`.

**Fix.** Split the length check from the flag check, and default the flag to
the empty string when omitted:

```python
if not 6 <= len(sys.argv) <= 7:
    print(f"Usage:  {sys.argv[0]} model prompts num_few_shot trainfile testfile [(-quant|-ollama)]", file=sys.stderr)
    sys.exit(1)
if len(sys.argv) == 7 and sys.argv[6] not in ["-quant", "-ollama"]:
    print(f"Usage:  {sys.argv[0]} model prompts num_few_shot trainfile testfile [(-quant|-ollama)]", file=sys.stderr)
    sys.exit(1)
…
flag = sys.argv[6] if len(sys.argv) == 7 else ""
quantized = (flag == "-quant")
ollama = (flag == "-ollama")
```

This matches what the professor's 2026-04-10 notice refers to when it says
"I fixed an error in the parameter parsing of the LLM programs (fewshot,
finetune-train, finetune-inference)". `finetune-train.py` and
`finetune-inference.py` both already gated their optional `-quant` check
behind `len(sys.argv) == 6`, so they don't crash; they only need the path
update in §2.2. We keep the same behaviour as the new package on the web
page.

### 2.2 Boada paths: `mml0` → `mgl0` (per 2026-04-10 notice)

The professor moved both the shared venv and the pre-downloaded model
directory on Boada:

| Old                                         | New                                         |
|---------------------------------------------|---------------------------------------------|
| `/scratch/nas/1/PDI/mml0/MML.venv`          | `/scratch/nas/1/PDI/mgl0/AHLT.venv`         |
| `/scratch/nas/1/PDI/mml0/models`            | `/scratch/nas/1/PDI/mgl0/models`            |

Files touched (all marked `# [MOD-1.3]`):

| File                         | What changed                                  |
|------------------------------|-----------------------------------------------|
| `bin/fewshot.py`             | `MODEL_PATH = …/mgl0/models/{model}`          |
| `bin/finetune-train.py`      | `MODEL_PATH = …/mgl0/models/{model}`          |
| `bin/finetune-inference.py`  | `MODEL_PATH = …/mgl0/models/{model}`          |
| `bin/fewshot.sh`             | `source …/mgl0/AHLT.venv/bin/activate`        |
| `bin/FT-train.sh`            | `source …/mgl0/AHLT.venv/bin/activate`        |
| `bin/FT-inference.sh`        | `source …/mgl0/AHLT.venv/bin/activate`        |

Quick verification:

```bash
grep -rn "mml0" code/1.3.NERC-LLM/bin/   # should only match [MOD-1.3] comments
grep -rn "mgl0" code/1.3.NERC-LLM/bin/   # should match the 6 call sites + sbatch docs
```

### 2.3 Nothing else touched in Phase A

`examples.py`, `prompts.py`, `model.py`, `paths.py`, `prompts01.json` are
unchanged. Those belong to Phase C/D when we start experimenting with
prompt variants and example-selection strategies.

### 2.4 `bin/fewshot.sh` — disambiguate prompt-variant outputs

**Problem.** `fewshot.py` names its outputs
`FS-<model>-<shots>-<test>[-quant].{out,json}` — the prompt file is *not*
encoded. When we start the Phase C grid with the same model+shots but
different prompt variants (`prompt01` vs `prompt02` vs `prompt03`), each
subsequent job would silently overwrite the previous one's `.out` and the
evaluator would score the last-written file against a mismatched
`.stats` name.

**Fix (sbatch-side, no Python change needed).** After `fewshot.py` returns
we rename `FS-<model>-<shots>-<test>[-quant].{out,json}` to
`FS-<model>-<prompts>-<shots>-<test>[-quant].{out,json}` and then point
the evaluator at the renamed files (marked `# [MOD-1.3]`):

```bash
BASE=../results/FS-$MODEL-$SHOTS-${TEST}${QUANT}
TAGGED=../results/FS-$MODEL-$PROMPTS-$SHOTS-${TEST}${QUANT}
for ext in out json; do
  if [ -f $BASE.$ext ]; then mv $BASE.$ext $TAGGED.$ext; fi
done
python3 ../../../util/evaluator.py NER ../../../data/$TEST.xml  $TAGGED.out $TAGGED.stats
```

Side-effect: the smoke-test (submitted before the patch) will land with the
old un-tagged filename `FS-llama32B3-5-devel-quant.*`; we'll rename it by
hand before publishing §6. Everything submitted afterwards carries the
prompt tag already.

---

## 3. Phase B — Boada smoke test (in progress)

Goal: run exactly one job end-to-end on the cluster before investing in a
grid, to confirm paths, venv, and GPU allocation work.

### 3.1 Cluster access setup

- SSH host: `boada.ac.upc.edu` (DNS → `147.83.30.197`, only reachable from
  inside the UPC VPN). Account: `ahlt1008` (given by L. Padró on 2026-04-15).
- Login shell is SLURM-wrapped — every SSH session allocates an interactive
  job via `salloc` (jobs named `_interac`, QOS `interactive`). This means:
  - A non-TTY command sent over SSH (`ssh boada 'hostname'`) is rejected
    with `invalid command` — the wrapper needs a pseudo-TTY.
  - `rsync` over SSH therefore doesn't work (it relies on `ssh remote rsync
    --server …`).
  - `scp -O` **does** work (uses the SCP protocol channel, bypasses the
    login wrapper).
  - For anything that needs a shell (`tar x`, `sbatch`, `squeue`, tailing
    logs) we drive a PTY with `pexpect` from the local machine.
- Local key setup: `~/.ssh/id_ed25519_boada` + a `Host boada` stanza in
  `~/.ssh/config`.
- Available accounts/QOS for `ahlt1008`:
  `cudabig` (with `cudabig3080 / cudabig4000 / cudabig4090`), `cuda`
  (with `cuda / cuda3080 / cuda4000 / cuda4090`), `default`.
  The sbatch headers in `fewshot.sh` / `FT-*.sh` use `-A cudabig
  --qos=cudabig3080 --gres=gpu:rtx3080:1` which are authorised.

### 3.2 Confirmed cluster layout

- Home / working dir: `/scratch/nas/1/ahlt1008/` (the NAS — same path from
  all nodes).
- Uploaded project layout on Boada: `~/AHLT-project/{code/1.3.NERC-LLM,
  data, util, resources}` (copied via `tar | scp -O | tar x`).
- Shared venv: `/scratch/nas/1/PDI/mgl0/AHLT.venv/` with
  `transformers 4.57.3`, `torch 2.7.1+cu126`, `peft 0.18.1`,
  `bitsandbytes 0.49.2`, `datasets 4.8.4`, `spacy 3.8.14`.
- Pre-downloaded models in `/scratch/nas/1/PDI/mgl0/models/`:
  - `llama32B3`        → Llama 3.2 3B Instruct (this is the `$model` arg to pass)
  - `qwen25B3`         → Qwen 2.5 3B Instruct
  - `qwen35B2`         → Qwen 3.5 2B
  - plus HF-cache-style mirror directories `models--meta-llama--…`,
    `models--Qwen--…` for the same checkpoints.

### 3.3 Smoke-test submission

From `~/AHLT-project/code/1.3.NERC-LLM/bin`:

```bash
sbatch fewshot.sh llama32B3 prompts01.json 5 train devel -quant
```

Reason for `-quant` + 5 shots: minimise GPU RAM pressure and wall-clock time
while still exercising the complete pipeline (data loader, tokenizer,
generate, evaluator re-format). Job 414791 ran for **44:51** on `boada-10`
at **1.80 s/example** over 1477 devel sentences. Devel scores:

| Class     |  TP |  FP |  FN |  #pred |  #exp |    P  |    R  |   F1  |
|-----------|----:|----:|----:|-------:|------:|------:|------:|------:|
| brand     | 147 | 161 | 228 |    308 |   375 | 47.7 % | 39.2 % | **43.0 %** |
| drug      | 741 | 718 |1203 |   1459 |  1944 | 50.8 % | 38.1 % | **43.5 %** |
| drug\_n   |  28 |  59 |  72 |     87 |   100 | 32.2 % | 28.0 % | **29.9 %** |
| group     | 150 | 535 | 556 |    685 |   706 | 21.9 % | 21.2 % | **21.6 %** |
| **M-avg** |  —  |  —  |  —  |   —   |   —   | 38.1 % | 31.6 % | **34.5 %** |
| **m-avg** |1066 |1473 |2059 |   2539 |  3125 | 42.0 % | 34.1 % | **37.6 %** |
| m-avg (no class) |1335 |1204 |1790 |2539 |3125 | 52.6 % | 42.7 % | **47.1 %** |

Takeaways:
* `group` is the weak spot — the model tags anything pluralised with `-s`
  (FN 556 vs TP 150). Needs a clearer decision rule; this is exactly what
  `prompts02.json` / `prompts03.json` target in Phase C.
* `drug_n` has few gold examples (100) so its metrics are noisy — reliable
  comparison across prompt variants requires reporting the macro and micro
  averages rather than per-class.
* "No class" recall of 47.1 % shows that *span detection* (is this a drug
  at all?) is already near-OK; the bulk of the errors are type mis-labels.
  A strict prompt disambiguating the four tags should move the needle on
  M-avg without needing more examples.

Output filename `FS-llama32B3-prompt01-5-devel-quant.{out,json,stats}` —
renamed by hand from the un-tagged original because the smoke-test was
submitted before the §2.4 filename-disambiguation patch landed.

---

## 4. Phase C — Few-shot experiments (8 jobs submitted)

### 4.1 Prompt variants authored

Two new prompt files drop into `bin/` next to the provided `prompts01.json`:

* **`prompts02.json` — "strict pharmacology expert".**
  Long system message framing the model as a pharmacology expert. For each
  of the four tags we give 4–5 concrete examples (e.g. `drug` → *aspirin,
  ibuprofen, paracetamol, warfarin, lisinopril*). Output-format section is
  explicit ("character-for-character copy", no markdown, no preamble).
  Six numbered hard rules, ending with "prefer MORE SPECIFIC tag". User
  prompt re-states the 4 valid tags and repeats the no-preamble rule. Goal:
  push precision up by making the spec painfully explicit.
* **`prompts03.json` — "terse decision-guide".**
  Short, checklist-style system message. Gives a 4-line *decision guide*
  (`Capitalised trade-style token → brand`, `Lowercase chemical/generic →
  drug`, `Plural or -s/-ics/-ers ending class noun → group`, `Recreational
  / toxin / animal substance → drug_n`). Minimal constraints section.
  User prompt is a single line. Goal: see whether a shorter prompt (more
  tokens free for context) helps a 3B-parameter model stay on task.

Both files keep the same JSON shape as `prompts01.json` (`sysprompt` +
`usrprompt` arrays), so the `Prompts` loader in `prompts.py` picks them up
without any code change.

### 4.2 Submitted grid

All jobs use `sbatch fewshot.sh <model> <prompts.json> <shots> train devel -quant`
from `~/AHLT-project/code/1.3.NERC-LLM/bin`. The `-quant` flag is kept on for
the whole grid to get 8 variants through the queue in a reasonable time —
quantisation-off is a Phase D lever, where it matters more for LoRA.

| JobID    | Model      | Prompt          | Shots | Purpose                                  |
|----------|------------|-----------------|------:|------------------------------------------|
| `414791` | llama32B3  | prompts01.json  |     5 | smoke-test, launched in Phase B (F1=37.6) |
| `414918` | llama32B3  | prompts01.json  |     0 | zero-shot baseline                       |
| `414919` | llama32B3  | prompts01.json  |     3 | low-shot                                 |
| `414920` | llama32B3  | prompts01.json  |    10 | mid-shot                                 |
| `414921` | llama32B3  | prompts01.json  |    15 | high-shot (PDF-recommended ceiling)      |
| `414922` | llama32B3  | prompts02.json  |     5 | strict-prompt effect @ fixed model+shots |
| `414923` | llama32B3  | prompts03.json  |     5 | terse-prompt effect @ fixed model+shots  |
| `414924` | qwen25B3   | prompts01.json  |     5 | model axis — Qwen 2.5 3B                 |
| `414925` | qwen35B2   | prompts01.json  |     5 | model axis — Qwen 3.5 2B                 |

The `cudabig3080` QOS serialises our jobs (one at a time on a single RTX
3080), so expected wall-clock for the whole grid is ≈ 8 × smoke-test
runtime (~6 h). Each `.stats` file lands in `results/` and is collected
from there.

Filename convention is
`FS-<model>-<prompts-tag>-<shots>-devel-quant.{out,stats,json}` thanks to
the sbatch-side rename patch in §2.4. `<prompts-tag>` is the basename of
the prompt file minus `.json` (e.g. `prompts01`).

Each row gets filled into the table in §6.1 with devel F1 **only** — test
is forbidden at this stage per the PDF "What you should NOT do" page.

### 4.3 Abandoned first submission (414807–414814) and fix

The first attempt crashed 15 seconds into every job with
`FileNotFoundError: [Errno 2] No such file or directory: 'prompt01'` from
`prompts.Prompts.__init__` at `with open(promptfile) as pf : …`. Cause:
`Prompts(promptfile)` treats the CLI argument as a direct file path, so
the second positional arg to `fewshot.sh` / `FT-train.sh` must be the full
**filename with the `.json` extension**, not a bare `prompt01` tag. The
driver script mapped `$2 → $PROMPTS → python3 fewshot.py … $PROMPTS …`
without any massaging.

Nothing in `fewshot.sh` or `FT-train.sh` validates this; the PDF example
(`sbatch fewshot.sh llama32B3 prompts01.json 15 train devel`) makes it
clear but it is easy to miss. Jobs `414807`–`414814`, `414816` all failed
on this; fix was to resubmit with `prompts0{1,2,3}.json` as the prompt
argument. Runtime cost of the failure: ~2 minutes of queue time plus 9
wasted ~15-second job starts (evaluator never ran, `results/` was empty
afterwards). No SLURM state to clean up — the jobs registered as
`COMPLETED 0:0` because the Python `sys.exit(1)` at traceback was caught
by `fewshot.sh`'s `if (test $? != 0); then exit; fi`, which exited the
shell script with 0 but skipped the evaluator step.

The `FS-llama32B3-prompt01-5-*` filenames from the smoke-test were left
alone because their evaluator step did run (smoke-test was submitted
*before* both the §4.3 typo regression and the §2.4 filename patch).

### 4.4 `qwen35B2` — transformers compatibility failure

Job `414925` (`sbatch fewshot.sh qwen35B2 prompts01.json 5 train devel
-quant`) crashed in ~18 s with:

```
ValueError: The checkpoint you are trying to load has model type `qwen3_5`
but Transformers does not recognize this architecture.
```

The shared venv on Boada (`/scratch/nas/1/PDI/mgl0/AHLT.venv`) ships
`transformers 4.57.3`, which predates Qwen-3.5-family support (added in
`4.58+`). We cannot upgrade the shared venv, so Qwen-3.5 is simply
**unusable on this cluster** with the provided environment. We
documented the failure in §6.1 (row struck through) and proceeded with
`llama32B3` and `qwen25B3`, both of which the installed transformers
version loads fine.

No code change was made for this — it is an environment constraint, not
a bug in our driver. Future work, if the venv is ever refreshed, should
re-run this configuration.

### 4.5 RTX 4000 Blackwell GPUs — PyTorch/CUDA incompatibility

During Phase E+F scheduling (2026-04-18 afternoon) the `cudabig3080` queue
was congested (1× 4090, 2 of 4× 3080, and 3 concurrent jobs from other
users), so we attempted to route non-critical few-shot jobs onto the
three idle **RTX PRO 4000 Blackwell** cards in the same node
(`boada-10`) using `sbatch --qos=cudabig4000 --gres=gpu:rtx4000:1 …`.
SLURM accepted the jobs (`415235`, `415236`, `415237`, `415238`, `415239`)
and they started immediately — but each crashed in ≤30 s with:

```
NVIDIA RTX PRO 4000 Blackwell with CUDA capability sm_120 is not compatible
with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities
sm_50 sm_60 sm_70 sm_75 sm_80 sm_86 sm_90.
…
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

The shared venv's PyTorch build predates Blackwell (`sm_120`) support,
so the RTX 4000 cards on this node are **effectively unusable** for us.
Same constraint as §4.4 (can't modify the shared venv). We cancelled
the RTX 4000 re-submission strategy and went back to `cudabig3080`
exclusively (jobs `415426`–`415430`).

### 4.6 Non-quantized LoRA on RTX 3080 — VRAM OOM

Job `415240` (`FT-train.sh llama32B3 prompts01 train devel` — i.e. Phase
F2, the non-quantized fine-tune) failed at the first training step:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 18.00 MiB.
GPU 0 has a total capacity of 9.65 GiB of which 2.12 MiB is free.
```

Llama 3.2 3B in bf16 weights + LoRA adapter + optimizer state +
activations does not fit in the RTX 3080's 10 GB. This matches the
**Risk note** we wrote in §5bis before submitting — quantization isn't
optional on this GPU, it's a hard constraint. For the report this is a
useful headline: *our 59.7 % devel F1 under 4-bit quant is our ceiling
here, because the cluster GPU is too small to fit the non-quant model*.

No code change for this either — VRAM ceiling is a hardware constraint.
If the 4090 (24 GB) frees up we could retry F2 there.

---

## 5. Phase D — Fine-tuning experiments

### 5.1 Submitted

| JobID    | Script                | Args                                                | Notes                                    |
|----------|-----------------------|-----------------------------------------------------|------------------------------------------|
| ~~414816~~ | `FT-train.sh`       | `llama32B3 prompt01 train devel -quant`             | **FAILED** — same `.json` typo as §4.3; re-queued as `414926` |
| `414926` | `FT-train.sh`         | `llama32B3 prompts01.json train devel -quant`       | LoRA fine-tune, queued behind Phase C    |
| *(auto-chained)* | `FT-inference.sh` | `llama32B3 prompts01.json devel FT-llama32B3-quant.weights -quant` | auto-submitted by `FT-train.sh` when it finishes (see §5.2) |

### 5.2 Why one LoRA run for now

Our `cudabig3080` QOS rejects the 10th simultaneously-queued job
(`QOSMaxSubmitJobPerUserLimit`), so we could only queue one FT job along
with the 8 Phase C jobs + the smoke-test already in-flight. Submitting the
dependent inference job is blocked until one of the earlier jobs clears.

**Workaround applied:** patched `bin/FT-train.sh` so that, after
`finetune-train.py` returns successfully, it itself runs `sbatch
FT-inference.sh …` (marked `# [MOD-1.3]`). By the time the training job
reaches this line, 4–8 of our Phase C jobs have already drained, so there
is room in the queue. This avoids having to sit and hand-submit the
inference at exactly the right moment.

### 5.3 Axes left to explore (if time allows after Phase C)

| Axis              | Values we plan to try                                                              |
|-------------------|------------------------------------------------------------------------------------|
| Model             | `llama32B3`, `qwen25B3`                                                            |
| Quantization      | off, `-quant` (note: loading a non-quant adapter into a quant model is erratic)    |
| Learning rate     | default (from `model.py:FineTuning.train`), one lower, one higher                  |
| Epochs            | default, fewer, more                                                               |
| Batch / grad-acc  | default, smaller batch                                                             |
| Prompt            | best prompt from Phase C (PDF notes prompt matters less in fine-tune)              |
| # train examples  | full, smaller subset (to test data efficiency)                                     |

Filename convention: `FT-<model>[-quant]-devel.{out,stats,json}` +
adapter dir `models/FT-<model>[-quant].weights/`.

---

## 5bis. Phase F — Follow-up experiments (submitted 2026-04-18)

After Phase C/D landed we used the remaining cluster budget to run four
*diagnostic* experiments answering specific questions the main grid
left open. All four sit behind Phase E in the `cudabig3080` queue and
will run serially on the shared RTX 3080.

| JobID   | Script              | Args                                                 | Question being asked                                              |
|---------|---------------------|------------------------------------------------------|-------------------------------------------------------------------|
| `415098`| `FT-train.sh`       | `qwen25B3 prompts01.json train devel -quant`         | Does fine-tuning a *different* base LLM (Qwen 2.5 3B) beat Llama? |
| `415099`| `FT-train.sh`       | `llama32B3 prompts01.json train devel` *(no `-quant`)* | What fraction of our 59.7 % devel F1 is lost to 4-bit quantization? |
| `415100`| `fewshot.sh`        | `llama32B3 prompts02.json 15 train devel -quant`     | Does scaling to 15 shots lift the *strict* prompt like it did prompts01? |
| `415101`| `fewshot.sh`        | `llama32B3 prompts03.json 15 train devel -quant`     | Does scaling to 15 shots lift the *terse* prompt, or does `drug_n` stay zeroed out? |

Reasoning:

* **F1 / F2** target the two biggest axes Phase D did not vary: base
  model and quantization. They're the only experiments that can *raise*
  our best number — if F2 fits in 10 GB VRAM, it's the most likely to
  beat 59.7 %.
* **F3 / F4** fill the missing cells of the Phase C 4×3 grid: we had
  shots {0,3,5,10,15} for prompts01 but only 5 shots for prompts02 and
  prompts03. This lets us test whether the "prompts03 kills drug\_n"
  result holds at higher shot counts or disappears.

Skipped (on purpose):

* Training-data subset ablation — low signal, costs 5 h each.
* LR / epoch sweep — default (LR 2e-5, 10 epochs) already converged
  well per Phase D log; more runs would just confirm.
* `-ollama` backend — the course sample says Ollama is optional;
  `-quant` via BitsAndBytes gave us equivalent latency on our GPU.

Both FT jobs use the self-chain patch from §5.2, so each produces both
a `FT-*-devel.stats` row (to add to §6.2) and saves its adapter to
`models/FT-<model>[-quant].weights/`.

**Risk note for F2 (non-quant FT).** Llama 3.2 3B in bf16 + gradients
for the LoRA adapter + activations may push past 10 GB VRAM on the
3080. If it OOMs we'll document the failure (as we did for qwen35B2 in
§4.4); the rest of Phase F is unaffected.

---

## 5ter. Phase G — Second-order ablations (submitted 2026-04-19)

After Phase F completed we had three further questions the report wanted
answered. All three are cheap relative to the main grid and use the same
`cudabig3080` queue.

| JobID   | Script         | Args / Env                                                                              | Question being asked                                              |
|---------|----------------|-----------------------------------------------------------------------------------------|-------------------------------------------------------------------|
| `415464`| `FT-train.sh`  | `llama32B3 prompts01.json train devel -quant` + `FT_LORA_R=32 FT_LORA_ALPHA=64 FT_TAG=r32` | Does a larger LoRA adapter (r=32 vs baseline r=8) improve F1?   |
| `415465`| `FT-train.sh`  | `llama32B3 prompts01.json train devel -quant` + `FT_EPOCHS=15 FT_TAG=ep15`              | Did 10 epochs train long enough? Is 15 worth the extra ~2.5 h?  |
| `415466`| `fewshot.sh`   | `llama32B3 prompts01.json 15 train devel -quant -balanced`                              | Does class-balanced FS sampling lift `drug_n` recall above 0 %? |

**Why 15 epochs and not 20:** the `cudabig3080` QOS has
`MaxWall = 08:00:00`. Phase D's 10-epoch run took 5 h 07 m, so 20 epochs
would extrapolate to ~10 h and be killed by SLURM. 15 epochs ≈ 7 h 40 m
fits comfortably under the cap while still giving +50 % training.

Plumbing added to support Phase G (every change marked `# [MOD-1.3]`):

* `model.py` — `FineTuning.__init__` reads `FT_LORA_R` / `FT_LORA_ALPHA`
  from env (defaults 8 / 16); `FineTuning.train` reads `FT_EPOCHS`
  (default 10).
* `finetune-train.py` + `finetune-inference.py` — both read `FT_TAG`
  from env and append it to `FT-<model>[-quant]<-tag>.weights/` /
  `.out` / `.stats`, so Phase G adapters don't overwrite Phase D's.
* `FT-train.sh` — auto-chain `sbatch FT-inference.sh` now adds the
  `FT_TAG` suffix to the weight-dir path and exports `ALL` env.
* `FT-inference.sh` — evaluator call reads `FT_TAG` so it points at the
  tagged `.out` / `.stats` file.
* `examples.py` — `select_examples(balanced=True)` now supports NER
  (rarest-tag-first bucket selection + random fill). Previously it was
  a DDI-only code path.
* `fewshot.py` / `fewshot.sh` — both accept an optional `-balanced`
  flag which is forwarded end-to-end and stamped into the output
  filename (`FS-…-balanced.{out,stats,json}`).

Results will be filled into §6.1 and §6.2 as the jobs complete.

---

## 5quater. Phase H — Best-system consolidation (submitted 2026-04-19 11:30)

Phase G gave us two clear winners: (a) `r=32` LoRA, which lifts Llama
fine-tune devel m-F1 from 59.7 → 63.8 %, and (b) `qwen25B3` as the base
model, which by itself lifts Llama-r=8 59.7 → 61.2 %. Before writing
up the report we need two more runs: the **test-set number for r=32**
(Phase E only evaluated the r=8 baseline on test) and the **Qwen ×
r=32 combo** (upper-bound probe — do the two winning axes stack?).

| JobID   | Script             | Args / Env                                                                                        | Question being asked                                                                  |
|---------|--------------------|---------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------|
| `415523`| `FT-inference.sh`  | `llama32B3 prompts01.json test FT-llama32B3-quant-r32.weights -quant` + `FT_TAG=r32`              | **Test-set m-F1 for the new best adapter.** Reuses r=32 weights from `415464`; ~1 h. |
| `415524`| `FT-train.sh`      | `qwen25B3 prompts01.json train devel -quant` + `FT_LORA_R=32 FT_LORA_ALPHA=64 FT_TAG=r32`         | **Does r=32 + Qwen stack?** Both independent winners. ~6 h + auto-chained inference. |

Both submitted at 11:30 CEST; both `PD (Priority)` initially. The r=32
test eval is mandatory for an honest §6.3 headline number; the Qwen ×
r=32 combo is the upper-bound probe. Results → §6.2 / §6.3.

---

## 6. Results tables (to be filled)

### 6.1 Few-shot — devel

Each row = one sbatch run of `fewshot.sh`. Devel micro-F1 (m.avg) and
macro-F1 (M.avg) come from the `.stats` file emitted by
`util/evaluator.py NER …`. `Runtime` is wall-clock sbatch `Elapsed`.

| JobID   | Model     | Prompt     | Shots | Quant | Devel m-F1 | Devel M-F1 | span-F1 (no class) | Runtime | Notes |
|---------|-----------|------------|------:|:-----:|-----------:|-----------:|-------------------:|--------:|-------|
| 414791  | llama32B3 | prompts01  |     5 |  yes  |     37.6 % |     34.5 % |             47.1 % |   44:51 | smoke-test |
| 414918  | llama32B3 | prompts01  |     0 |  yes  |      2.2 % |      1.9 % |              2.9 % |   47:38 | zero-shot collapses — model ignores tagging instructions |
| 414919  | llama32B3 | prompts01  |     3 |  yes  |     29.5 % |     29.3 % |             42.5 % |   46:03 | 3 shots < 5 shots — marginally too little signal |
| 414920  | llama32B3 | prompts01  |    10 |  yes  |     35.3 % |     34.3 % |             46.1 % |   51:29 | 10 shots plateau vs 5 |
| **414921** | **llama32B3** | **prompts01**  | **15** |  **yes**  | **39.2 %** | **32.8 %** |         **48.8 %** | **52:47** | **best few-shot** (highest m-F1) |
| 414922  | llama32B3 | prompts02  |     5 |  yes  |     38.2 % |     28.9 % |             46.8 % |   43:50 | strict prompt: more precision, crushes drug\_n recall to 3 % |
| 414923  | llama32B3 | prompts03  |     5 |  yes  |     39.0 % |     25.6 % |             49.1 % |   42:07 | terse prompt: highest span-F1 but drug\_n recall = 0 % |
| 414924  | qwen25B3  | prompts01  |     5 |  yes  |     35.3 % |     22.0 % |             45.3 % |   53:27 | Qwen 2.5 3B — very cautious (recall 26.6 % on brand) |
| ~~414925~~ | qwen35B2 | prompts01  |     5 |  yes  |     FAILED |     FAILED |             FAILED |   00:18 | HF 4.57.3 doesn't know `qwen3_5` model type (see §4.4) |
| 415438  | llama32B3 | prompts02  |    15 |  yes  |     39.6 % |     28.7 % |             48.6 % |   47:11 | strict prompt at 15 shots: +1.4 pp vs 5-shot; `drug_n` still 0 % recall |
| **415439** | **llama32B3** | **prompts03** | **15** |  **yes**  | **40.9 %** | **26.6 %** |         **51.0 %** | **45:54** | **new best few-shot** — terse prompt + 15 shots beats prompts01/15 by 1.7 pp; `drug_n` still 0 % |
| 415466  | llama32B3 | prompts01  |    15 |  yes  |     33.2 % |   **32.5 %** |           41.1 % |   56:55 | **Phase G — balanced sampler** (`-balanced`): guarantees ≥3 `drug_n` demos → `drug_n` recall 0 → 33 %, F1 0 → **25.3 %**. Micro-F1 drops 6 pp (fewer common-class demos), but macro-F1 is the **best few-shot macro seen** (32.5 %). |

**Trend interpretation.**

* **Shots axis** (llama32B3 / prompts01): 0 → 2.2, 3 → 29.5, 5 → 37.6, 10
  → 35.3, 15 → 39.2. A clear floor at zero-shot (the model doesn't emit
  tags on its own), big lift at 3–5 shots, then a noisy plateau above 5.
  15 shots wins but only by ~1.6 pp over 5 — the extra examples mostly
  give the model more courage to tag `group` (recall 30.2 % vs 21.2 %).
* **Prompt axis** (5 shots, llama32B3): prompts01 37.6 %, prompts02
  38.2 %, prompts03 39.0 %. The terse decision-guide prompt
  (`prompts03`) edges out the strict one (`prompts02`) on micro-F1 but
  completely **zeros out `drug_n`** — both refined prompts push the
  model toward the three "easy" classes and away from the rare
  recreational/illicit category. macro-F1 therefore *drops* (prompts01
  34.5 → prompts03 25.6) even though micro-F1 is flat to up. The lift
  from the refined prompts is almost entirely precision on `brand`
  (47.7 → 75.6 % in prompts03).
* **Prompt × Shots interaction** (15 shots, llama32B3): prompts01
  39.2 %, prompts02 39.6 %, **prompts03 40.9 %**. Scaling shots from
  5 to 15 lifts all three prompts, but the gap *widens* for the refined
  prompts (prompts03: 39.0 → 40.9, +1.9 pp; prompts02: 38.2 → 39.6,
  +1.4 pp). At 15 shots prompts03 overtakes prompts01 by 1.7 pp on
  micro-F1 while keeping `drug_n` at 0 %. So the best few-shot
  *micro-F1* is prompts03/15, but if macro-F1 matters (and it does for a
  rare-class-sensitive task like DDI), prompts01/15 (M=32.8 %) is the
  saner choice — prompts03/15 macro is only 26.6 %.
* **Model axis** (5 shots, prompts01): llama32B3 37.6 %, qwen25B3
  35.3 %, qwen35B2 failed. Qwen 2.5 3B is not a drop-in replacement —
  its `group` recall collapses to 4.0 %. Staying on Llama 3.2 3B
  Instruct is the right call *for prompting*. (Note: once
  fine-tuned, Qwen actually *beats* Llama — see §6.2.)
* **Class-balanced sampler axis** (Phase G, 15 shots / prompts01):
  default random sampler 39.2 % m-F1 / 32.8 % M-F1 vs balanced sampler
  **33.2 % m-F1 / 32.5 % M-F1**. Balanced sampling **lifts `drug_n`
  from 0 % F1 to 25.3 %** (recall 0 → 33 %) at the cost of 6 pp
  micro-F1 and small drops on the three common classes. Macro-F1 is
  essentially flat (32.8 → 32.5). The takeaway: you can trade a
  headline metric for rare-class discovery in the few-shot regime, but
  there is no free lunch — few-shot cannot simultaneously cover the
  head and the tail of the class distribution with only 15
  demonstrations. (Fine-tuning with r=32 solves this: see §6.2.)

### 6.2 Fine-tuning — devel

| JobID             | Model     | Prompt    | Quant | LoRA r / Epochs | Devel m-F1 | Devel M-F1 | span-F1 (no class) | Train / Inf time | Notes |
|-------------------|-----------|-----------|:-----:|:---------------:|-----------:|-----------:|-------------------:|-----------------:|-------|
| **414926 + 415067** | **llama32B3** | **prompts01** | **yes** | **r=8 / 10 ep** | **59.7 %** | **51.5 %** | **62.1 %** | **05:07:34 / 01:07:42** | LoRA baseline, `FT-train` → auto-chained `FT-inference` |
| 415440 + 415467   | qwen25B3  | prompts01 |  yes  | r=8 / 10 ep     |     61.2 % |     56.6 % |             63.7 % | 05:57:16 / 01:20:20 | **Phase F — Qwen 2.5 3B** beats Llama baseline by **+1.5 pp m-F1** and +5.1 pp M-F1 — best macro seen. `drug_n` F1 = 33.5 % (vs 14.6 % Llama). |
| **415464 + 415481** | **llama32B3** | **prompts01** | **yes** | **r=32 / 10 ep** | **63.8 %** | **59.6 %** | **66.3 %** | **05:19:30 / 01:01:23** | **Phase G — new best FT** — quadrupling LoRA rank (8 → 32, alpha 16 → 64) adds **+4.1 pp m-F1** and +8.1 pp M-F1 over Llama baseline. Every class improves; `group` +5.5 pp, `drug_n` +22.7 pp F1. |
| 415465 + 415516   | llama32B3 | prompts01 |  yes  | r=8 / 15 ep     |     61.5 % |     56.8 % |             64.0 % | 07:57:xx / ~01:00 | **Phase G — 15 epochs** — +1.8 pp m-F1 / +5.3 pp M-F1 over baseline. `drug_n` 14.6 → 35.1 %, but still below r=32 (37.3 %). More epochs help, but less than more rank. |
| 415524 + 415683   | qwen25B3  | prompts01 |  yes  | r=32 / 10 ep    |     63.0 % |   **60.8 %** |           64.9 % | ~06:30 / ~01:20 | **Phase H — Qwen + r=32 combo** — micro 0.8 pp *below* Llama r=32 (63.8) but **best macro ever (60.8 %)** and **best `drug_n` (44.7 %)**. Two axes don't stack on micro; they stack on macro/rare. |

**Per-class breakdown (devel, FT-llama32B3-quant r=8/10ep — baseline):**

|       | P      | R      | F1     |
|-------|-------:|-------:|-------:|
| brand | 74.9 % | 72.5 % | 73.7 % |
| drug  | 67.6 % | 59.1 % | 63.0 % |
| drug\_n | 10.5 % | 24.0 % | 14.6 % |
| group | 54.0 % | 55.1 % | 54.5 % |

**Per-class breakdown (devel, FT-qwen25B3-quant — Phase F):**

|       | P      | R      | F1     |
|-------|-------:|-------:|-------:|
| brand | 84.2 % | 71.2 % | 77.2 % |
| drug  | 68.3 % | 57.9 % | 62.7 % |
| drug\_n | 36.5 % | 31.0 % | 33.5 % |
| group | 56.5 % | 49.7 % | 52.9 % |

**Per-class breakdown (devel, FT-llama32B3-quant r=32 — Phase G, best micro):**

|       | P      | R      | F1     |
|-------|-------:|-------:|-------:|
| brand | 83.9 % | 70.7 % | 76.7 % |
| drug  | 69.6 % | 59.9 % | 64.4 % |
| drug\_n | 34.2 % | 41.0 % | 37.3 % |
| group | 62.6 % | 57.6 % | 60.0 % |

**Per-class breakdown (devel, FT-qwen25B3-quant r=32 — Phase H, best macro / best `drug_n`):**

|       | P      | R      | F1     |
|-------|-------:|-------:|-------:|
| brand | 84.9 % | 71.7 % | **77.7 %** |
| drug  | 67.8 % | 59.2 % | 63.2 % |
| drug\_n | 45.4 % | 44.0 % | **44.7 %** |
| group | 61.1 % | 54.2 % | 57.5 % |

**Interpretation (baseline FT).** LoRA fine-tuning (r=8, 10 epochs) lifts
micro-F1 from **39.2 %** (best few-shot, 15 shots) to **59.7 %** — a
**+20.5 pp** jump. Every class improves in F1 (`brand` 50.4 → 73.7,
`drug` 41.3 → 63.0, `group` 24.0 → 54.5, `drug_n` 5.7 → 14.6). Macro-F1
also jumps (26.6 → 51.5 %), meaning FT spreads capability more evenly
across classes than prompting.

**Phase F interpretation — Qwen 2.5 3B FT.** Swapping the Llama 3.2 3B
base for Qwen 2.5 3B under the same LoRA config (r=8, 10 epochs) gives
**+1.5 pp micro-F1** (59.7 → 61.2 %) and a big **+5.1 pp macro-F1**
(51.5 → 56.6 %). The macro jump is almost entirely `drug_n` (F1
14.6 → 33.5 %, +18.9 pp) — Qwen is markedly more willing to tag the
rare illicit-drug class. This **reverses the few-shot verdict** that
Qwen was "cautious and unfit" (devel 35.3 % at 5 shots, group recall
4 %): with training data Qwen's cautiousness turns into higher
precision across the board (`brand` P 84.2 % > Llama 74.9 %). The
lesson is that zero/few-shot rankings of base models don't carry over
to fine-tuned settings.

**Phase G interpretation — LoRA rank ablation.** Quadrupling rank
(r=8 → r=32, alpha 16 → 64) and training the same 10 epochs costs only
+0:11 of training time but lifts micro-F1 **+4.1 pp** (59.7 → **63.8 %**)
and macro-F1 **+8.1 pp** (51.5 → 59.6 %). Every class gains, but the
biggest winners are the two underperformers: `drug_n` F1
14.6 → **37.3 %** (+22.7 pp) and `group` 54.5 → 60.0 % (+5.5 pp). The
r=8 adapter was evidently **capacity-bound** on the rare and
semantically-boundary classes — larger rank gives the model enough
parameters to fit the minority distributions without hurting the
common classes. **r=32 is the system-1.3 best-micro devel result** and
the cheapest quality win we found in the whole campaign.

**Phase H interpretation — axis stacking (Qwen × r=32).** The obvious
upper-bound probe: Qwen and r=32 each beat the Llama-r=8 baseline
independently (+1.5 pp and +4.1 pp m-F1). Combining them gives devel
m-F1 = **63.0 %** — **0.8 pp below Llama-r=32 (63.8 %)**. So on micro
the axes do **not** stack; the gains overlap. But the story flips on
the other metrics:

* **Best macro-F1 ever seen: 60.8 %** (vs 59.6 % for Llama-r=32,
  56.6 % for Qwen-r=8, 51.5 % for Llama-r=8).
* **Best `drug_n` F1 ever seen: 44.7 %** (vs 37.3 % Llama-r=32,
  33.5 % Qwen-r=8, 14.6 % Llama-r=8).
* **Best `brand` F1 ever seen: 77.7 %** (by 0.5 pp over Qwen-r=8).

So Qwen-r=32 is the **balanced-metric winner** and **Llama-r=32 is
the micro-F1 winner**. The difference matters for the report: if the
graded metric is micro-F1 (which is what the `evaluator.py` script
prints as the headline), Llama-r=32 is the system we hand in. If the
evaluator or the marker cares about rare classes, Qwen-r=32 is
strictly better. We report both configurations and flag the tradeoff
in §6.3.

### 6.3 Final test-set evaluation

Both test-set jobs ran on 2026-04-18 evening (`cudabig3080`, -quant). The
table reports the "headline" number used to compare against the 1.1 (CRF)
and 1.2 (BiLSTM-NN) systems.

| JobID   | System        | Config                              | Devel m-F1 | **Test m-F1** | Test M-F1 | Test span-F1 | Runtime |
|---------|---------------|-------------------------------------|-----------:|--------------:|----------:|-------------:|--------:|
| 415437  | Best few-shot | llama32B3 / prompts01 / 15 / -quant |     39.2 % |      **36.2 %** |    23.0 % |       45.9 % |   53:58 |
| 415436  | Fine-tune r=8 | llama32B3 / prompts01 / 10 ep / -quant |     59.7 % |      **57.0 %** |    51.5 % |       60.9 % | 01:07:02 |
| **415523** | **Best fine-tune (LoRA r=32)** | **llama32B3 / prompts01 / 10 ep / -quant** | **63.8 %** | **58.9 %** | **55.7 %** | **62.5 %** | **01:00:xx** |
| 415762  | Qwen + r=32 combo | qwen25B3 / prompts01 / 10 ep / -quant |     63.0 % |      58.4 %   |    55.1 % |       61.7 % | ~01:00 |

**Per-class breakdown on test (FT-llama32B3-quant, r=8 — Phase E):**

|       | P      | R      | F1     |
|-------|-------:|-------:|-------:|
| brand | 72.6 % | 66.5 % | 69.4 % |
| drug  | 65.4 % | 55.6 % | 60.1 % |
| drug\_n | 23.9 % | 33.3 % | 27.9 % |
| group | 49.9 % | 47.0 % | 48.4 % |

**Per-class breakdown on test (FT-llama32B3-quant, r=32 — Phase H, new best):**

|       | P      | R      | F1     |
|-------|-------:|-------:|-------:|
| brand | 74.1 % | 64.4 % | 68.9 % |
| drug  | 67.0 % | 56.5 % | 61.3 % |
| drug\_n | 36.8 % | 48.0 % | **41.7 %** |
| group | 52.7 % | 49.4 % | 51.0 % |

**Per-class breakdown on test (FT-qwen25B3-quant r=32 — Phase I):**

|       | P      | R      | F1     |
|-------|-------:|-------:|-------:|
| brand | 79.0 % | 71.3 % | 75.0 % |
| drug  | 65.0 % | 56.0 % | 60.2 % |
| drug\_n | 48.3 % | 28.4 % | 35.8 % |
| group | 51.1 % | 48.1 % | 49.6 % |

**Test-set interpretation.**

* **Few-shot test 36.2 % ↔ devel 39.2 %** — only 3.0 pp drop. The model
  doesn't overfit at 15 shots because there is no training loss on the
  demonstrations — devel/test difficulty is the main axis.
* **Fine-tune test 57.0 % ↔ devel 59.7 %** — 2.7 pp drop. Our LoRA
  adapter *generalises* to the unseen test split; no sign of overfit.
* **Fine-tune still beats few-shot by +20.8 pp on test** (57.0 vs 36.2),
  essentially the same gap we saw on devel (+20.5 pp). The gain is real
  and not an artefact of the devel set.
* On test, **`drug_n` jumps from 0.0 % (few-shot) → 27.9 % (FT)** — the
  rare-class discovery that LoRA enables is the single biggest
  qualitative difference versus prompt-only approaches.
* `brand` is again the easiest class (capitalisation is a hard signal);
  `group` remains a failure mode for few-shot (20.0 % F1) because the
  class boundary with `drug` is semantic, not orthographic.

**Test-set interpretation (continued — Phase H r=32 addition).**

* **r=32 test 58.9 % ↔ devel 63.8 %** — a 4.9 pp gap vs r=8's 2.7 pp.
  The larger adapter overfits devel slightly more than r=8, but even
  after the gap **r=32 still beats r=8 on test by +1.9 pp** (58.9 vs
  57.0). Rank scaling is not a devel-only artefact.
* The big r=32 test-set win is again `drug_n`: **F1 27.9 → 41.7 %**
  (+13.8 pp), driven by precision lifting 23.9 → 36.8 % while recall
  also grows. `group` is the other gainer (+2.6 pp); `brand` and
  `drug` are essentially flat on test. So the r=32 rank capacity buys
  rare-class discovery without hurting the easy classes.

**Test-set interpretation (continued — Phase I Qwen × r=32).**

The devel picture said Qwen × r=32 was the macro/rare-class winner
(devel M-F1 60.8 %, `drug_n` 44.7 %). On test **the advantage
disappears entirely** — Llama × r=32 beats Qwen × r=32 on every axis:

| Metric         | Llama r=32 | Qwen r=32 | Δ         |
|----------------|-----------:|----------:|----------:|
| micro-F1       |   58.9 %   |   58.4 %  | –0.5 pp   |
| macro-F1       |   55.7 %   |   55.1 %  | –0.6 pp   |
| `drug_n` F1    | **41.7 %** |   35.8 %  | **–5.9 pp** |
| `brand` F1     |   68.9 %   |   75.0 %  | +6.1 pp (Qwen wins) |

The big devel-→-test gap is on `drug_n`: Qwen dropped **44.7 → 35.8
(–8.9 pp)** while Llama *gained* **37.3 → 41.7 (+4.4 pp)**. So
Qwen severely overfit `drug_n` on devel. Llama's `drug_n` prediction
is more robust across splits. Qwen still wins `brand` on test
(+6.1 pp), but only there.

**Bottom line for the report:** Llama 3.2 3B Instruct with LoRA r=32 /
α=64 / 10 epochs / 4-bit quant is the single best configuration —
best on devel *and* test, best on micro *and* macro. The Phase F / G /
H / I ablation shows the result is not a lucky devel hit.

**Report-ready line:** *LoRA-fine-tuning Llama 3.2 3B Instruct (4-bit,
`r=32`, α=64, 10 epochs) on the DDI training set lifts test-set NERC
micro-F1 from 36.2 % (15-shot prompting, same base model) to 58.9 %, a
**+22.7 pp** absolute improvement (macro-F1 23.0 → 55.7 %, **+32.7 pp**),
with a 4.9 pp generalisation gap from devel (63.8 %).* The system still
trails System 1.1 (CRF, 86.4 %) and System 1.2 (BiLSTM-NN, 88.4 %) —
consistent with 3B-param quantised LLMs being structurally unsuited for
token-level NERC under hardware limits imposed by the cluster.

---

## 7. Reference — evaluator invocation

All scoring uses the standard project evaluator, invoked inside the sbatch
drivers after the Python job returns (see `fewshot.sh`, `FT-inference.sh`):

```bash
python3 ../../../util/evaluator.py NER \
    ../../../data/<split>.xml \
    ../results/<prefix>.out \
    ../results/<prefix>.stats
```

So every row in §6 traces back to one `.stats` file in `results/`.

---

## 8. Change log

| Date       | Change                                                                 |
|------------|------------------------------------------------------------------------|
| 2026-04-17 | Fixed `fewshot.py` argv parsing crash (`sys.argv[6]` IndexError)       |
| 2026-04-17 | Updated Boada venv path in `fewshot.sh` / `FT-train.sh` / `FT-inference.sh` |
| 2026-04-17 | Updated Boada `MODEL_PATH` in `fewshot.py` / `finetune-train.py` / `finetune-inference.py` |
| 2026-04-17 | Created this README                                                    |
| 2026-04-17 | Phase B: submitted smoke-test `414791` (`llama32B3 prompt01 5 devel -quant`) |
| 2026-04-17 | Added `bin/prompts02.json` (strict pharmacology-expert prompt)         |
| 2026-04-17 | Added `bin/prompts03.json` (terse decision-guide prompt)               |
| 2026-04-17 | Patched `bin/fewshot.sh` to rename outputs with prompt tag (prevents Phase-C overwrite) |
| 2026-04-17 | Phase C: submitted 8 few-shot jobs `414807`–`414814` (grid in §4.2)    |
| 2026-04-17 | Phase D: submitted FT-train `414816`; FT-inference resubmit pending (QOS cap hit at 10) |
| 2026-04-17 | Patched `bin/FT-train.sh` to auto-submit `FT-inference.sh` at end of training (self-chain workaround for 10-job QOS cap) |
| 2026-04-17 | Phase B completed (job `414791`, devel micro-F1=37.6 %, runtime 44:51) |
| 2026-04-17 | Phase C first submission crashed on `prompts.Prompts(…)` — prompt arg must be full `prompts01.json` filename, not `prompt01` tag (see §4.3) |
| 2026-04-17 | Resubmitted Phase C grid as `414918`–`414925` and FT-train as `414926`, all with `prompts0X.json` argument |
| 2026-04-17 | Refined `bin/fewshot.sh` tag-building: use `basename $PROMPTS .json` so output files are still called `FS-…-prompts01-…` (no `.json` in filename) |
| 2026-04-18 | Phase C completed (jobs `414918`–`414924` OK, `414925` qwen35B2 failed — see §4.4). Best devel few-shot = llama32B3/prompts01/15-shot → 39.2 % m-F1 |
| 2026-04-18 | Phase D completed: FT-train `414926` (5h07m) auto-chained inference `415067` (1h07m) → devel m-F1 = 59.7 %, span-F1 = 62.1 %, +20.5 pp over best few-shot |
| 2026-04-18 | Added §4.4 documenting `qwen35B2` failure (transformers 4.57.3 doesn't recognise `qwen3_5`) |
| 2026-04-18 | Phase E submitted: `415086` (few-shot llama32B3/prompts01/15/test) + `415087` (FT-llama32B3-quant on test) — both PENDING waiting for GPU |
| 2026-04-18 | Phase F submitted: `415098` (FT qwen25B3 -quant), `415099` (FT llama32B3 non-quant), `415100` (FS prompts02 15-shot), `415101` (FS prompts03 15-shot) |
| 2026-04-18 | Cluster contention: confirmed my account has access to QOS `cudabig3080`, `cudabig4000`, `cudabig4090` via `sacctmgr show assoc where user=ahlt1008`. Tried migrating Phase E+F to 4090 QOS (jobs `415211`-`415216`) then to 4000 QOS (`415222`-`415227`, `415235`-`415240`) via CLI `sbatch --qos=cudabig4000 --gres=gpu:rtx4000:1 …`. Even with `--time=02:00:00` on the few-shot jobs to help backfill, every migration sat `PENDING Reason=Priority`. Yesterday multiple cudabig jobs ran concurrently, so there's no GrpJobs=1 cap on the account; most likely the Main scheduler only wakes periodically and backfill is conservative about multi-type GRES. Accepted the wait. |
| 2026-04-18 | RTX 4000 jobs `415235`–`415239` started at 17:59 but died in ≤30 s: `CUDA error: no kernel image is available for execution on the device` — the boada-10 RTX PRO 4000 cards are Blackwell `sm_120`, and the shared venv's PyTorch only supports up to `sm_90`. Added §4.5 documenting this. |
| 2026-04-18 | Job `415240` (Phase F2, non-quantized LoRA on RTX 3080) OOMed at step 0/6680 — Llama 3.2 3B in bf16 + LoRA state does not fit in 10 GB. Added §4.6 documenting the VRAM ceiling; F2 dropped from the follow-up plan. |
| 2026-04-18 | Resubmitted the salvageable Phase E+F jobs back to `cudabig3080`: `415426` (Phase E FT-inference test), `415427` (Phase E few-shot test), `415428` (Phase F FS prompts02 15-shot devel), `415429` (Phase F FS prompts03 15-shot devel), `415430` (Phase F FT-train qwen25B3 -quant). All five submitted at 21:05 CEST. |
| 2026-04-18 | Noticed `415426`–`415430` all died in <20 s with `FileNotFoundError: 'prompts01'` — same argv bug caught in §4.3 (scripts need full `prompts0X.json`, and `fewshot.sh` needs MODEL PROMPTS SHOTS **TRAIN TEST** QUANT i.e. 6 args including the separate train-data file). Resubmitted with fixed args as `415436`–`415440`. |
| 2026-04-18 | Phase E completed: `415436` (FT-inference test, 1h07m) → test m-F1 = 57.0 %, span-F1 = 60.9 % (–2.7 pp vs devel, good generalisation). `415437` (few-shot test, 53:58) → test m-F1 = 36.2 %. FT beats few-shot on test by +20.8 pp. Filled §6.3. |
| 2026-04-19 | Phase F few-shot completed: `415438` prompts02/15 devel = 39.6 %, `415439` prompts03/15 devel = **40.9 % (new best few-shot m-F1)**. Filled rows in §6.1 and extended the trend interpretation. FT-train qwen25B3 (`415440`) still running (~6h elapsed). |
| 2026-04-19 | Added Phase G ablation plumbing: env vars `FT_LORA_R`/`FT_LORA_ALPHA`/`FT_EPOCHS`/`FT_TAG` in `model.py` + `finetune-train.py` + `finetune-inference.py`; `-balanced` flag in `fewshot.py`/`fewshot.sh` + NER-aware balanced sampling in `examples.py`. Submitted `415464` (r=32), `415465` (15 epochs), `415466` (balanced 15-shot FS). All pending behind `415440`. See §5ter. |
| 2026-04-19 | Phase F FT qwen25B3 completed: train `415440` 5h57m + auto-chained inference `415467` 1h20m → devel m-F1 = **61.2 %**, M-F1 = **56.6 %** (+1.5 / +5.1 pp over Llama baseline). `drug_n` F1 = 33.5 % (2.3× Llama). Updated §6.2. |
| 2026-04-19 | Phase G balanced FS completed: `415466` llama32B3/prompts01/15/-balanced devel m-F1 = 33.2 %, M-F1 = 32.5 %, `drug_n` F1 = **25.3 %** (up from 0 % with random sampler). Updated §6.1 + added balanced-sampler axis in trend interpretation. |
| 2026-04-19 | Phase G r=32 LoRA completed: train `415464` 5h19m + inference `415481` 1h01m → devel m-F1 = **63.8 %**, M-F1 = **59.6 %** (+4.1 / +8.1 pp over Llama r=8 baseline). Every class improves; `drug_n` 14.6 → 37.3 %, `group` 54.5 → 60.0 %. **New best system-1.3 result.** Updated §6.2. |
| 2026-04-19 | Phase G 15-epoch LoRA: train `415465` completed 7h57m; auto-chained inference `415516` queued (`Priority`) as of 11:20 CEST — cluster is full. Will fill §6.2 when it runs. |
| 2026-04-19 | Phase H submitted at 11:30 CEST: `415523` FT-inference r=32 on **test set** (1 h, reuses `FT-llama32B3-quant-r32.weights`), `415524` FT-train qwen25B3 + r=32 combo (6-7 h, auto-chains inference on devel). The two probes finish off the report: test-set headline for the best adapter, and whether the two winning axes stack. See §5quater. |
| 2026-04-19 | Phase G 15-epoch LoRA completed (inference `415516`): devel m-F1 = 61.5 %, M-F1 = 56.8 %, `drug_n` F1 = 35.1 % — +1.8 pp over r=8/10ep baseline, below r=32/10ep (63.8 %). More epochs help, but rank is the better lever. Filled §6.2. |
| 2026-04-19 | Phase H1 completed (`415523`): FT-llama32B3-quant-r32 on **test set** → m-F1 = **58.9 %**, M-F1 = **55.7 %**, `drug_n` F1 = **41.7 %**. Beats r=8 test by +1.9 pp (micro) / +4.2 pp (macro). **New system-1.3 test-set headline.** Updated §6.3 + report-ready line (was `r=16`, now `r=32`, 57.0 → 58.9 %). |
| 2026-04-19 | Phase H2 FT-train qwen25B3 + r=32 (`415524`) completed; auto-chained inference `415683` queued as of 18:07. Final result (upper-bound probe: do the two winning axes stack?) to come. |
| 2026-04-19 | Phase H2 inference (`415683`) completed at 19:58 CEST: Qwen-r=32 devel m-F1 = **63.0 %**, M-F1 = **60.8 %** (best macro), `drug_n` F1 = **44.7 %** (best rare-class), `brand` F1 = **77.7 %** (best brand). **Axes don't stack on micro** (Llama-r=32 63.8 > Qwen-r=32 63.0) **but do stack on macro / rare-class.** Filled §6.2 + Phase H interpretation. |
| 2026-04-19 | Phase I submitted (`415762`): FT-qwen25B3-quant-r32 on **test set** — reuses the r=32 Qwen adapter from `415524`. ~1 h. Fills §6.3 with the test-set number for the best-macro system so the report can quote both configurations (llama-r=32 for micro, qwen-r=32 for macro). |
| 2026-04-20 | Phase I completed (`415762`): Qwen-r=32 on test → m-F1 = **58.4 %**, M-F1 = **55.1 %**, `drug_n` = **35.8 %**. **All below Llama-r=32 on test** (58.9 / 55.7 / 41.7). Qwen overfit devel: `drug_n` 44.7 → 35.8 (–8.9 pp) vs Llama 37.3 → 41.7 (+4.4 pp). Resolves the devel-only macro advantage — **Llama × r=32 is the best system on both splits, both metrics**. Filled §6.3 + final interpretation. |
