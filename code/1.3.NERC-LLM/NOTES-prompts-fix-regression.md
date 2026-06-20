# Prof's 2026-04-23 `prompts.py` fix — regression note (System 1.3)

## What the professor changed

On 2026-04-23 the professor posted a corrected `prompts.py` on the FIB notices page
("Error in prompts.py", Thu 23/04/2026 18:10), explaining that the shipped
`prompts.py` was "repeating the instructions too many times, making the prompt
longer and thus more difficult for the model to understand the few-shot
examples." The fix was scoped explicitly to few-shot: *"This should affect
only the few-shot experiments, not the fine-tuning."*

The structural difference is:

| Version | Message structure for N-shot |
|---|---|
| Shipped (buggy) | `sys + (N × [user: usrprompt + TEXT: input ; assistant: gold]) + user: usrprompt + TEXT: query` → usrprompt repeated **N+1** times |
| Prof's fix | `sys + user: usrprompt + (N × [user: TEXT: input ; assistant: gold]) + user: usrprompt + TEXT: query` → usrprompt repeated **2** times |

For 15-shot that is 16× → 2× — a large token-count reduction.

## What we did

1. Swapped in prof's `prompts.py` on Boada.
2. Re-ran all 12 few-shot configurations with the fixed prompts.py
   (script: [`code/1.3.NERC-LLM/bin/rerun_fewshot_all.sh`](code/1.3.NERC-LLM/bin/rerun_fewshot_all.sh)).
3. All 12 re-run `.stats` files pulled locally and diffed against the pre-fix backups.

## What happened — **a regression, not an improvement**

We expected FS scores to climb by 2–5 pp on micro-F1. Instead:

| Config            | OLD m-F1 | NEW m-F1 |   Δ m  | OLD M-F1 | NEW M-F1 |   Δ M  |
|-------------------|---------:|---------:|-------:|---------:|---------:|-------:|
| p01 / 0-shot      |      2.1 |      2.3 |   +0.2 |      1.9 |      1.4 |   -0.5 |
| p01 / 3-shot      |     36.3 |     37.2 |   +0.9 |     34.5 |     33.9 |   -0.6 |
| p01 / 5-shot      |     46.9 |     42.6 |   -4.3 |     40.1 |     36.3 |   -3.8 |
| p01 / 10-shot     |     45.3 |     47.1 |   +1.8 |     41.5 |     43.1 |   +1.6 |
| p01 / 15-shot     |     50.3 |     49.3 |   -1.0 |     39.6 |     42.9 |   +3.3 |
| p01 / 15 balanced |     45.2 |     48.1 |   +2.9 |     41.8 |     42.4 |   +0.6 |
| p01 / 15 **test** |     48.8 |     46.3 |   -2.5 |     31.4 |     30.6 |   -0.8 |
| p02 / 5-shot      |     46.4 |     42.1 |   -4.3 |     34.2 |     30.3 |   -3.9 |
| p02 / 15-shot     |     50.9 |     43.2 |   -7.7 |     36.4 |     31.3 |   -5.1 |
| p03 / 5-shot      |     51.3 | **22.8** | **-28.5** |    33.5 | **16.0** | **-17.5** |
| p03 / 15-shot     |     54.2 |     47.2 |   -7.0 |     35.3 |     28.6 |   -6.7 |
| Qwen p01 / 5-shot |     42.7 |     35.1 |   -7.6 |     25.4 |     20.9 |   -4.5 |

Most configurations got worse; several got dramatically worse (`p03/5` collapsed
by 28.5 pp m-F1). Two small gains (`p01/10`, `p01/15 balanced`), one near-tie
(`p01/15`).

## Likely cause

Prof's fix produces a conversation with **two user turns in a row without an
assistant turn between them**:

```
system: sysprompt
user:   usrprompt                  ← no assistant reply
user:   TEXT: shot_1 input         ← 2nd consecutive user turn
assistant: gold_1
user:   TEXT: shot_2 input
...
```

In Llama-3 and Qwen chat templates, consecutive user messages are not idiomatic;
different tokenizers handle this differently (some merge, some insert empty
assistant turns, some raise). Llama-3.2-3B-Instruct at 4-bit seems to lose
its in-context-learning grip when the structural alternation is broken —
especially on `prompts03`, which has the tersest instructions.

The shipped (buggy) version attaches `usrprompt` to every demonstration. It
bloats the token count and is technically wasteful, but it keeps the
`user/assistant` alternation strict, which the model handles better.

## Decision

**Reverted everything back to pre-fix state.**

- `code/1.3.NERC-LLM/bin/prompts.py` — restored to the shipped (buggy) version
  that the report's numbers were produced with.
- `code/1.3.NERC-LLM/bin/prompts.py.prof-fix` — prof's version kept alongside,
  for reference / future comparison.
- `code/1.3.NERC-LLM/bin/prompts.py.pre-prompts-fix` — exact snapshot of the
  shipped version (same content as the active file).
- All 12 FS `.stats / .out / .json` files — restored from the
  `.pre-prompts-fix` backups (local and on Boada).
- **The report is NOT updated.** Its FS numbers remain the shipped-version
  ones, which are the numbers that were actually produced by the shipped
  pipeline the rest of the course evaluated on.

## What this does NOT change

- `examples.py` offset fix (the `cs -= openlen` shift) — that is correct
  and stays in place; both versions of `prompts.py` use the same `examples.py`.
- Fine-tuning results — the prof explicitly noted FT is unaffected by the
  prompts.py bug, and our re-run confirmed it (FT wasn't re-run anyway, as
  the pipeline loads prompts.py with `fs_examples=[]` which triggers neither
  codepath fully).
- System 1.1 and System 1.2 — entirely independent from this code path.

## Backup trail

- Git tag: `backup-pre-prompts-fix-2026-04-23` at commit `de68048`
- `.pre-prompts-fix` copies of all 12 FS stats files (local + Boada)
- `report.pre-prompts-fix.pdf` and `report/main.pre-prompts-fix.tex` snapshot
- Rerun produced `.stats / .out / .json` were inspected but not kept (no
  longer on disk after revert); can be regenerated any time by flipping
  `prompts.py → prompts.py.prof-fix` and re-running `rerun_fewshot_all.sh`.
