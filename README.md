# Drug Named-Entity Recognition & Drug–Drug Interaction Extraction

Biomedical NLP pipeline for extracting **drug entities** and **drug–drug interactions (DDI)**
from pharmacological texts. Each task is solved with four families of approaches and compared
head-to-head: a rule-based baseline, classical machine learning, neural networks, and large
language models (few-shot prompting and fine-tuning).

The corpus is the **DDI / SemEval drug-interaction dataset** (clinical and DrugBank texts in
XML), with separate train / devel / test splits and gold-standard outputs for both tasks.

## Tasks & approaches

### Task 1 — Named-Entity Recognition (drug / brand / group / drug_n)
| Folder | Approach |
|--------|----------|
| `code/1.0.NERC-baseline` | Rule-based / dictionary lookup baseline |
| `code/1.1.NERC-ML`       | Classical ML — CRF, SVM, MaxEnt with engineered features (affixes, word shape, POS/lemma, char n-grams, dictionary spans) |
| `code/1.2.NERC-NN`       | Neural sequence labelling (embeddings + BiLSTM, GloVe / pretrained variants) |
| `code/1.3.NERC-LLM`      | LLMs — few-shot prompting and LoRA fine-tuning (Llama 3.2, Qwen 2.5) |

### Task 2 — Drug–Drug Interaction extraction (mechanism / effect / advise / int)
| Folder | Approach |
|--------|----------|
| `code/2.0.DDI-baseline` | Rule/pattern-based baseline |
| `code/2.1.DDI-ML`       | Classical ML — SVM / MaxEnt over lexical & syntactic-path features, two-stage detection + classification, threshold tuning, ensembling |
| `code/2.2.DDI-NN`       | Neural network (embeddings + CNN/LSTM with relative-position features) |
| `code/2.3.DDI-LLM`      | LLMs — few-shot prompting and LoRA fine-tuning |

## Selected results

Classical ML on Task 1 NER (test F1): the best configuration was a **CRF with the full
feature set (mod8), c1=0.1, c2=0.1 — 68.2% F1**, with SVM (rbf/linear) and MaxEnt close
behind (~67.8–67.9%). Multi-token dictionary span features improved generalization to the
test set. Full ablations, hyperparameter sweeps and per-class breakdowns for every task and
approach are in the reports.

- `report/report_task1.pdf` — NER (baseline → ML → NN → LLM)
- `report/report_task2.pdf` — DDI (baseline → ML → NN → LLM)

## Repository layout

```
code/        source for each task/approach (bin/ = Python, *.sh = run scripts)
data/        DDI corpus: train/devel/test XML + gold-standard outputs
util/        shared evaluator and data-conversion helpers
report/      final reports (PDF)
figures/     figures used in the reports + the plot-generation scripts
```

## Running

```bash
pip install -r code/requirements.txt
python3 -m spacy download en_core_web_trf   # for ML feature extraction

# example: run a task pipeline
cd code/2.0.DDI-baseline/bin && python3 run.py
```

Each task reads `data/` and writes predictions/scores into a local `results/` folder
(git-ignored). Evaluation uses `util/evaluator.py`.

### External resources

The drug dictionaries used by the dictionary-based features (DrugBank, HSDB) are **not**
included to keep the repo lean. The ML code expects them under a top-level `resources/`
directory (see `paths.py` in each task). Provide your own `DrugBank.txt` / `HSDB.txt` to
enable the dictionary features; the models run without them, just with lower recall on
dictionary-matched entities.
