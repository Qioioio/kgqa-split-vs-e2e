# Skeleton-then-link vs. end-to-end generation for question answering over Wikidata

A controlled comparison of two ways to fine-tune a 7B language model to translate English questions into SPARQL queries over Wikidata:

- **E2E**: the model writes the whole query, Wikidata IDs included (`wd:Q1045 p:P1082 ?s`).
- **SPLIT**: the model writes a *skeleton* with names in place of IDs (`wd:<<Somalia>> p:<<population>> ?s`), and a separate linking step looks the names up in Wikidata.

Both use the same base model (Qwen2.5-Coder-7B-Instruct), the same QLoRA recipe, the same training questions and the same prompt. Only the training target differs.

SPLIT is a simplified re-implementation of the idea behind [GRISP](https://arxiv.org/abs/2604.21133) and [ChatKBQA](https://arxiv.org/abs/2310.08975), not a new method. The question here is how much of the gain survives under controlled conditions, and where it stops.

The project redoes my BSc final-year project (FYP). The thesis reported 71% execution accuracy on its own 59-question test set, but 35 of those 59 questions had their exact gold query in the training set. Here the thesis recipe is retrained on a deduplicated split and measured like everything else (rows FYP-plain and FYP-RAG).

## Results

Mean F1 per question (0-100). Each predicted query and each gold query is executed on the same Wikidata snapshot and the answers are compared. 95% bootstrap intervals for every row are in [`results/tables.md`](results/tables.md).

| System | LC-QuAD 2.0<br>(999 questions) | Held-out structures<br>(512) | QALD-10<br>(374) |
|---|---|---|---|
| *Fine-tuned Qwen2.5-Coder-7B* | | | |
| E2E (seed 42 / 43) | 20.7 / 20.0 | 27.6 / 28.0 | 9.8 / 10.0 |
| SPLIT (seed 42 / 43) | 60.5 / 60.9 | 54.1 / 51.4 | 18.2 / 16.7 |
| SPLIT, retry on an empty result | 64.8 | 57.3 | 18.5 |
| E2E-RAG (E2E-42, three retrieved training examples in the prompt) | 27.2 | – | 8.8 |
| E2E-B (trained without the held-out structures) | 13.4 | 1.9 | – |
| SPLIT-B (trained without the held-out structures) | 56.5 | 7.0 | – |
| *Not fine-tuned* | | | |
| COPY (query of the most similar training question) | 11.4 | 4.3 | – |
| Qwen-FS (base model, 8 examples, skeleton + linking) | 28.2 | 6.9 | 31.3 |
| Sonnet-FS (Claude Sonnet, 8 examples, skeleton + linking) | 50.6 | 29.9 | 58.3 |
| Sonnet-mem (Claude Sonnet, no examples, IDs from memory) | 39.8 | 25.3 | 60.7 |
| *Thesis recipe on the clean split* | | | |
| FYP-plain | 4.1 | 2.3 | 2.3 |
| FYP-RAG | 19.0 | 21.9 | 8.4 |
| *Published system: its authors' outputs, scored here* | | | |
| GRISP | 79.8 | – | 40.4 |
| GRISP, variant trained on WDQL | – | – | 55.3 |

- **Held-out structures.** Five query structures are removed from the training data of the "-B" models: two whose parts still occur in other structures, and all three ORDER BY structures. The column holds every test question with one of them. Rows not marked "-B" were trained on all structures and show what the same questions score when their structure was seen. In this column COPY searches only the training questions without these structures.
- **Few-shot rows.** The eight examples are fixed training questions with their skeletons. They cover every question type except ORDER BY, and none has a held-out structure. The linker is the one SPLIT uses.
- **Claude Sonnet** is `claude-sonnet-5-5` with effort "high", called through the Message Batches API. Sonnet-FS is the per-question mean of two runs (50.2 and 50.9 on LC-QuAD).
- **E2E-RAG** uses the E2E-42 adapter, which was trained without examples; the examples are added only at inference.

Five comparisons were fixed before the test run. They use a paired bootstrap and are Holm-corrected together:

| Comparison | Test set | Difference | 95% CI | Outcome |
|---|---|---|---|---|
| SPLIT vs E2E | LC-QuAD 2.0 | +39.8 | [36.3, 43.3] | better (Holm p < 0.001) |
| SPLIT vs Sonnet-FS, structures seen in training | LC-QuAD 2.0 without held-out structures (884) | +8.0 | [5.3, 10.7] | not worse (non-inferiority, 3-point margin) |
| SPLIT-B vs Sonnet-FS, held-out structures | held-out | −22.9 | [−26.5, −19.3] | worse |
| SPLIT vs Sonnet-FS | QALD-10 | −40.2 | [−45.5, −34.8] | worse |
| E2E-RAG vs E2E | LC-QuAD 2.0 | +6.5 | [4.1, 8.8] | better |

### What the numbers say

1. **On questions like the training data, writing names instead of IDs is worth about 40 points**, with both training seeds (+39.8 and +40.9). The two models write a query with the gold structure about equally often (85.8% and 85.3% of questions); they differ in the IDs. Putting the gold IDs into E2E's own queries, by position, raises its score to 86.9. On questions with an entity that never occurs in the training data, E2E scores 7.6 and SPLIT 56.6.
2. **The gain does not carry over to new structures or to another dataset.** The models trained without the held-out structures never write the structure such a question needs (0 of 512), so SPLIT-B scores 7.0 there and 0.0 on the ORDER BY structures. QALD-10 is a separate benchmark whose questions are not generated from query templates; there SPLIT scores 18.2, Claude Sonnet with eight examples and the same linker 58.3, and the base model without fine-tuning 31.3 (Qwen-FS; SPLIT − Qwen-FS = −13.1 [−18.1, −8.3], a descriptive difference).
3. **E2E-RAG's gain is concentrated where the retrieved examples contain the entities of the question**: +20.7 points on the 302 questions whose three examples contain all entities of the gold query, +0.3 on the other 697.
4. **The thesis recipe scores 4.1 (FYP-plain) and 19.0 (FYP-RAG)** on the clean split. The recipe never teaches the model to stop, and 1,075 of FYP-plain's 1,791 outputs run to the token limit; part of its low score comes from the text that follows the query (a more forgiving cut of the output gives 8.6, see the notes).
5. **SPLIT is 19 points below GRISP** on the same 999 questions. Of SPLIT's 406 wrong answers on LC-QuAD, the first difference from the gold query is the structure of the skeleton in 34%, a name that is not a label or alias of the gold entity or property in 39%, and a correct name linked to the wrong ID in 26% ([`results/analysis/analysis.md`](results/analysis/analysis.md)).

Only the five planned comparisons are tested; everything else is descriptive.

### Limitations

- One base model and one training dataset. LC-QuAD 2.0 questions follow about 30 query structures, which favours fine-tuned models on its own test set.
- Two training seeds for E2E and SPLIT, one for the other fine-tuned models. Claude Sonnet is not deterministic.
- Wikidata has changed since the datasets were made. Questions whose gold query no longer returns a usable answer are left out: 20 of the 394 QALD-10 questions, and part of the LC-QuAD pool before sampling.
- Only the first query in a model's output is scored. Claude sometimes writes a query, then a correction and a second query, which is ignored. The second queries were not executed. If every one of them were right, Sonnet-FS (48 and 65 such outputs in its two runs) would gain at most 2.5 points on the questions with seen structures, and the planned comparison there would become +5.4 [2.6, 8.2], with the same outcome. Sonnet-mem without thinking (the "-direct" row in `results/tables.md`; 233 such outputs) could gain up to 14.5 points on LC-QuAD.
- Claude Sonnet may have seen the datasets during training. Its queries almost never equal the gold queries used here (0 of 999 on LC-QuAD, 4 of 374 on QALD-10), but these gold queries are Freiburg's rewrites of the originals, so this says little about whether it has seen the original datasets.
- The linker uses Wikidata's own entity search, whose answers change over time; the answers used here are kept in `results/linking/`.
- The trained adapters are not part of this repository.

More detail, including where the run departed from the plan and the sensitivity checks, is in [`docs/experiment_notes.md`](docs/experiment_notes.md).

## Reproducing

### The tables, from the released predictions

Python 3.11 or later. No GPU, endpoint or API key is needed; each command takes a few minutes.

```bash
pip install -e . pytest
python scripts/baselines/grisp_outputs.py   # downloads GRISP's published outputs (4 MB) from its authors' server
python scripts/evaluate.py                  # results/tables.md, results/summary.json, results/per_question_f1.csv
python scripts/analysis/report.py           # results/analysis/analysis.md
python -m pytest                            # includes a check that the committed tables follow from the committed files
```

`results/predictions/` has one file per system with, for every test question, the raw model output and the query that was executed. `results/executions.jsonl.gz` has the endpoint's answer to each of the 26,661 distinct queries involved (gold and predicted), keyed by the SHA-256 of the query text, and `results/run_info.json` records the run that produced them. GRISP's outputs are not redistributed here; without them, the two scripts write the same files minus the GRISP rows.

### The whole pipeline

`pip install -r requirements.txt` for the CPU steps, and `requirements-gpu.txt` on the GPU machine.

| Step | Command | Needs |
|---|---|---|
| Data | `python scripts/prepare_data.py` | downloads 17 MB; deterministic |
| Test sets | `python scripts/sample_test_sets.py` | a SPARQL endpoint; the lists used here are in `data/splits/` |
| Training | `python scripts/train.py --train data/train/train_split.jsonl --dev data/train/dev_split.jsonl --seed 42 --model MODEL --out runs/SPLIT-42` | 24 GB GPU, about 3 hours per model |
| Generation | `python scripts/generate.py --adapter runs/SPLIT-42/final --skeleton --model MODEL --out results/predictions/SPLIT-42.jsonl` | GPU, about 1 hour |
| Linking | `python scripts/link.py results/predictions/SPLIT-42.jsonl --train data/train/train_e2e.jsonl --retries 2` | CPU; Wikidata search for names not in the cache |
| Execution | `python scripts/run_queries.py --endpoint http://127.0.0.1:7001` | a QLever endpoint with Wikidata ([`docs/local_qlever.md`](docs/local_qlever.md)) |
| Scoring | `python scripts/evaluate.py` | CPU |

`MODEL` is a local copy of Qwen2.5-Coder-7B-Instruct (the revision is in `configs/train.json`). The docstring of each script lists the variants for the other systems. The baselines are in `scripts/baselines/` (COPY, the Claude rows, the thesis recipe, GRISP's outputs), the diagnostic rows and the error breakdown in `scripts/analysis/`. The label snapshot in `data/labels/` comes from a query to the Wikidata Query Service (see `data/README.md`); the script for it is not included.

Training and generation ran on a rented RTX 4090 with the package versions in `requirements-gpu.txt`; `results/training_runs.json` records the steps, time and development loss of each run. Queries were executed on QLever 0.6.0 with an index built from the Wikidata dump of 5 November 2026.

## Layout

```
configs/      training recipe, prompts, the eight few-shot examples, Claude settings
data/         test questions, test-set lists, label snapshot; see data/README.md
src/kgqa/     cleaning, skeletons, prompts, post-processing, linking, execution, metric, statistics
scripts/      one script per pipeline step; baselines/ and analysis/
results/      predictions, execution results, per-question scores, tables, analysis
docs/         experiment notes; the local QLever endpoint
tests/        unit tests and consistency checks of the released files
```

## About this repository

This is the cleaned-up release of the project. The experiments were run from a working repository with more scaffolding; the code here was reorganised afterwards and each step was checked against the files of that run.

Jiaqi Xiong, 2026

## Licence and acknowledgements

Code: Apache-2.0. `src/kgqa/grasp_metrics.py` is copied from [GRASP](https://github.com/ad-freiburg/grasp) (Apache-2.0); see `NOTICE`.

Data: LC-QuAD 2.0 (CC BY 3.0, according to its [dataset card](https://huggingface.co/datasets/lc_quad)) and [QALD-10](https://github.com/KGQA/QALD_10) (MIT, licence text in `data/licenses/`), in the versions prepared for QLever by the Chair of Algorithms and Data Structures at the University of Freiburg, which also publishes GRISP's outputs and develops [QLever](https://github.com/ad-freiburg/qlever). `NOTICE` lists what was changed. Labels and query results are from Wikidata (CC0).
