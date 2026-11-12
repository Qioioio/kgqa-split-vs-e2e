# Experiment notes

What was decided before the test run, how the run went, where it departed from the plan, and what was added afterwards.

## 1. Fixed before the test run

- **One test run.** All systems were executed and scored once on the test questions. No test scores existed before that. Development questions were used only to check that the outputs had the expected format and that the scoring code ran; development scores are not reported and were not used to choose any setting.
- **The five comparisons** in the README: their test sets, the tests (two-sided paired bootstrap; one non-inferiority test with a 3-point margin), Holm correction across the five, and the rule that SPLIT vs E2E is only read if the second training seed points the same way.
- **The systems.** Training recipe, prompts, decoding settings, the linker and the eight few-shot examples were fixed before each system was run on test questions. Three epochs and the last checkpoint, with no model selection. For SPLIT the development loss was lowest after the second epoch (0.087) and rose in the third (0.104); as planned, the last checkpoint was used.
- **The scoring rules** (section 3).
- **The diagnostics**: the skeleton/linking breakdown, gold IDs put in by position, structure match and linker recall (sections 1-4 of `results/analysis/analysis.md`), the check for verbatim copies of gold queries (section 6), and the three subgroup contrasts at the end of `results/tables.md`.

## 2. The test run

The record is `results/run_info.json`; the endpoint is described in `local_qlever.md`.

- QLever 0.6.0 on a laptop, with an index built from the official Wikidata dump of 5 November 2026. Server-side timeout 30 s, at most 5,000 result rows kept per query, one request per second.
- All queries of a question (gold first, then every system) were sent one after the other, so every system saw the same data. The index did not change during the run, and 200 gold queries executed at the start and again at the end gave the same results.
- The run executed 30,692 distinct queries in 9.2 hours. The 26,661 that belong to the rows of this repository are in `results/executions.jsonl.gz`: 13,726 with results, 11,192 empty, 1,693 rejected as invalid, 37 timeouts, 13 engine errors. The others belong to side analyses (section 4) that are not part of this repository.
- Scored questions: 999 of 1,000 on LC-QuAD, 512 of 512 with held-out structures, 374 of 394 on QALD-10 (there 8 gold queries return nothing on this snapshot, 11 are COUNT queries that return 0, and 1 is rejected by QLever).

## 3. Scoring

- A prediction is the query extracted from the model output (`src/kgqa/postprocess.py`): the first query-like span, cut after its closing brace and any solution modifiers, with missing standard PREFIX declarations added. No output, or nothing that looks like a query, scores 0. So does a query that is rejected, times out or hits an engine error, and, for skeleton systems, a skeleton with a name for which the linker finds no candidate.
- F1 is GRASP's answer-set F1 (`src/kgqa/grasp_metrics.py`, copied unchanged), after normalising literals: numbers by value, midnight timestamps as dates, strings without language tag. Columns are matched by assignment, so variable names do not matter and extra predicted columns are not penalised. For results of more than 1,024 rows, rows are compared exactly. Results of more than 5,000 rows are compared on their first 5,000 (42 results, used by 50 scored predictions).
- ASK questions: a SELECT prediction scores 0 (strict). The column "lenient ASK" counts a non-empty SELECT result as "yes".
- "Gold LIMIT" is a second reading for ORDER BY questions: if the gold query ends in `ORDER BY ... LIMIT n` and the prediction ends in an ORDER BY, the prediction is also run with `LIMIT n`. It matters on the held-out ORDER BY questions, where Sonnet-FS scores 29.9 under the main rule and 55.7 under this one: it often writes the right ordering without the limit.
- A question is scored only if its gold query gives a usable answer (not empty, no error, not a COUNT of zero, at most 1,024 rows) and every query of every system for it ended with a definite outcome. Otherwise it is left out for all systems.
- "SPLIT + retry": if the first query returns nothing, up to two further candidate combinations are tried, ordered by the sum of candidate ranks; in practice only the first two names of a skeleton get their second candidate. The main SPLIT row uses the first candidates only.
- Linking orders properties with the same label by how often they occur in the training queries of the model that wrote the skeleton. For the systems that were not trained (Qwen-FS, Sonnet-FS) the frequencies come from the full training set, except on held-out-structure questions, where they come from the training set without those structures. This includes the 115 held-out-structure questions that are also among the 1,000.
- GRISP's published outputs contain no query for some questions (85 of 999 on LC-QuAD; 87 and 42 of 374 on QALD-10 for the two variants). These score 0.

## 4. Departures from the plan, and known problems

**Endpoint.** The plan was to use the public QLever endpoint. Because of its load, its maintainers suggested running my own, so I built an index locally from an official dump. Before the test run the development questions were run on both: all 295 gold queries got the same verdict, 3,232 of 3,245 predictions the same execution status, and 5 scores differed.

**One question lost to an unforeseen server response.** When QLever estimates that a sort cannot finish in time, it cancels the query with HTTP 429 and a message without the word "timeout". The rules, written before I had seen this, treat a 429 without a timeout message as a transient failure. This happened to one query (Sonnet-mem on LC-QuAD `test_1712`), in the main pass and again in the retry pass, so the question is left out for all systems. Counting it as a timeout instead (`evaluate.py --missing-as-failed`) changes SPLIT vs E2E from +39.82 to +39.78, SPLIT vs Sonnet-FS on seen structures from +7.98 to +7.96, and E2E-RAG vs E2E from +6.46 to +6.45.

**Errors in the middle of a response.** QLever streams results; when it fails after the first rows have gone out, it has already sent HTTP 200 and appends an error note to the broken JSON. Before the run I decided to read the note and classify such a response as a timeout or an engine error. Two predicted queries ended this way. Leaving their questions out (`--drop lcquad2:test_5626 qald10:test_237`) moves the five planned differences by at most 0.11 points.

**Only the first query of an output is scored.** Claude sometimes writes a query, then "Wait" or "Correction" and a second query. The extraction rule keeps the first. Outputs with a second query: 48 and 65 in the two Sonnet-FS runs, 16 for Sonnet-FS-direct, 16 for Sonnet-mem, 233 for Sonnet-mem-direct (and 20 and 36 for FYP-plain and FYP-RAG, which keep writing after the query). The second queries were not executed. If all of them were right, Sonnet-FS would gain at most 2.5 points on the questions with seen structures and SPLIT vs Sonnet-FS there would be +5.4 [2.6, 8.2] instead of +8.0 [5.3, 10.7]; the other planned comparisons would move in SPLIT's favour or not at all. Sonnet-mem-direct could gain up to 14.5 points on LC-QuAD, so the gap between Sonnet-mem with thinking (39.8) and without (18.9) is partly a matter of this rule.

**The label-service rewrite is unsafe in one case.** QLever does not support `SERVICE wikibase:label`, so the post-processing replaces it by `OPTIONAL { ?x rdfs:label ?xLabel . FILTER(LANG(?xLabel) = "en") }`. If `?x` is not bound elsewhere, this matches every label in Wikidata and the query times out. Only Sonnet-mem writes such queries. A guarded variant of the rewrite, run as a side analysis, changed Sonnet-mem by +0.4 points on LC-QuAD and no other system.

**The thesis recipe does not teach the model to stop.** It sets the pad token to `<|im_end|>`, the token that ends an answer, and padding is masked out of the loss. FYP-plain therefore often keeps writing after the query until the 128-token limit (1,075 of 1,791 test outputs; FYP-RAG, which has complete queries in its prompt, 385). The extraction rule keeps a solution modifier up to the end of its line, so text after `LIMIT` breaks many of these queries. With a more forgiving cut, which I defined after seeing development outputs and therefore did not use for the main table, FYP-plain scores 8.6 instead of 4.1 on LC-QuAD and FYP-RAG 21.5 instead of 19.0 (side analysis). The recipe also seeds only the trainer, after the LoRA weights are initialised, so the FYP adapter is not exactly reproducible. `scripts/train.py` trains on the answer and its closing `<|im_end|>` only, and seeds before the model is loaded.

**A dropped row.** A variant of E2E-RAG with the IDs in the retrieved examples masked was planned. It failed a validity check defined in advance on development questions and was not run on the test set.

**Missing PREFIX declarations.** In a skeleton, a name that contains `#` or `'` (`wd:<<Lorraine-Dietrich #6>>`, `wd:<<Shaquille O'Neal>>`) is read by the post-processing as the start of a comment or a string, so prefixes used after it may not be declared and QLever rejects the query. This affects one or two queries per skeleton system. Not fixed, so that the scores stay as they were run.

**Three flawed test questions**, found by a leakage check after the lists were fixed and before the test run: a held-out question whose text differs from a training question only in case and punctuation (its gold query is different), a QALD-10 question whose gold query equals an LC-QuAD training query up to variable names, and one question that is in the held-out list twice under two IDs. The lists were kept. Without the three (`--drop lcquad2:test_4698 lcquad2:test_5475 qald10:test_307`), SPLIT-B vs Sonnet-FS is −23.0 [−26.6, −19.4] on 510 questions and SPLIT vs Sonnet-FS on QALD-10 is −40.0 [−45.4, −34.6] on 373.

**Claude settings.** The model rejects sampling settings other than the defaults, so its outputs vary between runs. Sonnet-FS and Sonnet-mem ran with the model's default (adaptive) thinking. The "-direct" rows, added as exploratory rows, set `thinking` to `between_tools` (`configs/claude.json`), the lowest setting this model accepts; with it the model did not think on any question. Sonnet-FS thought on 8 of 1,791 questions in each run, Sonnet-mem on 1,630 (the `usage` field of the prediction files).

**Retrieval for E2E-RAG.** E2E-RAG uses the three training questions that FYP-RAG retrieved in its own run (the `examples` field of both files), so the two see the same examples. Recomputing the retrieval elsewhere can pick a different example for a few questions, because some training questions are near-duplicates and their similarity scores tie.

## 5. Added after scoring

These were not planned and are descriptive:

- the means behind the subgroup contrasts (end of `results/tables.md`);
- F1 by question type;
- the breakdown of SPLIT's wrong answers by first difference (section 5 of `results/analysis/analysis.md`). It compares text and IDs only, so a question whose gold query is itself debatable still counts as a system error.

## 6. Checks

**Leakage.** None of the 1,000 test questions shares its gold query with the raw training set, or its question text (up to case and whitespace) with the training or development files. One (`test_2439`) matches a development question if punctuation is ignored; development questions are not trained on.

**A second scoring script.** After scoring, the per-question scores, the table means and the five comparisons were recomputed from the raw files by a second script, written separately from the scoring code and without access to it. The results were identical. This was a check within the project, not an external review.

**This release.** The code was reorganised after the run, and each step was checked against the files of the original run. `prepare_data.py` writes byte-identical training files; `sample_test_sets.py`, fed the gold results of the sampling day, draws the same lists; `link.py` reproduces all seven skeleton systems' prediction files byte for byte from the cached entity searches; `evaluate.py` reproduces every per-question score, mean, interval and p-value of the original scoring. The post-processing, the query executor and `run_queries.py` were compared with the original code on all model outputs and on simulated server responses, and give the same results. `train.py`, `generate.py` and the thesis-recipe scripts were compared with the originals on a CPU with a small model (Qwen2.5-0.5B), with identical adapters and outputs; they were not re-run on a GPU. Re-running `copy_nearest.py` picks a different, identical training question for one test question (a tie); the file here is the one that was executed.
