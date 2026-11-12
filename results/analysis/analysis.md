# Analysis

Descriptive results; F1 in points with 95% bootstrap intervals. Written by `scripts/analysis/report.py`.

## 1. Skeleton or linking?

SPLIT-42 on the scored questions whose gold query has a skeleton. Rows: whose skeleton; columns: whose IDs.
The fourth cell (gold skeleton, gold IDs) is the gold query itself.

| Test set | n | model skeleton, our linker | model skeleton, gold IDs | gold skeleton, our linker | gain from gold skeleton | gain from gold IDs |
|---|---|---|---|---|---|---|
| lcquad | 997 | 60.6 [57.6, 63.5] | 72.0 [69.2, 74.7] | 85.4 [83.1, 87.5] | +24.8 [22.1, 27.4] | +11.4 [9.5, 13.4] |
| heldout | 511 | 54.2 [49.9, 58.4] | 63.9 [59.7, 68.0] | 84.9 [81.8, 87.9] | +30.7 [26.8, 34.7] | +9.6 [7.2, 12.2] |
| qald10 | 373 | 18.2 [14.4, 22.2] | 18.8 [14.9, 22.8] | 83.4 [79.4, 87.1] | +65.1 [60.2, 70.0] | +0.5 [-0.5, 1.6] |

## 2. Right structure, wrong IDs

Each system's query with its i-th entity and i-th property replaced by the i-th of the gold query.

| Test set | n | E2E-42 | with gold IDs | gain | SPLIT-42 | with gold IDs | gain |
|---|---|---|---|---|---|---|---|
| lcquad | 999 | 20.7 [18.2, 23.3] | 86.9 [84.8, 89.0] | +66.2 [63.3, 69.1] | 60.5 [57.6, 63.6] | 86.5 [84.4, 88.5] | +26.0 [23.2, 28.8] |
| heldout | 512 | 27.6 [23.8, 31.5] | 77.6 [73.9, 81.2] | +50.0 [45.4, 54.6] | 54.1 [49.8, 58.5] | 78.4 [74.8, 81.9] | +24.3 [20.2, 28.4] |
| qald10 | 374 | 9.8 [6.9, 12.8] | 22.0 [17.9, 26.2] | +12.2 [8.8, 15.7] | 18.2 [14.3, 22.1] | 22.2 [18.0, 26.5] | +4.0 [1.1, 6.9] |

## 3. Structure of the predicted query

Share of scored questions whose prediction has the structure of the gold query (text comparison after masking
entities, properties, literals, numbers and variable names).

| System | lcquad | heldout | qald10 |
|---|---|---|---|
| FYP-plain | 6.3 | 3.7 | 0.0 |
| FYP-RAG | 41.9 | 39.8 | 1.9 |
| COPY | 39.7 |  |  |
| COPY-B |  | 0.0 |  |
| E2E-42 | 85.8 | 74.6 | 2.4 |
| E2E-43 | 85.0 | 70.9 | 3.2 |
| E2E-B-42 | 79.4 | 0.0 |  |
| E2E-RAG | 79.8 |  | 2.4 |
| SPLIT-42 | 85.3 | 75.8 | 2.9 |
| SPLIT-43 | 84.7 | 73.4 | 3.2 |
| SPLIT-B-42 | 79.1 | 0.0 |  |
| Qwen-FS | 24.4 | 1.8 | 4.0 |
| Sonnet-FS-run1 | 45.8 | 25.8 | 5.9 |
| Sonnet-FS-run2 | 44.7 | 24.4 | 3.5 |
| Sonnet-FS-direct | 51.8 | 26.2 | 5.3 |
| Sonnet-mem | 1.4 | 0.0 | 5.3 |
| Sonnet-mem-direct | 7.3 | 0.2 | 8.0 |
| GRISP | 82.4 |  | 29.7 |
| GRISP-WDQL |  |  | 0.8 |

## 4. Linker recall

The linker is given the name used in the gold skeleton (the English label) for every distinct ID of a question.

| Test set | entities | first candidate | in the top 10 | properties | first candidate | in the top 10 |
|---|---|---|---|---|---|---|
| lcquad | 1409 | 88.6% | 99.3% | 1657 | 100.0% | 100.0% |
| heldout | 845 | 90.5% | 99.5% | 967 | 100.0% | 100.0% |
| qald10 | 580 | 87.6% | 98.6% | 646 | 100.0% | 100.0% |

## 5. SPLIT-42's wrong answers on LC-QuAD

406 of the 999 scored questions have F1 below 1 (394.1 points lost in total). Each is assigned to the first level at which the prediction differs from the gold query, by comparing text and IDs only.

| First difference | Questions | Share | Points lost |
|---|---|---|---|
| structure of the skeleton | 137 | 34% | 129.1 |
| a name is not a label or alias of the gold ID | 160 | 39% | 156.3 |
| names right, first candidate is not the gold ID | 104 | 26% | 103.7 |
| none of these (e.g. a literal) | 5 | 1% | 5.0 |

## 6. Verbatim copies of the gold query

Scored questions whose executed query equals the gold query up to PREFIX declarations, whitespace and case.
Sonnet-mem writes IDs from memory and gets no examples from us; E2E-42 was trained on LC-QuAD.

| System | lcquad | heldout | qald10 |
|---|---|---|---|
| Sonnet-mem | 0 of 999 | 0 of 512 | 4 of 374 |
| Sonnet-mem-direct | 0 of 999 | 0 of 512 | 2 of 374 |
| E2E-42 | 143 of 999 | 117 of 512 | 0 of 374 |
