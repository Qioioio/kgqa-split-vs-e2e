# Results

F1 and differences are in points (0-100), with 95% bootstrap intervals over questions.
This file is written by `scripts/evaluate.py`.

Columns: *Exact* is the share of questions with F1 = 1. *Lenient ASK* counts a non-empty SELECT answer to a yes/no
question as "yes" (the main F1 scores it 0). *Gold LIMIT* reruns a prediction that ends in ORDER BY with the LIMIT of
the gold query. Sonnet-FS is the per-question mean of its two runs (no Exact column).

## Planned comparisons

| Comparison | Test set | n | Difference | 95% CI | p | Holm p | Outcome |
|---|---|---|---|---|---|---|---|
| SPLIT vs E2E: SPLIT-42 - E2E-42 | lcquad | 999 | +39.8 | [36.3, 43.3] | < 0.0002 (two-sided) | < 0.001 | better |
| SPLIT vs Sonnet-FS, seen structures: SPLIT-42 - Sonnet-FS | lcquad_control | 884 | +8.0 | [5.3, 10.7] | < 0.0001 (one-sided, margin 3) | < 0.0005 | not worse |
| SPLIT-B vs Sonnet-FS, held-out structures: SPLIT-B-42 - Sonnet-FS | heldout | 512 | -22.9 | [-26.5, -19.3] | < 0.0002 (two-sided) | < 0.001 | worse |
| SPLIT vs Sonnet-FS, QALD-10: SPLIT-42 - Sonnet-FS | qald10 | 374 | -40.2 | [-45.5, -34.8] | < 0.0002 (two-sided) | < 0.001 | worse |
| E2E-RAG vs E2E: E2E-RAG - E2E-42 | lcquad | 999 | +6.5 | [4.1, 8.8] | < 0.0002 (two-sided) | < 0.001 | better |

p-values come from 10,000 bootstrap resamples; "<" means that no resample fell on the other side.
Second training seed (SPLIT-43 - E2E-43): +40.9 [37.3, 44.4].

## lcquad: LC-QuAD 2.0, 1,000 test questions

Scored 999 of 1000 questions. Left out: 1 x no result for a prediction of Sonnet-mem.

| System | F1 | 95% CI | Exact | F1, lenient ASK | F1, gold LIMIT |
|---|---|---|---|---|---|
| FYP-plain | 4.1 | [2.9, 5.3] | 3.8 | 4.1 | 4.1 |
| FYP-RAG | 19.0 | [16.6, 21.4] | 17.6 | 19.2 | 19.0 |
| COPY | 11.4 | [9.5, 13.3] | 8.4 | 14.0 | 11.4 |
| E2E-42 | 20.7 | [18.2, 23.3] | 20.1 | 20.8 | 20.7 |
| E2E-43 | 20.0 | [17.6, 22.5] | 19.1 | 20.2 | 20.0 |
| E2E-B-42 | 13.4 | [11.3, 15.6] | 12.8 | 13.6 | 13.4 |
| E2E-RAG | 27.2 | [24.5, 30.0] | 26.2 | 27.3 | 27.2 |
| SPLIT-42 | 60.5 | [57.6, 63.6] | 59.4 | 60.5 | 60.5 |
| SPLIT-42 + retry | 64.8 | [61.8, 67.7] | 63.6 |  |  |
| SPLIT-43 | 60.9 | [57.8, 63.9] | 59.6 | 61.0 | 60.9 |
| SPLIT-B-42 | 56.5 | [53.4, 59.5] | 55.1 | 56.5 | 56.5 |
| Qwen-FS | 28.2 | [25.5, 30.9] | 26.2 | 28.6 | 28.3 |
| Sonnet-FS | 50.6 | [47.6, 53.5] |  | 50.6 | 51.2 |
| Sonnet-FS-run1 | 50.2 | [47.2, 53.1] | 47.6 | 50.2 | 50.8 |
| Sonnet-FS-run2 | 50.9 | [47.9, 54.0] | 48.5 | 51.0 | 51.6 |
| Sonnet-FS-direct | 52.0 | [49.0, 55.0] | 49.1 | 52.0 | 52.7 |
| Sonnet-mem | 39.8 | [36.9, 42.8] | 35.6 | 40.0 | 40.3 |
| Sonnet-mem-direct | 18.9 | [16.5, 21.3] | 16.7 | 18.9 | 19.3 |
| GRISP | 79.8 | [77.3, 82.2] | 78.3 | 79.8 | 79.8 |

| Paired difference | n | Difference | 95% CI |
|---|---|---|---|
| SPLIT-42 - E2E-42 | 999 | +39.8 | [36.3, 43.3] |
| SPLIT-43 - E2E-43 | 999 | +40.9 | [37.3, 44.4] |
| SPLIT-42 - Sonnet-FS | 999 | +10.0 | [7.4, 12.6] |
| SPLIT-B-42 - Sonnet-FS | 999 | +5.9 | [3.3, 8.5] |
| E2E-RAG - E2E-42 | 999 | +6.5 | [4.1, 8.8] |
| E2E-42 - E2E-43 | 999 | +0.7 | [-1.0, 2.5] |
| SPLIT-42 - SPLIT-43 | 999 | -0.3 | [-2.2, 1.6] |
| E2E-42 - E2E-B-42 | 999 | +7.3 | [5.4, 9.3] |
| SPLIT-42 - SPLIT-B-42 | 999 | +4.1 | [1.8, 6.4] |
| SPLIT-42 - Qwen-FS | 999 | +32.4 | [29.2, 35.6] |
| Sonnet-FS - Qwen-FS | 999 | +22.4 | [19.7, 25.2] |
| Sonnet-FS-direct - Sonnet-FS | 999 | +1.4 | [0.2, 2.6] |
| Sonnet-FS-direct - Qwen-FS | 999 | +23.8 | [21.0, 26.7] |
| Sonnet-mem-direct - Sonnet-mem | 999 | -21.0 | [-23.7, -18.2] |
| SPLIT-42 - GRISP | 999 | -19.3 | [-22.3, -16.2] |
| SPLIT-42 + retry - SPLIT-42 | 999 | +4.2 | [3.0, 5.5] |
| FYP-RAG - FYP-plain | 999 | +14.9 | [12.5, 17.4] |

## lcquad_control: LC-QuAD 2.0, the 885 of the 1,000 that do not use a held-out structure

Scored 884 of 885 questions. Left out: 1 x no result for a prediction of Sonnet-mem.

| System | F1 | 95% CI | Exact | F1, lenient ASK | F1, gold LIMIT |
|---|---|---|---|---|---|
| FYP-plain | 4.3 | [3.0, 5.6] | 4.0 | 4.3 | 4.3 |
| FYP-RAG | 18.2 | [15.7, 20.7] | 16.9 | 18.4 | 18.2 |
| COPY | 11.5 | [9.5, 13.4] | 8.7 | 14.4 | 11.5 |
| E2E-42 | 19.7 | [17.2, 22.3] | 19.1 | 19.8 | 19.7 |
| E2E-43 | 18.8 | [16.4, 21.4] | 18.0 | 19.1 | 18.8 |
| E2E-B-42 | 14.9 | [12.7, 17.3] | 14.4 | 15.2 | 14.9 |
| E2E-RAG | 25.5 | [22.7, 28.4] | 24.8 | 25.6 | 25.5 |
| SPLIT-42 | 61.0 | [57.9, 64.2] | 59.8 | 61.0 | 61.0 |
| SPLIT-42 + retry | 65.7 | [62.5, 68.8] | 64.5 |  |  |
| SPLIT-43 | 62.0 | [58.8, 65.1] | 60.7 | 62.1 | 62.0 |
| SPLIT-B-42 | 62.6 | [59.5, 65.8] | 61.5 | 62.6 | 62.6 |
| Qwen-FS | 31.4 | [28.5, 34.4] | 29.3 | 31.8 | 31.4 |
| Sonnet-FS | 53.0 | [50.0, 56.2] |  | 53.1 | 53.0 |
| Sonnet-FS-run1 | 52.4 | [49.2, 55.6] | 50.3 | 52.4 | 52.4 |
| Sonnet-FS-run2 | 53.7 | [50.5, 57.0] | 51.8 | 53.9 | 53.7 |
| Sonnet-FS-direct | 54.8 | [51.6, 58.0] | 52.6 | 54.8 | 54.8 |
| Sonnet-mem | 41.0 | [37.8, 44.1] | 37.2 | 41.2 | 41.0 |
| Sonnet-mem-direct | 19.0 | [16.5, 21.5] | 17.2 | 19.0 | 19.0 |
| GRISP | 80.8 | [78.2, 83.3] | 79.3 | 80.8 | 80.8 |

| Paired difference | n | Difference | 95% CI |
|---|---|---|---|
| SPLIT-42 - E2E-42 | 884 | +41.3 | [37.6, 45.0] |
| SPLIT-43 - E2E-43 | 884 | +43.1 | [39.5, 46.8] |
| SPLIT-42 - Sonnet-FS | 884 | +8.0 | [5.3, 10.7] |
| SPLIT-B-42 - Sonnet-FS | 884 | +9.6 | [6.9, 12.3] |
| E2E-RAG - E2E-42 | 884 | +5.8 | [3.4, 8.2] |
| E2E-42 - E2E-43 | 884 | +0.8 | [-0.9, 2.6] |
| SPLIT-42 - SPLIT-43 | 884 | -1.0 | [-2.9, 1.0] |
| E2E-42 - E2E-B-42 | 884 | +4.8 | [2.8, 6.7] |
| SPLIT-42 - SPLIT-B-42 | 884 | -1.6 | [-3.7, 0.5] |
| SPLIT-42 - Qwen-FS | 884 | +29.6 | [26.3, 33.1] |
| Sonnet-FS - Qwen-FS | 884 | +21.7 | [18.7, 24.7] |
| Sonnet-FS-direct - Sonnet-FS | 884 | +1.8 | [0.5, 3.0] |
| Sonnet-FS-direct - Qwen-FS | 884 | +23.5 | [20.4, 26.6] |
| Sonnet-mem-direct - Sonnet-mem | 884 | -22.0 | [-25.1, -19.0] |
| SPLIT-42 - GRISP | 884 | -19.7 | [-22.9, -16.6] |
| SPLIT-42 + retry - SPLIT-42 | 884 | +4.6 | [3.3, 6.0] |
| FYP-RAG - FYP-plain | 884 | +13.9 | [11.4, 16.5] |

## heldout: Held-out structures, 512 questions

Scored 512 of 512 questions. Left out: none.

| System | F1 | 95% CI | Exact | F1, lenient ASK | F1, gold LIMIT |
|---|---|---|---|---|---|
| FYP-plain | 2.3 | [1.1, 3.7] | 2.1 | 2.3 | 2.7 |
| FYP-RAG | 21.9 | [18.4, 25.5] | 20.7 | 21.9 | 22.3 |
| COPY-B | 4.3 | [2.8, 6.1] | 3.3 | 4.3 | 4.3 |
| E2E-42 | 27.6 | [23.8, 31.5] | 26.4 | 27.6 | 27.6 |
| E2E-43 | 28.0 | [24.3, 31.9] | 26.8 | 28.0 | 28.0 |
| E2E-B-42 | 1.9 | [0.8, 3.1] | 1.6 | 1.9 | 1.9 |
| SPLIT-42 | 54.1 | [49.8, 58.5] | 52.5 | 54.1 | 54.1 |
| SPLIT-42 + retry | 57.3 | [53.0, 61.6] | 55.7 |  |  |
| SPLIT-43 | 51.4 | [47.2, 55.8] | 49.8 | 51.4 | 51.4 |
| SPLIT-B-42 | 7.0 | [5.0, 9.2] | 5.1 | 7.0 | 7.0 |
| Qwen-FS | 6.9 | [5.0, 9.0] | 4.9 | 6.9 | 8.3 |
| Sonnet-FS | 29.9 | [26.4, 33.5] |  | 29.9 | 36.3 |
| Sonnet-FS-run1 | 30.1 | [26.4, 33.8] | 23.4 | 30.1 | 36.3 |
| Sonnet-FS-run2 | 29.7 | [26.2, 33.4] | 22.5 | 29.7 | 36.3 |
| Sonnet-FS-direct | 30.9 | [27.2, 34.6] | 24.2 | 30.9 | 36.9 |
| Sonnet-mem | 25.3 | [21.8, 28.8] | 17.8 | 25.3 | 28.7 |
| Sonnet-mem-direct | 14.5 | [11.8, 17.4] | 10.4 | 14.5 | 17.6 |

| Paired difference | n | Difference | 95% CI |
|---|---|---|---|
| SPLIT-42 - E2E-42 | 512 | +26.5 | [21.6, 31.4] |
| SPLIT-43 - E2E-43 | 512 | +23.4 | [18.6, 28.2] |
| SPLIT-42 - Sonnet-FS | 512 | +24.3 | [20.1, 28.5] |
| SPLIT-B-42 - Sonnet-FS | 512 | -22.9 | [-26.5, -19.3] |
| E2E-42 - E2E-43 | 512 | -0.4 | [-3.5, 2.6] |
| SPLIT-42 - SPLIT-43 | 512 | +2.7 | [-0.2, 5.6] |
| E2E-42 - E2E-B-42 | 512 | +25.7 | [21.9, 29.6] |
| SPLIT-42 - SPLIT-B-42 | 512 | +47.1 | [42.7, 51.6] |
| SPLIT-42 - Qwen-FS | 512 | +47.2 | [42.5, 52.0] |
| Sonnet-FS - Qwen-FS | 512 | +23.0 | [19.6, 26.5] |
| Sonnet-FS-direct - Sonnet-FS | 512 | +1.0 | [-0.7, 2.7] |
| Sonnet-FS-direct - Qwen-FS | 512 | +24.0 | [20.4, 27.7] |
| Sonnet-mem-direct - Sonnet-mem | 512 | -10.8 | [-13.9, -7.7] |
| SPLIT-42 + retry - SPLIT-42 | 512 | +3.2 | [1.8, 4.8] |
| FYP-RAG - FYP-plain | 512 | +19.6 | [16.2, 23.3] |

## heldout_comp: Held-out structures whose parts occur in training, 385 questions

Scored 385 of 385 questions. Left out: none.

| System | F1 | 95% CI | Exact | F1, lenient ASK | F1, gold LIMIT |
|---|---|---|---|---|---|
| FYP-plain | 2.6 | [1.0, 4.2] | 2.6 | 2.6 | 2.6 |
| FYP-RAG | 22.1 | [18.1, 26.3] | 21.0 | 22.1 | 22.1 |
| COPY-B | 5.1 | [3.1, 7.2] | 3.9 | 5.1 | 5.1 |
| E2E-42 | 26.7 | [22.4, 31.0] | 25.7 | 26.7 | 26.7 |
| E2E-43 | 27.3 | [23.1, 31.6] | 26.0 | 27.3 | 27.3 |
| E2E-B-42 | 2.5 | [1.2, 4.1] | 2.1 | 2.5 | 2.5 |
| SPLIT-42 | 51.9 | [46.9, 56.9] | 50.6 | 51.9 | 51.9 |
| SPLIT-42 + retry | 55.3 | [50.4, 60.2] | 54.0 |  |  |
| SPLIT-43 | 49.2 | [44.1, 54.1] | 47.5 | 49.2 | 49.2 |
| SPLIT-B-42 | 9.3 | [6.6, 12.1] | 6.8 | 9.3 | 9.3 |
| Qwen-FS | 8.1 | [5.6, 10.7] | 6.2 | 8.1 | 8.1 |
| Sonnet-FS | 29.9 | [25.7, 34.2] |  | 29.9 | 29.9 |
| Sonnet-FS-run1 | 30.2 | [25.8, 34.7] | 27.5 | 30.2 | 30.2 |
| Sonnet-FS-run2 | 29.6 | [25.3, 34.0] | 26.2 | 29.6 | 29.6 |
| Sonnet-FS-direct | 30.6 | [26.2, 35.1] | 27.5 | 30.6 | 30.6 |
| Sonnet-mem | 29.4 | [25.2, 33.8] | 22.3 | 29.4 | 29.4 |
| Sonnet-mem-direct | 16.3 | [12.8, 19.9] | 13.2 | 16.3 | 16.3 |

| Paired difference | n | Difference | 95% CI |
|---|---|---|---|
| SPLIT-42 - E2E-42 | 385 | +25.2 | [19.5, 30.9] |
| SPLIT-43 - E2E-43 | 385 | +22.0 | [16.5, 27.5] |
| SPLIT-42 - Sonnet-FS | 385 | +22.0 | [16.9, 27.0] |
| SPLIT-B-42 - Sonnet-FS | 385 | -20.6 | [-24.9, -16.2] |
| E2E-42 - E2E-43 | 385 | -0.6 | [-4.3, 3.3] |
| SPLIT-42 - SPLIT-43 | 385 | +2.6 | [-0.9, 6.3] |
| E2E-42 - E2E-B-42 | 385 | +24.2 | [19.9, 28.5] |
| SPLIT-42 - SPLIT-B-42 | 385 | +42.5 | [37.4, 47.7] |
| SPLIT-42 - Qwen-FS | 385 | +43.8 | [38.2, 49.2] |
| Sonnet-FS - Qwen-FS | 385 | +21.8 | [17.8, 26.0] |
| Sonnet-FS-direct - Sonnet-FS | 385 | +0.7 | [-1.4, 2.9] |
| Sonnet-FS-direct - Qwen-FS | 385 | +22.6 | [18.3, 26.9] |
| Sonnet-mem-direct - Sonnet-mem | 385 | -13.2 | [-17.2, -9.3] |
| SPLIT-42 + retry - SPLIT-42 | 385 | +3.5 | [1.8, 5.5] |
| FYP-RAG - FYP-plain | 385 | +19.5 | [15.5, 23.8] |

## heldout_zero: Held-out ORDER BY structures, 127 questions

Scored 127 of 127 questions. Left out: none.

| System | F1 | 95% CI | Exact | F1, lenient ASK | F1, gold LIMIT |
|---|---|---|---|---|---|
| FYP-plain | 1.3 | [0.1, 3.3] | 0.8 | 1.3 | 3.1 |
| FYP-RAG | 21.3 | [14.6, 28.4] | 19.7 | 21.3 | 22.6 |
| COPY-B | 2.1 | [0.0, 4.7] | 1.6 | 2.1 | 2.1 |
| E2E-42 | 30.6 | [22.9, 38.7] | 28.3 | 30.6 | 30.6 |
| E2E-43 | 30.4 | [22.6, 38.5] | 29.1 | 30.4 | 30.4 |
| E2E-B-42 | 0.0 | [0.0, 0.0] | 0.0 | 0.0 | 0.0 |
| SPLIT-42 | 61.0 | [52.6, 69.4] | 58.3 | 61.0 | 61.0 |
| SPLIT-42 + retry | 63.4 | [55.0, 71.5] | 60.6 |  |  |
| SPLIT-43 | 58.1 | [49.5, 66.5] | 56.7 | 58.1 | 58.1 |
| SPLIT-B-42 | 0.0 | [0.0, 0.0] | 0.0 | 0.0 | 0.0 |
| Qwen-FS | 3.4 | [1.6, 5.8] | 0.8 | 3.4 | 9.1 |
| Sonnet-FS | 29.9 | [24.5, 35.5] |  | 29.9 | 55.7 |
| Sonnet-FS-run1 | 29.6 | [24.1, 35.3] | 11.0 | 29.6 | 54.7 |
| Sonnet-FS-run2 | 30.1 | [24.7, 35.7] | 11.0 | 30.1 | 56.8 |
| Sonnet-FS-direct | 31.6 | [25.8, 37.6] | 14.2 | 31.6 | 56.0 |
| Sonnet-mem | 12.7 | [8.5, 17.1] | 3.9 | 12.7 | 26.2 |
| Sonnet-mem-direct | 9.2 | [6.0, 12.7] | 1.6 | 9.2 | 21.8 |

| Paired difference | n | Difference | 95% CI |
|---|---|---|---|
| SPLIT-42 - E2E-42 | 127 | +30.5 | [20.6, 40.0] |
| SPLIT-43 - E2E-43 | 127 | +27.7 | [17.8, 37.5] |
| SPLIT-42 - Sonnet-FS | 127 | +31.2 | [23.8, 38.4] |
| SPLIT-B-42 - Sonnet-FS | 127 | -29.9 | [-35.5, -24.5] |
| E2E-42 - E2E-43 | 127 | +0.2 | [-3.7, 3.9] |
| SPLIT-42 - SPLIT-43 | 127 | +3.0 | [-1.0, 7.3] |
| E2E-42 - E2E-B-42 | 127 | +30.6 | [22.9, 38.7] |
| SPLIT-42 - SPLIT-B-42 | 127 | +61.0 | [52.6, 69.4] |
| SPLIT-42 - Qwen-FS | 127 | +57.6 | [48.9, 66.2] |
| Sonnet-FS - Qwen-FS | 127 | +26.5 | [21.0, 32.2] |
| Sonnet-FS-direct - Sonnet-FS | 127 | +1.7 | [-0.7, 4.5] |
| Sonnet-FS-direct - Qwen-FS | 127 | +28.2 | [22.2, 34.5] |
| Sonnet-mem-direct - Sonnet-mem | 127 | -3.4 | [-7.1, 0.0] |
| SPLIT-42 + retry - SPLIT-42 | 127 | +2.4 | [0.0, 5.5] |
| FYP-RAG - FYP-plain | 127 | +19.9 | [13.5, 26.7] |

## qald10: QALD-10 test set, 394 questions

Scored 374 of 394 questions. Left out: 8 x gold: empty; 1 x gold: query_error; 11 x gold: count_zero.

| System | F1 | 95% CI | Exact | F1, lenient ASK | F1, gold LIMIT |
|---|---|---|---|---|---|
| FYP-plain | 2.3 | [0.9, 3.9] | 1.9 | 4.1 | 2.3 |
| FYP-RAG | 8.4 | [5.7, 11.3] | 8.0 | 10.3 | 8.4 |
| E2E-42 | 9.8 | [6.9, 12.8] | 8.8 | 13.0 | 9.8 |
| E2E-43 | 10.0 | [7.2, 13.2] | 9.6 | 12.4 | 10.0 |
| E2E-RAG | 8.8 | [6.1, 11.8] | 8.3 | 12.3 | 8.7 |
| SPLIT-42 | 18.2 | [14.3, 22.1] | 17.4 | 20.6 | 18.4 |
| SPLIT-42 + retry | 18.5 | [14.6, 22.4] | 17.6 |  |  |
| SPLIT-43 | 16.7 | [13.0, 20.4] | 15.8 | 19.4 | 16.9 |
| Qwen-FS | 31.3 | [26.6, 35.8] | 30.2 | 32.1 | 31.3 |
| Sonnet-FS | 58.3 | [53.4, 63.2] |  | 58.6 | 58.3 |
| Sonnet-FS-run1 | 58.2 | [53.1, 63.1] | 57.5 | 58.7 | 58.2 |
| Sonnet-FS-run2 | 58.5 | [53.4, 63.5] | 58.0 | 58.5 | 58.5 |
| Sonnet-FS-direct | 59.5 | [54.4, 64.4] | 58.8 | 59.5 | 59.5 |
| Sonnet-mem | 60.7 | [55.9, 65.5] | 57.8 | 60.9 | 60.7 |
| Sonnet-mem-direct | 41.5 | [36.6, 46.5] | 39.6 | 41.5 | 41.5 |
| GRISP | 40.4 | [35.4, 45.3] | 39.6 | 46.0 | 40.4 |
| GRISP-WDQL | 55.3 | [50.5, 60.2] | 51.3 | 59.9 | 55.3 |

| Paired difference | n | Difference | 95% CI |
|---|---|---|---|
| SPLIT-42 - E2E-42 | 374 | +8.4 | [5.2, 11.7] |
| SPLIT-43 - E2E-43 | 374 | +6.7 | [3.2, 10.1] |
| SPLIT-42 - Sonnet-FS | 374 | -40.2 | [-45.5, -34.8] |
| E2E-RAG - E2E-42 | 374 | -1.0 | [-3.7, 1.8] |
| E2E-42 - E2E-43 | 374 | -0.3 | [-2.2, 1.6] |
| SPLIT-42 - SPLIT-43 | 374 | +1.5 | [-0.9, 4.0] |
| SPLIT-42 - Qwen-FS | 374 | -13.1 | [-18.1, -8.3] |
| Sonnet-FS - Qwen-FS | 374 | +27.1 | [22.2, 32.1] |
| Sonnet-FS-direct - Sonnet-FS | 374 | +1.2 | [-0.6, 2.9] |
| Sonnet-FS-direct - Qwen-FS | 374 | +28.2 | [23.3, 33.3] |
| Sonnet-mem-direct - Sonnet-mem | 374 | -19.2 | [-24.2, -14.3] |
| SPLIT-42 - GRISP | 374 | -22.2 | [-28.4, -16.0] |
| SPLIT-42 + retry - SPLIT-42 | 374 | +0.3 | [0.0, 0.8] |
| FYP-RAG - FYP-plain | 374 | +6.1 | [3.2, 9.2] |
| SPLIT-42 - GRISP-WDQL | 374 | -37.1 | [-42.6, -31.5] |

## lcquad: F1 by question type

| System | ASK (115) | COUNT (45) | ORDER BY (29) | label filter + LIMIT (72) | plain (667) | two columns (71) |
|---|---|---|---|---|---|---|
| FYP-plain | 18.3 | 4.4 | 0.0 | 0.0 | 1.7 | 9.0 |
| FYP-RAG | 44.3 | 17.8 | 20.7 | 49.5 | 10.0 | 31.6 |
| COPY | 40.9 | 0.0 | 10.8 | 5.9 | 6.4 | 23.6 |
| E2E-42 | 47.0 | 20.0 | 20.7 | 56.9 | 11.0 | 33.9 |
| E2E-43 | 46.1 | 24.4 | 17.2 | 49.5 | 10.8 | 32.7 |
| E2E-B-42 | 47.0 | 6.7 | 0.0 | 36.1 | 5.2 | 22.9 |
| E2E-RAG | 53.9 | 35.6 | 24.1 | 59.1 | 15.5 | 57.0 |
| SPLIT-42 | 79.1 | 53.3 | 55.2 | 65.2 | 56.6 | 69.4 |
| SPLIT-42 + retry | 79.1 | 53.3 | 55.2 | 74.9 | 61.9 | 69.4 |
| SPLIT-43 | 80.0 | 55.6 | 51.7 | 66.6 | 57.1 | 66.4 |
| SPLIT-B-42 | 78.3 | 37.8 | 0.0 | 64.3 | 54.1 | 70.3 |
| Qwen-FS | 68.7 | 20.0 | 2.3 | 37.5 | 22.1 | 25.7 |
| Sonnet-FS | 77.4 | 38.9 | 23.7 | 63.9 | 45.7 | 57.1 |
| Sonnet-FS-run1 | 77.4 | 40.0 | 24.1 | 64.6 | 44.9 | 57.8 |
| Sonnet-FS-run2 | 77.4 | 37.8 | 23.2 | 63.2 | 46.5 | 56.5 |
| Sonnet-FS-direct | 80.9 | 33.3 | 24.4 | 63.0 | 47.8 | 56.0 |
| Sonnet-mem | 61.7 | 31.1 | 11.7 | 35.2 | 39.0 | 34.2 |
| Sonnet-mem-direct | 52.2 | 11.1 | 7.1 | 26.5 | 13.7 | 16.0 |
| GRISP | 91.3 | 71.1 | 74.4 | 90.7 | 76.6 | 87.9 |

## lcquad: subgroups

- SPLIT-42 - E2E-42: unseen - seen entities: +19.6 [12.6, 26.7] (532 vs 467 questions). Means: unseen: SPLIT-42 56.6, E2E-42 7.6; seen: SPLIT-42 65.0, E2E-42 35.6.
- E2E-RAG - E2E-42: examples cover the entities - they do not: +20.3 [13.9, 26.8] (302 vs 697 questions). Means: covered: E2E-RAG 56.7, E2E-42 36.0; not_covered: E2E-RAG 14.4, E2E-42 14.1.
- E2E-RAG - E2E-42, seen entities only: covered - not covered: +21.9 [12.5, 31.3] (302 vs 165 questions). Means: covered: E2E-RAG 56.7, E2E-42 36.0; not_covered: E2E-RAG 33.7, E2E-42 34.9.
