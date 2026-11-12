# Data

## Sources

| Dataset | Used for | Source | Licence |
|---|---|---|---|
| LC-QuAD 2.0 (Dubey et al., ISWC 2019) | training, test | version prepared for QLever by the University of Freiburg: `https://ad-publications.cs.uni-freiburg.de/grisp/benchmark/wikidata/lcquad2-qlever/` | CC BY 3.0 (original dataset, according to its [dataset card](https://huggingface.co/datasets/lc_quad)) |
| QALD-10 (Usbeck et al., Semantic Web journal) | second test set | same place, `.../qald10/test.jsonl` | MIT (original dataset; text in `licenses/`) |
| Wikidata labels and aliases | skeletons, linking | Wikidata Query Service | CC0 |

The Freiburg versions differ from the originals in the SPARQL text (rewritten to run on QLever and on current Wikidata). Their server states no separate licence for these files. If you are an author of one of these datasets and would like the copies here removed, please open an issue.

## What is in this folder

| File | Contents |
|---|---|
| `test_questions.jsonl` | The 1,791 test questions with their gold queries: 1,397 from LC-QuAD 2.0 and the 394 of the QALD-10 test set. `question_source` is `paraphrase` for the 27 LC-QuAD questions whose original text was unusable and was replaced by the first paraphrase. `structure` is the number of the query structure (LC-QuAD only). The order of the file is the order used in the bootstrap. |
| `splits/test_sets.json` | Which LC-QuAD questions belong to which test set: `lcquad` (1,000), `lcquad_control` (the 885 of them without a held-out structure), `heldout_comp` (385), `heldout_zero` (127). 115 questions are in both `lcquad` and a held-out list. |
| `splits/structures.json` | The 35 query structures of the raw training set, most frequent first, with question type, number of training questions, and number of questions in the test pool and in the 1,000. Four of them occur once and are broken queries. |
| `splits/subgroups.json` | Two partitions of the 1,000 questions used in the subgroup analysis (`scripts/analysis/subgroups.py`). |
| `labels/*.jsonl.gz` | English label (or the language-neutral `mul` label) and aliases of every entity and property in the gold queries, and of all 14,017 Wikidata properties. |
| `licenses/` | The licence of QALD-10. |

`scripts/prepare_data.py` downloads the raw files into `raw/` (checked by SHA-256) and writes `clean/` and `train/`. These are not in the repository; the script is deterministic, so the files it writes should have these SHA-256 values:

| File | Rows | SHA-256 |
|---|---|---|
| `train/train_e2e.jsonl` | 21,989 | `d56091821c3265ef57412d60277e9e5a068d989169564f5a5abc8dd33330b3fa` |
| `train/train_split.jsonl` | 21,989 | `547897e2bb5aad9de33a39f54bdd39d5ee20c51d8c769f0e8515a484d05df857` |
| `train/train_e2e_B.jsonl` | 19,079 | `0e73a590784ca47180a8e6e749421963ad2195c50b1c4e95ae4ddbd30ddc054d` |
| `train/train_split_B.jsonl` | 19,079 | `cbba5947af13bb0afe66003b36dfbf93d77a66fc2f780857b854f6afad1735ec` |
| `train/dev_e2e.jsonl` | 1,160 | `5a6abbb89abb50fdf5e50559d77b4d00249fc08d0887769af064808ef7e4f743` |
| `train/dev_split.jsonl` | 1,160 | `8e356f76d693cf501befd43149a0ece4b77ff0f3e9648cbef9aee5e43694462d` |
| `train/dev_e2e_B.jsonl` | 1,029 | `a2938cf159f5cd4f0dd63d2945b51577f32a8a78c03bf9596231bb7d8ea16035` |
| `train/dev_split_B.jsonl` | 1,029 | `6f4c808a064a5838bab5bae4661a227a4bb1e1d804c48b11b134b754a3972034` |

## How the split was made

Cleaning (`src/kgqa/clean.py`), from 24,124 raw training and 6,028 raw test questions:

- 779 training and 208 test questions have no usable text, even after falling back to the first paraphrase; 4 + 1 queries do not parse. They are dropped.
- A development set of 5% (1,167 questions) is cut from the remaining training questions by groups: questions that share a gold query or a question text stay on the same side. This leaves 22,174 training questions.
- 180 test questions whose gold query or question text also occurs in the training set are dropped.
- 1,015 test questions whose gold query returned nothing when the GRISP authors ran it are dropped, and 26 with more than 1,024 gold rows are set aside. This leaves a test pool of 4,598.

185 training and 7 development questions contain an entity or property without an English or `mul` label and cannot be written as a skeleton; they are left out of the training files of *both* E2E and SPLIT, so the two are trained on the same 21,989 questions.

Test sets (`scripts/sample_test_sets.py`, drawn against `https://qlever.dev/api/wikidata`):

- `lcquad`: 1,000 questions drawn from the pool, stratified by query structure. By question type: 668 plain, 115 ASK, 72 label filter + LIMIT, 71 two columns, 45 COUNT, 29 ORDER BY.
- Held-out structures: five structures are removed from the training files of the `-B` models. Two of them (structures 4 and 12, 2,231 raw training questions) are built from parts that other structures still contain (`heldout_comp`); the other three are all ORDER BY structures there are (24, 25, 26; 921 raw training questions), so nothing similar is left (`heldout_zero`). The test lists hold every pool question with such a structure.
- A question is only used if its gold query returned a usable answer on the day of sampling. Gold queries are executed again in the test run, on a different Wikidata snapshot; a question whose gold query is then unusable is left out by `scripts/evaluate.py`.

Three test questions have small flaws that were found after the lists were fixed (a near-duplicate question text in training; a QALD-10 gold query that equals an LC-QuAD training query; one question that is in `heldout_comp` twice under two ids). They were kept; the experiment notes give the results without them.

## The label snapshot

`labels/` was fetched from the Wikidata Query Service, 200 IDs per request, with

```sparql
SELECT ?item ?kind ?lang ?label WHERE {
  VALUES ?item { wd:Q1045 wd:P1082 ... }
  { ?item rdfs:label ?label . BIND("label" AS ?kind) }
  UNION { ?item skos:altLabel ?label . BIND("alias" AS ?kind) }
  BIND(LANG(?label) AS ?lang) FILTER(?lang IN ("en", "mul")) }
```

The English label is used when there is one, otherwise the `mul` label. For IDs without any label, `owl:sameAs` was followed to find redirects (151 entities), and the rest were checked for whether they still exist (3 entities and 37 properties no longer do). The script that did this is not included.
