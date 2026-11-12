"""Question-level scoring from stored execution results.

Shared by scripts/run_queries.py (which queries does a prediction need?) and
scripts/evaluate.py (what does it score?).
"""
import glob
import os

from .execute import query_key
from .files import read_jsonl
from .metrics import f1, gold_status, with_gold_limit

METRICS = ("f1", "f1_lenient_ask", "f1_gold_limit")
MAX_RETRIES = 2
# The order in which systems are listed in tables; others follow alphabetically.
ROW_ORDER = ["FYP-plain", "FYP-RAG", "COPY", "COPY-B", "E2E-42", "E2E-43", "E2E-B-42", "E2E-RAG",
             "SPLIT-42", "SPLIT-42 + retry", "SPLIT-43", "SPLIT-B-42", "Qwen-FS", "Sonnet-FS", "Sonnet-FS-run1",
             "Sonnet-FS-run2", "Sonnet-FS-direct", "Sonnet-mem", "Sonnet-mem-direct", "GRISP", "GRISP-WDQL"]


def load_predictions(*directories):
    """{system name: {(dataset, id): row}} for every .jsonl file in the directories."""
    out = {}
    for d in directories:
        for path in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
            name = os.path.basename(path)[:-len(".jsonl")]
            if name in out:
                raise ValueError(f"two prediction files are called {name}")
            out[name] = {(r["dataset"], r["id"]): r for r in read_jsonl(path)}
    return out


def prediction_results(pred, gold_query, result_of):
    """The three results a prediction is scored on: of its query, of its query with the gold LIMIT
    (the same result unless that variant exists), and after retries (the first non-empty result
    among the query and up to two retry queries, else the last one tried).
    `result_of` maps a query to its result, or to None if there is none."""
    query = pred["query"]
    main = result_of(query)
    variant = with_gold_limit(query, gold_query)
    limited = result_of(variant) if variant and variant != query else main
    final = main
    for retry in pred.get("retry_queries", [])[:MAX_RETRIES]:
        if final is None or final["status"] != "empty":
            break
        final = result_of(retry)
    return main, limited, final


def score_prediction(pred, gold_query, gold, result_of):
    """{"f1", "f1_lenient_ask", "f1_gold_limit"[, "f1_retry"]} for one prediction row,
    or None if a result it needs is missing. A row without a query scores 0."""
    has_retry = "retry_queries" in pred
    if not pred.get("query"):
        return {**{m: 0.0 for m in METRICS}, **({"f1_retry": 0.0} if has_retry else {})}
    main, limited, final = prediction_results(pred, gold_query, result_of)
    if main is None or limited is None or final is None:
        return None
    out = {"f1": f1(gold, main), "f1_lenient_ask": f1(gold, main, strict_ask=False), "f1_gold_limit": f1(gold, limited)}
    if has_retry:
        out["f1_retry"] = f1(gold, final)
    return out


def score_all(questions, predictions, executions, drop=(), missing_as_failed=False):
    """One record per question: the status of its gold query, the scores of every system that
    answered it, and whether it is scored at all. A question is scored when its gold query gave a
    usable answer and no system's query lacks a result (unless `missing_as_failed`, which scores
    such a query as 0 instead)."""
    def result_of(query):
        return executions.get(query_key(query)) or ({"status": "no_result"} if missing_as_failed else None)

    records = []
    for q in questions:
        key = (q["dataset"], q["id"])
        gold = executions.get(query_key(q["sparql"]))
        rec = {"dataset": q["dataset"], "id": q["id"], "structure": q.get("structure"),
               "gold": gold_status(q["sparql"], gold), "scores": {}, "no_result": []}
        if f"{q['dataset']}:{q['id']}" in drop:
            rec["gold"] = "dropped"
        elif rec["gold"] == "ok":
            for name, preds in predictions.items():
                if key in preds:
                    scores = score_prediction(preds[key], q["sparql"], gold, result_of)
                    if scores is None:
                        rec["no_result"].append(name)
                    else:
                        rec["scores"][name] = scores
        rec["scored"] = rec["gold"] == "ok" and not rec["no_result"]
        records.append(rec)
    return records
