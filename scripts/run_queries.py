#!/usr/bin/env python3
"""Execute the gold queries and every system's queries on one SPARQL endpoint.

All queries of a question are run one after the other (gold first), so that every system
sees the same state of the knowledge graph. For each prediction this means its query, the
variant with the gold LIMIT if there is one, and retry queries while the result is empty.
Predictions for a question whose gold query gives no usable answer are not run.

Results go to a working file as they arrive, so an interrupted run can simply be started
again; at the end they are written to results/executions.jsonl.gz, which is all that
scripts/evaluate.py needs.

    python scripts/run_queries.py --endpoint http://127.0.0.1:7001
"""
import argparse
import datetime
import os

from kgqa.execute import DEFINITE, QLEVER_WIKIDATA, Executor, save_executions
from kgqa.files import read_jsonl, write_json
from kgqa.metrics import gold_status
from kgqa.scoring import load_predictions, prediction_results

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PING_AFTER = 3   # failures in a row before checking whether the endpoint is still there


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


class Runner:
    """One pass over the questions. A query that fails without a definite outcome is not sent again
    in the same pass (several systems often predict the very same query)."""

    def __init__(self, executor):
        self.executor, self.failed, self.in_a_row = executor, {}, 0

    def run(self, query):
        key = query.strip()
        if key in self.failed:
            return self.failed[key]
        result = self.executor.run(query)
        if result["status"] in DEFINITE:
            self.in_a_row = 0
            return result
        self.failed[key] = result
        self.in_a_row += 1
        if self.in_a_row >= PING_AFTER:
            if not self.executor.ping():
                raise SystemExit(f"{now()} the endpoint does not answer; run the script again to continue")
            self.in_a_row = 0
        return result

    def run_question(self, question, predictions):
        gold = self.run(question["sparql"])
        if gold["status"] in DEFINITE and gold_status(question["sparql"], gold) != "ok":
            return
        for preds in predictions.values():
            pred = preds.get((question["dataset"], question["id"]))
            if pred and pred.get("query"):
                prediction_results(pred, question["sparql"], self.run)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--endpoint", default=QLEVER_WIKIDATA)
    ap.add_argument("--not-qlever", action="store_true", help="do not send QLever's timeout and send options")
    ap.add_argument("--predictions", nargs="+", default=[os.path.join(ROOT, "results", "predictions"),
                                                         os.path.join(ROOT, "results", "analysis", "oracle")])
    ap.add_argument("--store", default=os.path.join(ROOT, "cache", "executions.jsonl"), help="working file of this run")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "executions.jsonl.gz"))
    a = ap.parse_args()

    questions = read_jsonl(os.path.join(ROOT, "data", "test_questions.jsonl"))
    predictions = load_predictions(*a.predictions)
    os.makedirs(os.path.dirname(os.path.abspath(a.store)), exist_ok=True)
    executor = Executor(a.endpoint, store=a.store, qlever=not a.not_qlever)
    if not executor.ping():
        raise SystemExit(f"{a.endpoint} does not answer a test query")
    info = {"endpoint": a.endpoint, "started": now(), "index_at_start": executor.index_info(), "systems": sorted(predictions)}

    for attempt in (1, 2):   # the second pass retries what had no definite outcome in the first
        runner = Runner(executor)
        for n, question in enumerate(questions, 1):
            runner.run_question(question, predictions)
            if n % 100 == 0:
                print(f"{now()} pass {attempt}: {n}/{len(questions)} questions, {len(executor.results)} results", flush=True)
        if not runner.failed:
            break

    info.update(finished=now(), index_at_end=executor.index_info(), results=len(executor.results),
                without_definite_outcome=[{"query": q, "status": r["status"], "message": r.get("message")} for q, r in runner.failed.items()])
    info["index_unchanged"] = info["index_at_start"] == info["index_at_end"]
    save_executions(a.out, executor.results)
    write_json(os.path.join(os.path.dirname(a.out), "run_info.json"), info)
    print(f"{now()} {len(executor.results)} results -> {a.out}; {len(runner.failed)} queries without a definite outcome; "
          f"index unchanged: {info['index_unchanged']}")


if __name__ == "__main__":
    main()
