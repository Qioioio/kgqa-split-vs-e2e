"""The released result files are consistent with each other and with the code."""
import json
import os

import pytest

from kgqa.files import read_json, read_jsonl
from kgqa.postprocess import extract_query
from kgqa.scoring import load_predictions

import evaluate

ROOT = evaluate.ROOT
PREDICTIONS = load_predictions(os.path.join(ROOT, "results", "predictions"))


def without_grisp(obj):
    """The GRISP predictions are not distributed; drop every entry that names them."""
    if isinstance(obj, dict):
        return {k: without_grisp(v) for k, v in obj.items() if "GRISP" not in k}
    if isinstance(obj, list):
        return [without_grisp(v) for v in obj]
    return obj


def test_every_prediction_belongs_to_a_test_question():
    questions = {(q["dataset"], q["id"]): q for q in read_jsonl(os.path.join(ROOT, "data", "test_questions.jsonl"))}
    assert len(questions) == 1791
    assert {q["question_source"] for q in questions.values()} == {"question", "paraphrase"}
    for name, rows in PREDICTIONS.items():
        assert rows, name
        for key, row in rows.items():
            assert questions[key]["question"] == row["question"], (name, key)


def test_stored_queries_are_what_the_code_extracts_from_the_outputs():
    for name, rows in PREDICTIONS.items():
        for key, row in rows.items():
            if "slots" in row:        # skeleton systems: the query is the skeleton with the first candidates filled in
                skeleton = extract_query(row["output"])[0]
                assert (row["query"] is None) == (skeleton is None or any(not s["candidates"] for s in row["slots"])), (name, key)
            else:
                assert row["query"] == extract_query(row["output"])[0], (name, key)


@pytest.mark.slow
def test_the_committed_tables_follow_from_the_committed_predictions_and_executions():
    summary, _, _ = evaluate.evaluate(ROOT)
    computed = json.loads(json.dumps(summary))      # tuples -> lists, as in the file
    committed = read_json(os.path.join(ROOT, "results", "summary.json"))
    if "GRISP" in PREDICTIONS:
        assert computed == committed
        with open(os.path.join(ROOT, "results", "tables.md"), encoding="utf-8") as f:
            assert evaluate.to_markdown(summary) == f.read()
    else:
        assert without_grisp(computed) == without_grisp(committed)
