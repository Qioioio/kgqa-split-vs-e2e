#!/usr/bin/env python3
"""Download GRISP's published outputs and write them as prediction files.

This way GRISP is scored on the same questions, endpoint and metric as the other systems.

The files are downloaded from the authors' server (they are not redistributed here), checked
against the SHA256 of the copies we used, and converted to
    results/predictions/GRISP.jsonl        LC-QuAD 2.0 (our 1,000 questions) and QALD-10
    results/predictions/GRISP-WDQL.jsonl   QALD-10, the variant trained on WDQL

    python scripts/baselines/grisp_outputs.py
"""
import argparse
import os
import urllib.request

from kgqa.files import read_json, read_jsonl, sha256, write_jsonl
from kgqa.postprocess import extract_query

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = "https://ad-publications.cs.uni-freiburg.de/grisp/benchmark/wikidata/"
QALD = "grisp.qwen-2.5-7b-instruct-{}-09-26-qald10-{}-30-09-26.evaluation.json"
# local name -> (URL, SHA256)
FILES = {
    "lcquad2.json": (BASE + "lcquad2-qlever/outputs/grisp.qwen-2.5-7b-instruct-wikidata-09-26-lcquad2-qlever-twetp-08-10-26.evaluation.json",
                     "6d8640192cbca786813ea0bbe300ff8210944b335583f19be3518571a084874d"),
    "qald10.json": (BASE + "qald10/outputs/" + QALD.format("wikidata", "baseline"),
                    "ab045cce02f427ef9d28552ed73c152d1ce59aab4e053cc20649a00201d79e9a"),
    "qald10_wdql.json": (BASE + "qald10/outputs/" + QALD.format("wdql-full-wikidata", "twetp"),
                         "6125cd7495f73fcf8faa5955df067d85bb8c041bb261bae141ec2644909e0d14"),
}
SYSTEMS = {"GRISP": [("lcquad2", "lcquad2.json"), ("qald10", "qald10.json")], "GRISP-WDQL": [("qald10", "qald10_wdql.json")]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", default=os.path.join(ROOT, "data", "raw", "grisp"), help="where the downloaded files are kept")
    a = ap.parse_args()
    os.makedirs(a.dir, exist_ok=True)
    for name, (url, digest) in FILES.items():
        path = os.path.join(a.dir, name)
        if not os.path.exists(path):
            print("downloading", url)
            urllib.request.urlretrieve(url, path)
        if sha256(path) != digest:
            raise SystemExit(f"{path}: SHA256 differs from the file we used ({digest}); the published file may have changed")

    questions = read_jsonl(os.path.join(ROOT, "data", "test_questions.jsonl"))
    lcquad = set(read_json(os.path.join(ROOT, "data", "splits", "test_sets.json"))["lcquad"])
    for system, sources in SYSTEMS.items():
        outputs = {dataset: read_json(os.path.join(a.dir, name)) for dataset, name in sources}
        rows = []
        for q in questions:
            if q["dataset"] not in outputs or (q["dataset"] == "lcquad2" and q["id"] not in lcquad):
                continue
            output = (outputs[q["dataset"]][q["id"]].get("prediction") or {}).get("sparql") or ""
            rows.append({"dataset": q["dataset"], "id": q["id"], "question": q["question"], "output": output,
                         "query": extract_query(output)[0]})
        write_jsonl(os.path.join(ROOT, "results", "predictions", f"{system}.jsonl"), rows)
        print(f"{system}: {len(rows)} predictions")


if __name__ == "__main__":
    main()
