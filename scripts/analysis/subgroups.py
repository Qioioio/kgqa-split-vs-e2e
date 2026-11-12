#!/usr/bin/env python3
"""Two ways of splitting the 1,000 LC-QuAD test questions, used in the subgroup analysis.

    entities   seen:    every entity of the gold query occurs in some training query
                        (questions without entities count as seen)
               unseen:  at least one entity never occurs in training
    examples   covered: the three training questions retrieved for the question (the ones in the
                        prompts of FYP-RAG and E2E-RAG) together contain all entities of the gold query
               not_covered: otherwise

Writes data/splits/subgroups.json.
"""
import argparse
import os
import re

from kgqa.files import read_json, read_jsonl, write_json

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENTITY = re.compile(r"(?:\bwd:|<http://www\.wikidata\.org/entity/)(Q\d+)\b")


def entities(sparql):
    return set(ENTITY.findall(sparql))


def main():
    argparse.ArgumentParser(description=__doc__.split("\n")[0]).parse_args()   # only --help
    data = os.path.join(ROOT, "data")
    train = {r["id"]: r["sparql"] for r in read_jsonl(os.path.join(data, "clean", "train.jsonl"))}
    in_training = set()
    for row in read_jsonl(os.path.join(data, "train", "train_e2e.jsonl")):
        in_training |= entities(row["messages"][-1]["content"])
    retrieved = {r["id"]: r["examples"] for r in read_jsonl(os.path.join(ROOT, "results", "predictions", "FYP-RAG.jsonl"))
                 if r["dataset"] == "lcquad2"}
    gold = {q["id"]: q["sparql"] for q in read_jsonl(os.path.join(data, "test_questions.jsonl")) if q["dataset"] == "lcquad2"}
    out = {"entities": {"seen": [], "unseen": []}, "examples": {"covered": [], "not_covered": []}}
    for qid in read_json(os.path.join(data, "splits", "test_sets.json"))["lcquad"]:
        needed = entities(gold[qid])
        out["entities"]["seen" if needed <= in_training else "unseen"].append(qid)
        in_examples = set().union(*(entities(train[x]) for x in retrieved[qid]))
        out["examples"]["covered" if needed <= in_examples else "not_covered"].append(qid)
    write_json(os.path.join(data, "splits", "subgroups.json"), out)
    print({kind: {name: len(ids) for name, ids in groups.items()} for kind, groups in out.items()})


if __name__ == "__main__":
    main()
