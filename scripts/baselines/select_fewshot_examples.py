#!/usr/bin/env python3
"""Choose the eight fixed examples of the few-shot prompts (Qwen-FS, Sonnet-FS).

From the training questions, without the five held-out structures (so there is no ORDER BY
example): take the most frequent structure of every question type, fill up to eight with the
next most frequent structures, and draw one question from each with a fixed seed. The examples
are shown in order of structure frequency. Writes configs/fewshot_examples.json.
"""
import argparse
import collections
import os
import random

from kgqa.files import read_json, read_jsonl, write_json

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SEED, K = 20261030, 8


def select(pool, structure_of, family_of, excluded):
    """[(structure, question id)] for the K chosen structures, most frequent first."""
    by_structure = collections.defaultdict(list)
    for qid in pool:
        if structure_of[qid] not in excluded:
            by_structure[structure_of[qid]].append(qid)
    ranked = sorted(by_structure, key=lambda s: (-len(by_structure[s]), s))
    chosen = [next(s for s in ranked if family_of[s] == fam) for fam in sorted({family_of[s] for s in ranked})]
    chosen += [s for s in ranked if s not in chosen][: K - len(chosen)]
    rng = random.Random(SEED)
    return [(s, rng.choice(sorted(by_structure[s], key=lambda i: int(i.rsplit("_", 1)[1]))))
            for s in sorted(chosen, key=lambda s: (-len(by_structure[s]), s))]


def main():
    argparse.ArgumentParser(description=__doc__.split("\n")[0]).parse_args()   # only --help
    data = os.path.join(ROOT, "data")
    structure_of = {r["id"]: r["structure"] for r in read_jsonl(os.path.join(data, "clean", "train.jsonl"))}
    pool = [r["id"] for r in read_jsonl(os.path.join(data, "train", "train_split.jsonl"))]
    family_of = {s["id"]: s["family"] for s in read_json(os.path.join(data, "splits", "structures.json"))}
    heldout = read_json(os.path.join(data, "splits", "test_sets.json"))["heldout_structures"]
    picked = select(pool, structure_of, family_of, set(heldout["comp"]) | set(heldout["zero"]))
    path = os.path.join(ROOT, "configs", "fewshot_examples.json")
    description = read_json(path)["description"] if os.path.exists(path) else ""
    write_json(path, {"description": description, "ids": [qid for _, qid in picked], "structures": [s for s, _ in picked]})
    print(picked)


if __name__ == "__main__":
    main()
