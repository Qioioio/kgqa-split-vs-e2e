#!/usr/bin/env python3
"""Draw the LC-QuAD test sets from the cleaned test pool.

    lcquad          1,000 questions, stratified by query structure
    lcquad_control  the questions of `lcquad` that do not use a held-out structure
    heldout_comp    every usable pool question with one of the two held-out structures whose
                    parts still occur in the training data
    heldout_zero    every usable pool question with one of the three held-out ORDER BY structures,
                    of which nothing similar is left in the training data

A question is only used if its gold query returns a usable answer on the endpoint now: a
non-empty result of at most 1,024 rows, and not a COUNT of zero. Within a structure, questions
are tried in a fixed random order, so a question that fails is replaced by the next one.

This step depends on the state of Wikidata on the day it runs. The lists in data/splits/ were
drawn on the public endpoint https://qlever.dev/api/wikidata; running the script again will
give somewhat different lists, so it writes to data/splits/new/ unless told otherwise.

    python scripts/sample_test_sets.py
"""
import argparse
import collections
import os
import random

from kgqa.clean import HELDOUT_STRUCTURES, family, structure
from kgqa.execute import DEFINITE, QLEVER_WIKIDATA, Executor
from kgqa.files import read_jsonl, write_json
from kgqa.metrics import gold_status

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = 20261030
N_TEST = 1000
HELDOUT = HELDOUT_STRUCTURES


def allocate(counts, total):
    """Share `total` among the keys in proportion to their counts (largest remainders, ties to the smaller key)."""
    whole = sum(counts.values())
    quotas = {k: total * v / whole for k, v in counts.items()}
    out = {k: int(q) for k, q in quotas.items()}
    for k in sorted(counts, key=lambda k: (-(quotas[k] - out[k]), k))[: total - sum(out.values())]:
        out[k] += 1
    return out


def fixed_order(ids, tag):
    ids = sorted(ids)
    random.Random(f"{SEED}:{tag}").shuffle(ids)
    return ids


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--endpoint", default=QLEVER_WIKIDATA)
    ap.add_argument("--data", default=os.path.join(ROOT, "data"))
    ap.add_argument("--out", help="where test_sets.json and structures.json go; default: <data>/splits/new")
    ap.add_argument("--store", default=os.path.join(ROOT, "cache", "sampling.jsonl"), help="working file with the gold results")
    a = ap.parse_args()
    out = a.out or os.path.join(a.data, "splits", "new")
    test = read_jsonl(os.path.join(a.data, "clean", "test.jsonl"))
    by_id = {r["id"]: r for r in test}
    os.makedirs(os.path.dirname(os.path.abspath(a.store)), exist_ok=True)
    executor = Executor(a.endpoint, store=a.store)
    status = {}

    def usable(qid):
        if qid not in status:
            result = executor.run(by_id[qid]["sparql"])
            if result["status"] not in DEFINITE:
                raise SystemExit(f"{qid}: no definite result ({result['status']}); run the script again later, it resumes")
            status[qid] = gold_status(by_id[qid]["sparql"], result)
        return status[qid] == "ok"

    # COUNT questions are checked first: a count of zero is an empty answer in disguise
    pool = [r for r in test if not r["over_1024"]]
    pool = [r for r in pool if not r["is_count"] or usable(r["id"])]

    by_structure = collections.defaultdict(list)
    for r in pool:
        by_structure[r["structure"]].append(r["id"])
    quota = allocate({s: len(ids) for s, ids in by_structure.items()}, N_TEST)
    lcquad = []
    for s in sorted(quota):
        picked = 0
        for qid in fixed_order(by_structure[s], f"s1000:{s}"):
            if picked == quota[s]:
                break
            if usable(qid):
                lcquad.append(qid)
                picked += 1
    lists = {"heldout_structures": {k: list(v) for k, v in HELDOUT.items()}, "lcquad": sorted(lcquad)}
    held = {s for group in HELDOUT.values() for s in group}
    lists["lcquad_control"] = [i for i in lists["lcquad"] if by_id[i]["structure"] not in held]
    for name, group in HELDOUT.items():
        lists[f"heldout_{name}"] = [i for i in sorted(r["id"] for r in pool if r["structure"] in group) if usable(i)]
    write_json(os.path.join(out, "test_sets.json"), lists)

    # the table of structures
    raw_train = collections.Counter(structure(x["sparql"]) for x in read_jsonl(os.path.join(a.data, "raw", "lcquad2", "train.jsonl")))
    in_pool = collections.Counter(r["structure"] for r in pool)
    in_test = collections.Counter(by_id[i]["structure"] for i in lcquad)
    table = [{"id": i, "family": family(pattern), "pattern": pattern, "train_questions": n, "test_pool": in_pool.get(i, 0),
              "test_1000": in_test.get(i, 0), "heldout": next((k for k, group in HELDOUT.items() if i in group), None)}
             for i, (pattern, n) in enumerate(raw_train.most_common(), 1)]
    write_json(os.path.join(out, "structures.json"), table)
    reasons = collections.Counter(v for v in status.values() if v != "ok")
    print(f"{len(lists['lcquad'])} test questions ({len(lists['lcquad_control'])} control), held-out {len(lists['heldout_comp'])} + "
          f"{len(lists['heldout_zero'])}; gold queries checked {len(status)}, not usable {dict(reasons)}")


if __name__ == "__main__":
    main()
