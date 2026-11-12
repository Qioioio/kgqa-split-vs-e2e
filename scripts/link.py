#!/usr/bin/env python3
"""Link the skeletons of one prediction file: names -> Wikidata IDs.

Adds to every row the query to execute ("query"), the slots with their candidates ("slots")
and, with --retries, the next-best queries ("retry_queries").

    python scripts/link.py results/predictions/SPLIT-42.jsonl --train data/train/train_e2e.jsonl --retries 2
    python scripts/link.py results/predictions/SPLIT-B-42.jsonl --train data/train/train_e2e_B.jsonl
    python scripts/link.py results/predictions/Sonnet-FS-run1.jsonl --train data/train/train_e2e.jsonl \
        --train-heldout data/train/train_e2e_B.jsonl

--train is the training file with full queries of the model that wrote the skeletons; it is only
used to order properties with the same name by frequency. Systems that were not trained use the
full training set, except on the held-out-structure questions (--train-heldout), where they get
the same frequencies as the model trained without those structures.

Entity names that are not in the cache yet are searched online; --offline forbids that.
"""
import argparse
import os

from kgqa.files import read_json, read_jsonl, write_jsonl
from kgqa.linking import EntitySearch, PropertyIndex, link, named_slots, property_frequencies
from kgqa.postprocess import extract_query

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("predictions")
    ap.add_argument("--train", required=True)
    ap.add_argument("--train-heldout")
    ap.add_argument("--retries", type=int, default=0)
    ap.add_argument("--out", help="default: rewrite the predictions file")
    ap.add_argument("--labels", default=os.path.join(ROOT, "data", "labels", "property_labels.jsonl.gz"))
    ap.add_argument("--entity-cache", default=os.path.join(ROOT, "results", "linking", "entity_search.jsonl"))
    ap.add_argument("--embedder", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--vector-cache", help=".npy file for the property vectors (computed and saved if missing)")
    ap.add_argument("--offline", action="store_true")
    a = ap.parse_args()

    rows = read_jsonl(a.predictions)
    index = {False: PropertyIndex(a.labels, property_frequencies(a.train), a.embedder, a.vector_cache)}
    heldout = set()
    if a.train_heldout:
        sets = read_json(os.path.join(ROOT, "data", "splits", "test_sets.json"))
        heldout = set(sets["heldout_comp"]) | set(sets["heldout_zero"])
        index[True] = index[False].with_frequencies(property_frequencies(a.train_heldout))

    entities = EntitySearch(a.entity_cache)
    names = []
    for r in rows:
        query, _ = extract_query(r["output"])
        names += [name for _, _, prefix, name in named_slots(query or "") if prefix == "wd"]
    missing = [n for n in dict.fromkeys(names) if n.strip() and entities.lookup(n) is None]
    if missing and a.offline:
        raise SystemExit(f"--offline: {len(missing)} names are not in the cache, e.g. {missing[:3]}")
    entities.search(missing)

    memo = {}

    def properties(name, which):
        if (which, name) not in memo:
            memo[(which, name)] = index[which].candidates(name)
        return memo[(which, name)]

    out = a.out or a.predictions
    linked = []
    for r in rows:
        which = r["dataset"] == "lcquad2" and r["id"] in heldout
        res = link(r["output"], lambda n: entities.lookup(n) or [], lambda n, w=which: properties(n, w), a.retries)
        if not a.retries:
            del res["retry_queries"]
        linked.append({**{k: v for k, v in r.items() if k not in ("query", "slots", "retry_queries")}, **res})
    write_jsonl(out, linked)
    print(f"{len(linked)} rows -> {out}: {sum(r['query'] is not None for r in linked)} with a query, "
          f"{sum(r['query'] is None and bool(r['slots']) for r in linked)} with a slot that has no candidate")


if __name__ == "__main__":
    main()
