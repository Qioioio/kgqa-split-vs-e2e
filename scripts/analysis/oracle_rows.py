#!/usr/bin/env python3
"""Diagnostic rows that use the gold query, to see where the points are lost.

    oracle-skeleton        the gold skeleton, linked by our linker
    oracle-ids             SPLIT-42's own skeleton, with the gold ID in every slot whose name is a label
                           or alias of a gold entity (property) of the question; other slots as linked
    oracle-position-NAME   NAME's query with its i-th entity (property) replaced by the i-th entity
                           (property) of the gold query, for SPLIT-42 and E2E-42

The first two split SPLIT's errors into skeleton and linking; they are defined for questions
whose gold query has a skeleton (every ID has a label). The third shows how often a system has
the right structure with the wrong IDs. The files go to results/analysis/oracle/ and are executed
by scripts/run_queries.py like any prediction; scripts/analysis/report.py scores them.

    python scripts/analysis/oracle_rows.py
"""
import argparse
import os

from kgqa.files import read_jsonl, write_jsonl
from kgqa.linking import EntitySearch, PropertyIndex, link, property_frequencies
from kgqa.postprocess import extract_query
from kgqa.skeleton import fill, find_slots, ids_by_kind, label_names, strip_prefixes, to_skeleton

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def rewrite(query, slots, new_ids):
    """Replace the slots by the given IDs (None: leave the slot as it is)."""
    out, pos = [], 0
    for slot, new in zip(slots, new_ids, strict=True):
        out += [query[pos:slot[0]], f"{slot[2]}:{new}" if new is not None else query[slot[0]:slot[1]]]
        pos = slot[1]
    out.append(query[pos:])
    return "".join(out)


def with_gold_ids_by_name(query, gold_sparql, names, candidates):
    """oracle-ids. `candidates` are the linker's candidate lists for the named slots, in order.
    None if a slot matches no gold name and has no candidate either."""
    slots, gold = find_slots(query), ids_by_kind(gold_sparql)
    distinct = {kind: list(dict.fromkeys(ids)) for kind, ids in gold.items()}
    new, k = [], 0
    for *_, kind, _wid, name in slots:
        if name is None:
            new.append(None)
            continue
        hit = next((g for g in distinct[kind] if name.strip().casefold() in names.get(g, ())), None)
        if hit is None:
            if not candidates[k]:
                return None
            hit = candidates[k][0]
        new.append(hit)
        k += 1
    return rewrite(query, slots, new)


def with_gold_ids_by_position(query, gold_sparql, candidates=None):
    """oracle-position. Slots beyond the number of gold IDs keep what they had (the linker's first
    candidate for a named slot). None if such a named slot has no candidate."""
    slots, gold = find_slots(query), ids_by_kind(gold_sparql)
    seen, new, k = {"E": 0, "P": 0}, [], 0
    for *_, kind, _wid, name in slots:
        i = seen[kind]
        seen[kind] += 1
        if i < len(gold[kind]):
            new.append(gold[kind][i])
        elif name is not None:
            if not candidates or not candidates[k]:
                return None
            new.append(candidates[k][0])
        else:
            new.append(None)
        k += name is not None
    return rewrite(query, slots, new)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--embedder", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--vector-cache")
    a = ap.parse_args()
    data, predictions = os.path.join(ROOT, "data"), os.path.join(ROOT, "results", "predictions")
    out = os.path.join(ROOT, "results", "analysis", "oracle")
    questions = read_jsonl(os.path.join(data, "test_questions.jsonl"))
    label_rows = [r for n in ("entity_labels", "property_labels") for r in read_jsonl(os.path.join(data, "labels", f"{n}.jsonl.gz"))]
    labels, names = {r["id"]: r for r in label_rows}, label_names(label_rows)
    skeleton, gold_slots = {}, {}
    for q in questions:
        s, slots = to_skeleton(q["sparql"], labels)
        if s is not None and fill(s, [wid for _, wid, _ in slots]) == strip_prefixes(q["sparql"]):
            skeleton[(q["dataset"], q["id"])] = s
            gold_slots[(q["dataset"], q["id"])] = [wid for _, wid, _ in slots]

    # oracle-skeleton: the linker with the settings used for SPLIT-42 (entity names must be in the cache)
    entities = EntitySearch(os.path.join(ROOT, "results", "linking", "entity_search.jsonl"))
    index = PropertyIndex(os.path.join(data, "labels", "property_labels.jsonl.gz"),
                          property_frequencies(os.path.join(data, "train", "train_e2e.jsonl")), a.embedder, a.vector_cache)
    memo = {}

    def properties(name):
        if name not in memo:
            memo[name] = index.candidates(name)
        return memo[name]

    wanted = [name for s in skeleton.values() for _, _, prefix, _, _, name in find_slots(s) if prefix == "wd"]
    entities.search(wanted)
    rows = []
    for q in questions:
        key = (q["dataset"], q["id"])
        row = {"dataset": q["dataset"], "id": q["id"], "query": None}
        if key in skeleton:
            linked = link(skeleton[key], lambda n: entities.lookup(n) or [], properties)
            row["query"] = linked["query"]
            row["slots"] = [{**slot, "gold": wid} for slot, wid in zip(linked["slots"], gold_slots[key], strict=True)]
        else:
            row["covered"] = False
        rows.append(row)
    write_jsonl(os.path.join(out, "oracle-skeleton.jsonl"), rows)

    split = {(r["dataset"], r["id"]): r for r in read_jsonl(os.path.join(predictions, "SPLIT-42.jsonl"))}
    e2e = {(r["dataset"], r["id"]): r for r in read_jsonl(os.path.join(predictions, "E2E-42.jsonl"))}
    by_name, by_position = [], {"SPLIT-42": [], "E2E-42": []}
    for q in questions:
        key = (q["dataset"], q["id"])
        base = {"dataset": q["dataset"], "id": q["id"]}
        own = extract_query(split[key]["output"])[0]   # SPLIT-42's skeleton, with PREFIX declarations
        candidates = [s["candidates"] for s in split[key]["slots"]]
        row = {**base, "query": with_gold_ids_by_name(own, q["sparql"], names, candidates) if own and key in skeleton else None}
        if key not in skeleton:
            row["covered"] = False
        by_name.append(row)
        by_position["SPLIT-42"].append({**base, "query": with_gold_ids_by_position(own, q["sparql"], candidates) if own else None})
        full = e2e[key]["query"]
        by_position["E2E-42"].append({**base, "query": with_gold_ids_by_position(full, q["sparql"]) if full else None})
    write_jsonl(os.path.join(out, "oracle-ids.jsonl"), by_name)
    for name, rows in by_position.items():
        write_jsonl(os.path.join(out, f"oracle-position-{name}.jsonl"), rows)
    print(f"{len(questions)} questions, {len(skeleton)} with a gold skeleton -> {out}")


if __name__ == "__main__":
    main()
