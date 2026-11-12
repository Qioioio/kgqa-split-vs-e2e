#!/usr/bin/env python3
"""Where the points are lost: writes results/analysis/analysis.md. No endpoint is needed.

1. SPLIT's score with the gold skeleton, and with gold IDs (rows built by oracle_rows.py).
2. Scores after putting the gold IDs into a system's own query, slot by slot.
3. How often each system's query has the structure of the gold query.
4. How often the linker finds the gold ID when it is given the gold name.
5. SPLIT's wrong answers on LC-QuAD, by the first thing that differs from the gold query.
6. How often a query is the gold query word for word (a check for memorised test data).

All of this is descriptive: none of it was among the planned comparisons.
"""
import argparse
import collections
import os

from kgqa.clean import same_query_key, structure
from kgqa.execute import load_executions
from kgqa.files import read_json, read_jsonl
from kgqa.postprocess import extract_query
from kgqa.scoring import ROW_ORDER, load_predictions, score_all
from kgqa.skeleton import SLOT, find_slots, ids_by_kind, label_names
from kgqa.stats import mean_ci, paired_test

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SETS = {"lcquad": ("lcquad2", ["lcquad"]), "heldout": ("lcquad2", ["heldout_comp", "heldout_zero"]), "qald10": ("qald10", None)}
SKELETON_SYSTEMS = ("SPLIT-42", "SPLIT-43", "SPLIT-B-42", "Qwen-FS", "Sonnet-FS-run1", "Sonnet-FS-run2", "Sonnet-FS-direct")


def fmt(mean_and_ci):
    mean, (low, high) = mean_and_ci
    return f"{mean:.1f} [{low:.1f}, {high:.1f}]"


def diff(a, b):
    t = paired_test(a, b)
    return f"{t['diff']:+.1f} [{t['ci95'][0]:.1f}, {t['ci95'][1]:.1f}]"


def same_structure(prediction, gold, is_skeleton):
    """The prediction equals the gold query once entities, properties, literals, numbers and
    variable names are masked. Names in a skeleton are masked like IDs."""
    if not prediction:
        return False
    if is_skeleton:
        prediction = SLOT.sub(lambda m: f"{m.group(1)}:{'Q0' if m.group(1) == 'wd' else 'P0'}", prediction)
    return structure(prediction) == structure(gold)


def error_category(skeleton, slots, gold, names):
    """The first level at which SPLIT's prediction differs from the gold query:
    structure - the skeleton does not have the structure of the gold query;
    name      - same structure, but a name is not a label or alias of the gold ID at that position;
    linking   - the names are right, but the linker's first candidate for one of them is not the gold ID;
    other     - none of these (for example a different literal)."""
    if not same_structure(skeleton, gold, True):
        return "structure"
    gold_ids, seen, right_names, right_ids, k = ids_by_kind(gold), {"E": 0, "P": 0}, True, True, 0
    for *_, kind, wid, name in find_slots(skeleton):
        expected = gold_ids[kind][seen[kind]] if seen[kind] < len(gold_ids[kind]) else None
        seen[kind] += 1
        if name is None:
            right_ids &= wid == expected
        else:
            candidates = slots[k]["candidates"]
            k += 1
            right_names &= expected is not None and name.strip().casefold() in names.get(expected, ())
            right_ids &= bool(candidates) and candidates[0] == expected
    return "name" if not right_names else "linking" if not right_ids else "other"


def main():
    argparse.ArgumentParser(description=__doc__.split("\n")[0]).parse_args()   # only --help
    data, results = os.path.join(ROOT, "data"), os.path.join(ROOT, "results")
    questions = read_jsonl(os.path.join(data, "test_questions.jsonl"))
    lists = read_json(os.path.join(data, "splits", "test_sets.json"))
    predictions = load_predictions(os.path.join(results, "predictions"), os.path.join(results, "analysis", "oracle"))
    records = score_all(questions, predictions, load_executions(os.path.join(results, "executions.jsonl.gz")))
    gold = {(q["dataset"], q["id"]): q["sparql"] for q in questions}
    names = label_names(r for n in ("entity_labels", "property_labels") for r in read_jsonl(os.path.join(data, "labels", f"{n}.jsonl.gz")))
    oracle = predictions["oracle-skeleton"]
    out = ["# Analysis", "", "Descriptive results; F1 in points with 95% bootstrap intervals. Written by `scripts/analysis/report.py`.", ""]

    scored = {}
    for name, (dataset, keys) in SETS.items():
        ids = None if keys is None else {i for k in keys for i in lists[k]}
        scored[name] = [r for r in records if r["scored"] and r["dataset"] == dataset and (ids is None or r["id"] in ids)]

    out += ["## 1. Skeleton or linking?", "",
            "SPLIT-42 on the scored questions whose gold query has a skeleton. Rows: whose skeleton; columns: whose IDs.",
            "The fourth cell (gold skeleton, gold IDs) is the gold query itself.", "",
            "| Test set | n | model skeleton, our linker | model skeleton, gold IDs | gold skeleton, our linker "
            "| gain from gold skeleton | gain from gold IDs |", "|---|---|---|---|---|---|---|"]
    for name, rows in scored.items():
        rows = [r for r in rows if oracle[(r["dataset"], r["id"])].get("covered", True)]
        split, ids_, skel = ([r["scores"][s]["f1"] for r in rows] for s in ("SPLIT-42", "oracle-ids", "oracle-skeleton"))
        out.append(f"| {name} | {len(rows)} | {fmt(mean_ci(split))} | {fmt(mean_ci(ids_))} | {fmt(mean_ci(skel))} "
                   f"| {diff(skel, split)} | {diff(ids_, split)} |")

    out += ["", "## 2. Right structure, wrong IDs", "",
            "Each system's query with its i-th entity and i-th property replaced by the i-th of the gold query.", "",
            "| Test set | n | E2E-42 | with gold IDs | gain | SPLIT-42 | with gold IDs | gain |", "|---|---|---|---|---|---|---|---|"]
    for name, rows in scored.items():
        cells = []
        for system in ("E2E-42", "SPLIT-42"):
            own = [r["scores"][system]["f1"] for r in rows]
            fixed = [r["scores"][f"oracle-position-{system}"]["f1"] for r in rows]
            cells += [fmt(mean_ci(own)), fmt(mean_ci(fixed)), diff(fixed, own)]
        out.append(f"| {name} | {len(rows)} | " + " | ".join(cells) + " |")

    out += ["", "## 3. Structure of the predicted query", "",
            "Share of scored questions whose prediction has the structure of the gold query (text comparison after masking",
            "entities, properties, literals, numbers and variable names).", "",
            "| System | " + " | ".join(scored) + " |", "|---|" + "---|" * len(scored)]
    for system in sorted((s for s in predictions if not s.startswith("oracle")), key=lambda s: (ROW_ORDER.index(s) if s in ROW_ORDER else 99, s)):
        preds, cells = predictions[system], []
        for rows in scored.values():
            if any((r["dataset"], r["id"]) not in preds for r in rows):
                cells.append("")
                continue
            is_skeleton = system in SKELETON_SYSTEMS
            hits = sum(same_structure(extract_query(preds[(r["dataset"], r["id"])]["output"])[0], gold[(r["dataset"], r["id"])], is_skeleton)
                       for r in rows)
            cells.append(f"{100 * hits / len(rows):.1f}")
        out.append(f"| {system} | " + " | ".join(cells) + " |")

    out += ["", "## 4. Linker recall", "",
            "The linker is given the name used in the gold skeleton (the English label) for every distinct ID of a question.", "",
            "| Test set | entities | first candidate | in the top 10 | properties | first candidate | in the top 10 |",
            "|---|---|---|---|---|---|---|"]
    for name, rows in scored.items():
        counts = collections.Counter()
        for r in rows:
            first = {}
            for slot in oracle[(r["dataset"], r["id"])].get("slots", []):
                first.setdefault(slot["gold"], slot)
            for wid, slot in first.items():
                kind = "E" if slot["prefix"] == "wd" else "P"
                counts[kind] += 1
                counts[kind + "1"] += slot["candidates"][:1] == [wid]
                counts[kind + "10"] += wid in slot["candidates"][:10]
        out.append(f"| {name} | {counts['E']} | {100 * counts['E1'] / counts['E']:.1f}% | {100 * counts['E10'] / counts['E']:.1f}% "
                   f"| {counts['P']} | {100 * counts['P1'] / counts['P']:.1f}% | {100 * counts['P10'] / counts['P']:.1f}% |")

    wrong = [r for r in scored["lcquad"] if r["scores"]["SPLIT-42"]["f1"] < 1]
    by_category, lost = collections.Counter(), collections.Counter()
    for r in wrong:
        key = (r["dataset"], r["id"])
        pred = predictions["SPLIT-42"][key]
        category = error_category(extract_query(pred["output"])[0], pred["slots"], gold[key], names)
        by_category[category] += 1
        lost[category] += 1 - r["scores"]["SPLIT-42"]["f1"]
    out += ["", "## 5. SPLIT-42's wrong answers on LC-QuAD", "",
            f"{len(wrong)} of the {len(scored['lcquad'])} scored questions have F1 below 1 ({sum(lost.values()):.1f} points lost in total). "
            "Each is assigned to the first level at which the prediction differs from the gold query, by comparing text and IDs only.", "",
            "| First difference | Questions | Share | Points lost |", "|---|---|---|---|"]
    explain = {"structure": "structure of the skeleton", "name": "a name is not a label or alias of the gold ID",
               "linking": "names right, first candidate is not the gold ID", "other": "none of these (e.g. a literal)"}
    for category in ("structure", "name", "linking", "other"):
        out.append(f"| {explain[category]} | {by_category[category]} | {100 * by_category[category] / len(wrong):.0f}% | {lost[category]:.1f} |")

    out += ["", "## 6. Verbatim copies of the gold query", "",
            "Scored questions whose executed query equals the gold query up to PREFIX declarations, whitespace and case.",
            "Sonnet-mem writes IDs from memory and gets no examples from us; E2E-42 was trained on LC-QuAD.", "",
            "| System | " + " | ".join(scored) + " |", "|---|" + "---|" * len(scored)]
    for system in ("Sonnet-mem", "Sonnet-mem-direct", "E2E-42"):
        cells = []
        for rows in scored.values():
            keys = [(r["dataset"], r["id"]) for r in rows]
            same = sum(same_query_key(predictions[system][k]["query"] or "") == same_query_key(gold[k]) for k in keys)
            cells.append(f"{same} of {len(keys)}")
        out.append(f"| {system} | " + " | ".join(cells) + " |")

    path = os.path.join(results, "analysis", "analysis.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
