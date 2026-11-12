#!/usr/bin/env python3
"""Score every system and write the result tables. No endpoint is needed.

Reads   data/test_questions.jsonl, data/splits/,
        results/predictions/*.jsonl, results/executions.jsonl.gz
Writes  results/per_question_f1.csv, results/summary.json, results/tables.md

A system is listed for a test set when its prediction file covers every question of
the set. A question is scored only if its gold query returned a usable answer and every
query of every system for it has a stored result; otherwise it is left out for all systems.

    python scripts/evaluate.py
    python scripts/evaluate.py --drop lcquad2:test_5626 qald10:test_237   # sensitivity
    python scripts/evaluate.py --missing-as-failed --out /tmp/check        # sensitivity
"""
import argparse
import collections
import csv
import os

from kgqa.execute import load_executions
from kgqa.files import read_json, read_jsonl, write_json
from kgqa.scoring import METRICS, ROW_ORDER, load_predictions, score_all
from kgqa.stats import group_difference, holm, mean_ci, noninferiority_p, paired_test

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# name -> (title, dataset, lists in data/splits/test_sets.json; None = every question of the dataset)
TEST_SETS = {
    "lcquad": ("LC-QuAD 2.0, 1,000 test questions", "lcquad2", ["lcquad"]),
    "lcquad_control": ("LC-QuAD 2.0, the 885 of the 1,000 that do not use a held-out structure", "lcquad2", ["lcquad_control"]),
    "heldout": ("Held-out structures, 512 questions", "lcquad2", ["heldout_comp", "heldout_zero"]),
    "heldout_comp": ("Held-out structures whose parts occur in training, 385 questions", "lcquad2", ["heldout_comp"]),
    "heldout_zero": ("Held-out ORDER BY structures, 127 questions", "lcquad2", ["heldout_zero"]),
    "qald10": ("QALD-10 test set, 394 questions", "qald10", None),
}
AVERAGED = {"Sonnet-FS": ["Sonnet-FS-run1", "Sonnet-FS-run2"]}   # the API model is not deterministic: two runs, averaged per question
# The five comparisons fixed before the test run: (label, left, right, test set, test, margin in points).
COMPARISONS = [
    ("SPLIT vs E2E", "SPLIT-42", "E2E-42", "lcquad", "two-sided", 3.0),
    ("SPLIT vs Sonnet-FS, seen structures", "SPLIT-42", "Sonnet-FS", "lcquad_control", "non-inferiority", 3.0),
    ("SPLIT-B vs Sonnet-FS, held-out structures", "SPLIT-B-42", "Sonnet-FS", "heldout", "two-sided", 5.0),
    ("SPLIT vs Sonnet-FS, QALD-10", "SPLIT-42", "Sonnet-FS", "qald10", "two-sided", 5.0),
    ("E2E-RAG vs E2E", "E2E-RAG", "E2E-42", "lcquad", "two-sided", 3.0),
]
# Other paired differences, reported with intervals only (exploratory).
PAIRS = [("SPLIT-42", "E2E-42"), ("SPLIT-43", "E2E-43"), ("SPLIT-42", "Sonnet-FS"), ("SPLIT-B-42", "Sonnet-FS"),
         ("E2E-RAG", "E2E-42"), ("E2E-42", "E2E-43"), ("SPLIT-42", "SPLIT-43"), ("E2E-42", "E2E-B-42"),
         ("SPLIT-42", "SPLIT-B-42"), ("SPLIT-42", "Qwen-FS"), ("Sonnet-FS", "Qwen-FS"), ("Sonnet-FS-direct", "Sonnet-FS"),
         ("Sonnet-FS-direct", "Qwen-FS"), ("Sonnet-mem-direct", "Sonnet-mem"), ("SPLIT-42", "GRISP"),
         ("SPLIT-42 + retry", "SPLIT-42"), ("FYP-RAG", "FYP-plain"), ("SPLIT-42", "GRISP-WDQL")]


# ---------------------------------------------------------------- per test set
def members(name, questions, lists):
    _, dataset, keys = TEST_SETS[name]
    ids = None if keys is None else {i for k in keys for i in lists[k]}
    return {(q["dataset"], q["id"]) for q in questions if q["dataset"] == dataset and (ids is None or q["id"] in ids)}


def score_vectors(records, keys, predictions):
    """{row name: {metric: [score per scored question]}} for the systems that cover the whole test set."""
    scored = [r for r in records if (r["dataset"], r["id"]) in keys and r["scored"]]
    rows = {}
    for name, preds in predictions.items():
        if not keys <= set(preds):
            continue
        rows[name] = {m: [r["scores"][name][m] for r in scored] for m in METRICS}
        if any("retry_queries" in p for p in preds.values()):
            rows[name + " + retry"] = {"f1": [r["scores"][name]["f1_retry"] for r in scored]}
    for name, parts in AVERAGED.items():
        if all(p in rows for p in parts):
            rows[name] = {m: [sum(v) / len(v) for v in zip(*(rows[p][m] for p in parts), strict=True)] for m in METRICS}
            rows[name]["averaged"] = True
    order = {n: i for i, n in enumerate(ROW_ORDER)}
    return {n: rows[n] for n in sorted(rows, key=lambda n: (order.get(n, len(order)), n))}, scored


def has_scores(vec):
    return any(v["f1"] for v in vec.values())


def summarize(vec):
    x = vec["f1"]
    mean, ci = mean_ci(x)
    out = {"n": len(x), "f1": mean, "ci95": ci}
    if not vec.get("averaged"):
        out["exact"] = 100 * sum(v >= 1 - 1e-12 for v in x) / len(x)
    for m in METRICS[1:]:
        if m in vec:
            out[m] = 100 * sum(vec[m]) / len(vec[m])
    return out


def evaluate(root=ROOT, drop=(), missing_as_failed=False):
    questions = read_jsonl(os.path.join(root, "data", "test_questions.jsonl"))
    lists = read_json(os.path.join(root, "data", "splits", "test_sets.json"))
    subgroups = read_json(os.path.join(root, "data", "splits", "subgroups.json"))
    family = {s["id"]: s["family"] for s in read_json(os.path.join(root, "data", "splits", "structures.json"))}
    executions = load_executions(os.path.join(root, "results", "executions.jsonl.gz"))
    predictions = load_predictions(os.path.join(root, "results", "predictions"))
    records = score_all(questions, predictions, executions, set(drop), missing_as_failed)

    summary = {"test_sets": {}, "comparisons": [], "subgroups": {}}
    vectors = {}
    for name, (title, _, _) in TEST_SETS.items():
        keys = members(name, questions, lists)
        vec, scored = score_vectors(records, keys, predictions)
        vectors[name] = (vec, scored)
        left_out = collections.Counter(("no result for a prediction of " + ", ".join(r["no_result"])) if r["gold"] == "ok" else f"gold: {r['gold']}"
                                       for r in records if (r["dataset"], r["id"]) in keys and not r["scored"])
        pairs = {f"{a} - {b}": paired_test(vec[a]["f1"], vec[b]["f1"]) for a, b in PAIRS if a in vec and b in vec and scored}
        summary["test_sets"][name] = {"title": title, "questions": len(keys), "scored": len(scored), "left_out": dict(left_out),
                                      "systems": {n: summarize(v) for n, v in vec.items()} if scored else {}, "pairs": pairs}

    # the planned comparisons, Holm-corrected together
    pvalues = {}
    for label, left, right, test_set, test, margin in COMPARISONS:
        vec, _ = vectors[test_set]
        if left not in vec or right not in vec or not has_scores(vec):
            continue
        a, b = vec[left]["f1"], vec[right]["f1"]
        c = {"label": label, "left": left, "right": right, "test_set": test_set, "test": test, "margin": margin, **paired_test(a, b)}
        if test == "non-inferiority":
            c["p"] = noninferiority_p(a, b, margin)
        pvalues[label] = c["p"]
        summary["comparisons"].append(c)
    adjusted = holm(pvalues) if pvalues else {}
    for c in summary["comparisons"]:
        c["p_holm"], rejected = adjusted[c["label"]]
        if c["test"] == "non-inferiority":
            c["verdict"] = "not worse" if rejected else "not shown to be not worse"
        elif rejected:
            c["verdict"] = "better" if c["diff"] > 0 else "worse"
        elif -c["margin"] < c["ci90"][0] and c["ci90"][1] < c["margin"]:
            c["verdict"] = "about the same"
        else:
            c["verdict"] = "inconclusive"
    # the first comparison was to be read only if the second training seed points the same way
    second = summary["test_sets"]["lcquad"]["pairs"].get("SPLIT-43 - E2E-43")
    if summary["comparisons"] and summary["comparisons"][0]["left"] == "SPLIT-42" and second:
        first = summary["comparisons"][0]
        first["second_seed"] = {"diff": second["diff"], "ci95": second["ci95"]}
        if (second["diff"] > 0) != (first["diff"] > 0):
            first["verdict"] = "no conclusion (the second seed points the other way)"

    # LC-QuAD: scores by question type, and the two subgroup contrasts
    vec, scored = vectors["lcquad"]
    by_family = collections.defaultdict(list)
    for i, r in enumerate(scored):
        by_family[family.get(r["structure"], "?")].append(i)
    summary["by_question_type"] = {fam: {"n": len(idx), "systems": {n: 100 * sum(v["f1"][i] for i in idx) / len(idx) for n, v in vec.items()}}
                                   for fam, idx in sorted(by_family.items())}
    seen = {i: g for g, ids in subgroups["entities"].items() for i in ids}
    covered = {i: g for g, ids in subgroups["examples"].items() for i in ids}
    contrasts = [
        ("SPLIT-42 - E2E-42: unseen - seen entities", "SPLIT-42", "E2E-42", seen.get, "unseen", "seen"),
        ("E2E-RAG - E2E-42: examples cover the entities - they do not", "E2E-RAG", "E2E-42", covered.get, "covered", "not_covered"),
        ("E2E-RAG - E2E-42, seen entities only: covered - not covered", "E2E-RAG", "E2E-42",
         lambda i: "covered" if covered.get(i) == "covered" else ("not_covered" if seen.get(i) == "seen" else None), "covered", "not_covered"),
    ]
    for label, left, right, group, g1, g2 in contrasts:
        if left in vec and right in vec:
            d = [x - y for x, y in zip(vec[left]["f1"], vec[right]["f1"], strict=True)]
            groups = [group(r["id"]) for r in scored]
            res = group_difference(d, groups, g1, g2)
            for g in (g1, g2):
                idx = [i for i, x in enumerate(groups) if x == g]
                res[g] = {n: 100 * sum(vec[n]["f1"][i] for i in idx) / len(idx) for n in (left, right)}
            summary["subgroups"][label] = res
    return summary, records, predictions


# ---------------------------------------------------------------- output
def num(x):
    return "" if x is None else f"{x:.1f}"


def interval(ci):
    return f"[{ci[0]:.1f}, {ci[1]:.1f}]"


def pvalue(p, floor):
    """With 10,000 resamples a p of 0 only means "below the resolution": 1/10,000 one-sided, twice that two-sided."""
    return f"< {floor:g}" if p == 0 else f"{p:.4f}"


def to_markdown(s):
    out = ["# Results", "", "F1 and differences are in points (0-100), with 95% bootstrap intervals over questions.",
           "This file is written by `scripts/evaluate.py`.", "",
           "Columns: *Exact* is the share of questions with F1 = 1. *Lenient ASK* counts a non-empty SELECT answer to a yes/no",
           "question as \"yes\" (the main F1 scores it 0). *Gold LIMIT* reruns a prediction that ends in ORDER BY with the LIMIT of",
           "the gold query. Sonnet-FS is the per-question mean of its two runs (no Exact column).", "",
           "## Planned comparisons", "",
           "| Comparison | Test set | n | Difference | 95% CI | p | Holm p | Outcome |", "|---|---|---|---|---|---|---|---|"]
    for c in s["comparisons"]:
        one_sided = c["test"] == "non-inferiority"
        test, floor = (f"one-sided, margin {c['margin']:g}", 0.0001) if one_sided else ("two-sided", 0.0002)
        out.append(f"| {c['label']}: {c['left']} - {c['right']} | {c['test_set']} | {c['n']} | {c['diff']:+.1f} | {interval(c['ci95'])} "
                   f"| {pvalue(c['p'], floor)} ({test}) | {pvalue(c['p_holm'], floor * len(s['comparisons']))} | {c['verdict']} |")
    out += ["", "p-values come from 10,000 bootstrap resamples; \"<\" means that no resample fell on the other side."]
    second = s["comparisons"][0].get("second_seed") if s["comparisons"] else None
    if second:
        out += [f"Second training seed (SPLIT-43 - E2E-43): {second['diff']:+.1f} {interval(second['ci95'])}."]
    for name, t in s["test_sets"].items():
        left = "; ".join(f"{n} x {why}" for why, n in t["left_out"].items()) or "none"
        out += ["", f"## {name}: {t['title']}", "", f"Scored {t['scored']} of {t['questions']} questions. Left out: {left}.", "",
                "| System | F1 | 95% CI | Exact | F1, lenient ASK | F1, gold LIMIT |", "|---|---|---|---|---|---|"]
        for n, r in t["systems"].items():
            out.append(f"| {n} | {r['f1']:.1f} | {interval(r['ci95'])} | {num(r.get('exact'))} | {num(r.get('f1_lenient_ask'))} "
                       f"| {num(r.get('f1_gold_limit'))} |")
        if t["pairs"]:
            out += ["", "| Paired difference | n | Difference | 95% CI |", "|---|---|---|---|"]
            out += [f"| {k} | {p['n']} | {p['diff']:+.1f} | {interval(p['ci95'])} |" for k, p in t["pairs"].items()]
    fams = list(s["by_question_type"])
    if fams:
        systems = list(next(iter(s["by_question_type"].values()))["systems"])
        out += ["", "## lcquad: F1 by question type", "", "| System | " + " | ".join(f"{f} ({s['by_question_type'][f]['n']})" for f in fams) + " |",
                "|---|" + "---|" * len(fams)]
        out += [f"| {n} | " + " | ".join(f"{s['by_question_type'][f]['systems'][n]:.1f}" for f in fams) + " |" for n in systems]
    if s["subgroups"]:
        out += ["", "## lcquad: subgroups", ""]
        for label, g in s["subgroups"].items():
            names = [k for k in g if k not in ("n1", "n2", "diff", "ci95")]
            parts = "; ".join(f"{k}: " + ", ".join(f"{n} {v:.1f}" for n, v in g[k].items()) for k in names)
            out.append(f"- {label}: {g['diff']:+.1f} {interval(g['ci95'])} ({g['n1']} vs {g['n2']} questions). Means: {parts}.")
    return "\n".join(out) + "\n"


def write_csv(path, records, predictions):
    names = sorted(predictions)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "id", "scored", "gold"] + names)
        for r in records:
            w.writerow([r["dataset"], r["id"], int(r["scored"]), r["gold"]]
                       + [("" if n not in r["scores"] else f"{r['scores'][n]['f1']:.6g}") for n in names])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "results"), help="where to write the three output files")
    ap.add_argument("--drop", nargs="*", default=[], metavar="DATASET:ID", help="leave these questions out as well")
    ap.add_argument("--missing-as-failed", action="store_true",
                    help="score a prediction whose query has no stored result as 0 instead of leaving the question out")
    a = ap.parse_args()
    summary, records, predictions = evaluate(ROOT, a.drop, a.missing_as_failed)
    if "GRISP" not in predictions:
        print("note: no GRISP predictions here, so the tables are written without the GRISP rows "
              "(scripts/baselines/grisp_outputs.py fetches them)")
    os.makedirs(a.out, exist_ok=True)
    write_csv(os.path.join(a.out, "per_question_f1.csv"), records, predictions)
    write_json(os.path.join(a.out, "summary.json"), summary)
    with open(os.path.join(a.out, "tables.md"), "w", encoding="utf-8") as f:
        f.write(to_markdown(summary))
    for c in summary["comparisons"]:
        print(f"{c['label']:45s} {c['diff']:+6.1f} {interval(c['ci95']):>16s}  {c['verdict']}")


if __name__ == "__main__":
    main()
