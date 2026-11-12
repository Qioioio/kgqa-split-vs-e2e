#!/usr/bin/env python3
"""Download the datasets, clean them and write the training files.

    data/raw/      LC-QuAD 2.0 and QALD-10 in the versions prepared for QLever by the University of
                   Freiburg, and the GRISP outputs on LC-QuAD 2.0 (used as a first filter for test
                   questions whose gold query returns nothing)
    data/clean/    train.jsonl, dev.jsonl, test.jsonl (the pool the test sets are drawn from),
                   qald10_test.jsonl, removed.jsonl (what was dropped and why)
    data/train/    chat-format training files: {train,dev}_e2e.jsonl (target: the full query),
                   {train,dev}_split.jsonl (target: the skeleton), and the same without the five
                   held-out structures (*_B.jsonl)

The downloads are checked against the SHA256 of the files we used. Everything else is
deterministic, so the output files are identical on every run.

    python scripts/prepare_data.py
"""
import argparse
import collections
import os
import urllib.request
from multiprocessing import Pool

from kgqa.clean import HELDOUT_STRUCTURES, choose_question, clean_lcquad, syntax_error
from kgqa.files import read_json, read_jsonl, sha256, write_jsonl
from kgqa.skeleton import fill, strip_prefixes, to_skeleton

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = "https://ad-publications.cs.uni-freiburg.de/grisp/benchmark/wikidata/"
DOWNLOADS = {   # local path under data/raw -> (URL, SHA256)
    "lcquad2/train.jsonl": (BASE + "lcquad2-qlever/train.jsonl", "fedd90953d08c454230b91b8f38cb59cd7d0b125fce050e8036fc378f2a6fa58"),
    "lcquad2/test.jsonl": (BASE + "lcquad2-qlever/test.jsonl", "da690c7f8e3db070a4ae99d35db7ddf65a23e3e5ca1b07037d0bcb433b34221c"),
    "qald10/test.jsonl": (BASE + "qald10/test.jsonl", "3a44fa5ddfeb9ee7d288c88bb22e4d53d0f4356c91b8eb008f7d22a31be8f149"),
    "grisp/lcquad2.json": (BASE + "lcquad2-qlever/outputs/grisp.qwen-2.5-7b-instruct-wikidata-09-26-lcquad2-qlever-twetp-08-10-26.evaluation.json",
                           "6d8640192cbca786813ea0bbe300ff8210944b335583f19be3518571a084874d"),
}


def download(raw):
    for name, (url, digest) in DOWNLOADS.items():
        path = os.path.join(raw, name)
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            print("downloading", url)
            urllib.request.urlretrieve(url, path)
        if sha256(path) != digest:
            raise SystemExit(f"{path}: SHA256 differs from the file we used ({digest}); the published file may have changed")


def training_files(clean, labels, system_prompt, heldout):
    """{file name: rows}. The e2e and split files hold the same questions and differ only in the
    target; questions whose query has no skeleton are left out of both."""
    files = collections.OrderedDict()
    for split in ("train", "dev"):
        rows = {"e2e": [], "split": [], "e2e_B": [], "split_B": []}
        for r in clean[split]:
            skeleton, slots = to_skeleton(r["sparql"], labels)
            if skeleton is None:
                continue
            if fill(skeleton, [wid for _, wid, _ in slots]) != strip_prefixes(r["sparql"]):
                raise SystemExit(f"{r['id']}: the skeleton does not convert back to the query")
            chat = [{"role": "system", "content": system_prompt}, {"role": "user", "content": r["question"]}]
            for kind, target in (("e2e", strip_prefixes(r["sparql"])), ("split", skeleton)):
                row = {"id": r["id"], "messages": chat + [{"role": "assistant", "content": target}]}
                rows[kind].append(row)
                if r["structure"] not in heldout:
                    rows[kind + "_B"].append(row)
        for kind, kind_rows in rows.items():
            files[f"{split}_{kind}.jsonl"] = kind_rows
    return files


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default=os.path.join(ROOT, "data"))
    a = ap.parse_args()
    raw = os.path.join(a.data, "raw")
    download(raw)

    raw_train = read_jsonl(os.path.join(raw, "lcquad2", "train.jsonl"))
    raw_test = read_jsonl(os.path.join(raw, "lcquad2", "test.jsonl"))
    with Pool() as pool:   # parsing 30,000 queries with rdflib takes a while
        errors = pool.map(syntax_error, [x["sparql"] for x in raw_train + raw_test], chunksize=200)
    train, dev, test, removed = clean_lcquad(raw_train, raw_test, read_json(os.path.join(raw, "grisp", "lcquad2.json")),
                                             errors[:len(raw_train)], errors[len(raw_train):])
    qald = []
    for x in read_jsonl(os.path.join(raw, "qald10", "test.jsonl")):
        question, source = choose_question(x)
        qald.append({"id": x["id"], "question": question, "question_source": source, "sparql": x["sparql"]})
    clean = {"train": train, "dev": dev, "test": test, "qald10_test": qald, "removed": removed}
    for name, rows in clean.items():
        write_jsonl(os.path.join(a.data, "clean", f"{name}.jsonl"), rows)

    labels = {r["id"]: r for name in ("entity_labels", "property_labels")
              for r in read_jsonl(os.path.join(a.data, "labels", f"{name}.jsonl.gz"))}
    system_prompt = read_json(os.path.join(ROOT, "configs", "prompts.json"))["system"]
    heldout = {s for group in HELDOUT_STRUCTURES.values() for s in group}
    files = training_files(clean, labels, system_prompt, heldout)
    for name, rows in files.items():
        write_jsonl(os.path.join(a.data, "train", name), rows)

    reasons = collections.Counter((r["split"], r["reason"]) for r in removed)
    print(f"train {len(train)}, dev {len(dev)}, test pool {sum(not r['over_1024'] for r in test)} (+{sum(r['over_1024'] for r in test)} "
          f"with more than 1,024 gold rows), QALD-10 {len(qald)}")
    print("removed:", ", ".join(f"{split} {reason} {n}" for (split, reason), n in sorted(reasons.items())))
    print("training files:", ", ".join(f"{name} {len(rows)}" for name, rows in files.items()))


if __name__ == "__main__":
    main()
