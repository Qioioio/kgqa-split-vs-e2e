#!/usr/bin/env python3
"""The Claude Sonnet rows, through the Message Batches API.

    Sonnet-FS    few-shot: the eight fixed examples, the model writes a skeleton (link it afterwards)
    Sonnet-mem   zero-shot: the model writes a query with Wikidata IDs from memory
    *-direct     the same with thinking reduced to the minimum the model allows

Three steps, each of which keeps its raw files in the run directory:

    python scripts/baselines/claude_batch.py prepare --system Sonnet-FS --run runs/Sonnet-FS-run1
    python scripts/baselines/claude_batch.py submit  --run runs/Sonnet-FS-run1
    python scripts/baselines/claude_batch.py fetch   --run runs/Sonnet-FS-run1 --wait \
        --out results/predictions/Sonnet-FS-run1.jsonl

Requests that errored or expired are sent again by another submit/fetch round, at most twice;
what still has no answer then counts as an empty output. The model does not accept a
temperature, so its outputs are not deterministic: Sonnet-FS was run twice and the two runs
are averaged. The API key is read from the environment variable ANTHROPIC_API_KEY.
"""
import argparse
import datetime
import glob
import json
import os
import time

from kgqa.files import read_json, read_jsonl, write_json, write_jsonl
from kgqa.postprocess import extract_query
from kgqa.prompts import messages

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
API = "https://api.anthropic.com/v1/messages/batches"
MAX_ROUNDS = 3


def build_requests(system, questions, settings, prompts, examples):
    """One Message Batches request per question; the system message goes into the `system` field."""
    spec = settings["systems"][system]
    requests = []
    for q in questions:
        chat = messages(spec["prompt"], q["question"], prompts, examples if spec["prompt"] == "fewshot" else None)
        params = {"model": settings["model"], "max_tokens": settings["max_tokens"], "system": chat[0]["content"], "messages": chat[1:]}
        if settings.get("output_config"):
            params["output_config"] = settings["output_config"]
        if spec["thinking"]:   # None: the field is not sent and the model thinks as much as it decides to
            params["thinking"] = spec["thinking"]
        requests.append({"custom_id": f"{q['dataset']}__{q['id']}", "params": params})
    return requests


def call(method, url, version, body=None):
    import requests
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise SystemExit("set the environment variable ANTHROPIC_API_KEY")
    r = requests.request(method, url, data=json.dumps(body) if body is not None else None, timeout=300,
                         headers={"x-api-key": key, "anthropic-version": version, "content-type": "application/json"})
    if r.status_code >= 400:
        raise SystemExit(f"API {r.status_code}: {r.text[:300]}")
    return r


def rounds(run):
    return sorted(int(os.path.basename(p)[len("batch_"):-len(".json")]) for p in glob.glob(os.path.join(run, "batch_*.json")))


def succeeded(run):
    """custom_id -> raw result, for the requests that succeeded in any round fetched so far."""
    out = {}
    for k in rounds(run):
        path = os.path.join(run, f"results_{k}.jsonl")
        if os.path.exists(path):
            out.update((res["custom_id"], res) for res in read_jsonl(path) if res["result"]["type"] == "succeeded")
    return out


def predictions(run, meta):
    """The prediction rows of a run: the text of each answer, in the order of the questions."""
    answered = succeeded(run)
    rows = []
    for q in meta["questions"]:
        res = answered.get(f"{q['dataset']}__{q['id']}")
        message = res["result"]["message"] if res else {}
        usage = message.get("usage") or {}
        row = {"dataset": q["dataset"], "id": q["id"], "question": q["question"],
               "output": "".join(c.get("text", "") for c in message.get("content", []) if c.get("type") == "text").strip(),
               "usage": {"input_tokens": usage.get("input_tokens", 0), "output_tokens": usage.get("output_tokens", 0),
                         "thinking_tokens": (usage.get("output_tokens_details") or {}).get("thinking_tokens", 0)}}
        if meta["prompt"] == "memory":
            row["query"] = extract_query(row["output"])[0]
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--system", required=True)
    p.add_argument("--run", required=True, help="new directory for this run")
    p.add_argument("--settings", default=os.path.join(ROOT, "configs", "claude.json"))
    p.add_argument("--examples", default=os.path.join(ROOT, "data", "train", "train_split.jsonl"))
    p.add_argument("--limit", type=int, default=0)
    s = sub.add_parser("submit")
    s.add_argument("--run", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--run", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--wait", action="store_true", help="poll every minute until the batch has ended")
    a = ap.parse_args()

    if a.command == "prepare":
        settings, prompts = read_json(a.settings), read_json(os.path.join(ROOT, "configs", "prompts.json"))
        questions = read_jsonl(os.path.join(ROOT, "data", "test_questions.jsonl"))[: a.limit or None]
        examples = None
        if settings["systems"][a.system]["prompt"] == "fewshot":
            pool = {r["id"]: r["messages"] for r in read_jsonl(a.examples)}
            ids = read_json(os.path.join(ROOT, "configs", "fewshot_examples.json"))["ids"]
            examples = [(pool[i][1]["content"], pool[i][2]["content"]) for i in ids]
        os.makedirs(a.run)
        write_jsonl(os.path.join(a.run, "requests.jsonl"), build_requests(a.system, questions, settings, prompts, examples))
        write_json(os.path.join(a.run, "meta.json"), {
            "system": a.system, "prompt": settings["systems"][a.system]["prompt"], "settings": settings,
            "prepared": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "questions": [{k: q[k] for k in ("dataset", "id", "question")} for q in questions]})
        print(f"{len(questions)} requests in {a.run}")
        return

    meta = read_json(os.path.join(a.run, "meta.json"))
    version = meta["settings"]["anthropic_version"]
    done = rounds(a.run)
    if a.command == "submit":
        if done and not os.path.exists(os.path.join(a.run, f"results_{done[-1]}.jsonl")):
            raise SystemExit(f"round {done[-1]} has not been fetched yet")
        if len(done) >= MAX_ROUNDS:
            raise SystemExit(f"{MAX_ROUNDS} rounds have been submitted already")
        done_ids = succeeded(a.run)
        todo = [r for r in read_jsonl(os.path.join(a.run, "requests.jsonl")) if r["custom_id"] not in done_ids]
        if not todo:
            raise SystemExit("every request has succeeded")
        batch = call("POST", API, version, {"requests": todo}).json()
        write_json(os.path.join(a.run, f"batch_{len(done) + 1}.json"), {"round": len(done) + 1, "requests": len(todo), "response": batch})
        print(f"round {len(done) + 1}: batch {batch['id']} with {len(todo)} requests")
        return

    k = done[-1]
    batch = read_json(os.path.join(a.run, f"batch_{k}.json"))["response"]
    while True:
        status = call("GET", f"{API}/{batch['id']}", version).json()
        if status["processing_status"] == "ended" or not a.wait:
            break
        time.sleep(60)
    if status["processing_status"] != "ended":
        raise SystemExit(f"round {k} has not ended: {status['processing_status']} {status.get('request_counts')}")
    with open(os.path.join(a.run, f"results_{k}.jsonl"), "w", encoding="utf-8") as out:
        out.write(call("GET", status["results_url"], version).content.decode("utf-8"))
    rows = predictions(a.run, meta)
    write_jsonl(a.out, rows)
    print(f"round {k}: {status['request_counts']}; {len(succeeded(a.run))} of {len(rows)} answered -> {a.out}")


if __name__ == "__main__":
    main()
