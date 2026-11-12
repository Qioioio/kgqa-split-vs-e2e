#!/usr/bin/env python3
"""Generate one output per test question with the local model (base model, or base model + adapter).

    # E2E: the model writes the full query
    python scripts/generate.py --adapter runs/E2E-42/final --model MODEL --out results/predictions/E2E-42.jsonl
    # SPLIT: the model writes a skeleton; scripts/link.py turns it into a query afterwards
    python scripts/generate.py --adapter runs/SPLIT-42/final --skeleton --model MODEL --out results/predictions/SPLIT-42.jsonl
    # E2E-RAG: the E2E adapter with three retrieved training examples in the prompt
    python scripts/generate.py --adapter runs/E2E-42/final --prompt rag --retrieved results/predictions/FYP-RAG.jsonl \
        --examples data/train/train_e2e.jsonl --sets lcquad qald10 --model MODEL --out results/predictions/E2E-RAG.jsonl
    # Qwen-FS: the base model without an adapter, eight fixed examples written as skeletons
    python scripts/generate.py --prompt fewshot --examples data/train/train_split.jsonl --skeleton \
        --model MODEL --out results/predictions/Qwen-FS.jsonl

Decoding is greedy, one question at a time, with the settings in configs/train.json. The models
trained without the held-out structures (-B) are run with --sets lcquad heldout. A run that was
interrupted continues where it stopped. --debug runs on a CPU without quantisation.
"""
import argparse
import copy
import json
import os

from kgqa.files import read_json, read_jsonl
from kgqa.postprocess import extract_query
from kgqa.prompts import messages

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def select_questions(sets):
    """The test questions that belong to any of the named sets, in the order of data/test_questions.jsonl."""
    lists = read_json(os.path.join(ROOT, "data", "splits", "test_sets.json"))
    lcquad = set()
    if "lcquad" in sets:
        lcquad |= set(lists["lcquad"])
    if "heldout" in sets:
        lcquad |= set(lists["heldout_comp"]) | set(lists["heldout_zero"])
    return [q for q in read_jsonl(os.path.join(ROOT, "data", "test_questions.jsonl"))
            if (q["dataset"] == "qald10" and "qald10" in sets) or (q["dataset"] == "lcquad2" and q["id"] in lcquad)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, help="local directory of the base model")
    ap.add_argument("--adapter", help="directory of a trained LoRA adapter")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", choices=["plain", "rag", "fewshot"], default="plain")
    ap.add_argument("--skeleton", action="store_true", help="the outputs are skeletons: do not extract a query from them yet")
    ap.add_argument("--sets", nargs="+", choices=["lcquad", "heldout", "qald10"], default=["lcquad", "heldout", "qald10"])
    ap.add_argument("--examples", help="training file (chat format) the prompt examples are taken from")
    ap.add_argument("--retrieved", help="rag: a prediction file whose rows name the retrieved training questions (\"examples\")")
    ap.add_argument("--fewshot-ids", default=os.path.join(ROOT, "configs", "fewshot_examples.json"))
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "train.json"))
    ap.add_argument("--debug", action="store_true", help="CPU, no quantisation; allows --limit")
    ap.add_argument("--limit", type=int, default=0, help="only the first N questions")
    a = ap.parse_args()
    if a.prompt != "plain" and not a.examples:
        raise SystemExit(f"--prompt {a.prompt} needs --examples")
    if a.prompt == "rag" and not a.retrieved:
        raise SystemExit("--prompt rag needs --retrieved")
    if a.limit and not a.debug:
        raise SystemExit("--limit is for --debug only")

    config, prompts = read_json(a.config), read_json(os.path.join(ROOT, "configs", "prompts.json"))
    gen, quant = config["generation"], config["quantization"]
    questions = select_questions(a.sets)[: a.limit or None]
    pool, retrieved, fixed = {}, {}, None
    if a.examples:
        for r in read_jsonl(a.examples):
            chat = {m["role"]: m["content"] for m in r["messages"]}
            pool[r["id"]] = (chat["user"], chat["assistant"])
    if a.prompt == "rag":
        retrieved = {(r["dataset"], r["id"]): r["examples"] for r in read_jsonl(a.retrieved)}
    if a.prompt == "fewshot":
        fixed = read_json(a.fewshot_ids)["ids"]
    done = {(r["dataset"], r["id"]) for r in read_jsonl(a.out)} if os.path.exists(a.out) else set()
    todo = [q for q in questions if (q["dataset"], q["id"]) not in done]
    print(f"{len(questions)} questions, {len(done)} already done", flush=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(a.model)
    eos_ids = [tokenizer.convert_tokens_to_ids(t) for t in gen["eos_tokens"]]
    if a.debug:
        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32)
    else:
        bnb = BitsAndBytesConfig(load_in_4bit=quant["load_in_4bit"], bnb_4bit_use_double_quant=quant["bnb_4bit_use_double_quant"],
                                 bnb_4bit_quant_type=quant["bnb_4bit_quant_type"],
                                 bnb_4bit_compute_dtype=getattr(torch, quant["bnb_4bit_compute_dtype"]))
        model = AutoModelForCausalLM.from_pretrained(a.model, quantization_config=bnb, device_map="auto")
    if a.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # start from the model's own generation config and change only what configs/train.json lists
    # (the model ships with repetition_penalty 1.1 and sampling parameters, which greedy decoding ignores)
    settings = copy.deepcopy(model.generation_config)
    settings.do_sample, settings.num_beams = gen["do_sample"], gen["num_beams"]
    settings.repetition_penalty, settings.max_new_tokens = gen["repetition_penalty"], gen["max_new_tokens"]
    settings.eos_token_id, settings.pad_token_id = eos_ids, tokenizer.convert_tokens_to_ids(gen["pad_token"])

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as out:
        for n, q in enumerate(todo, 1):
            example_ids = retrieved[(q["dataset"], q["id"])] if a.prompt == "rag" else fixed
            examples = [pool[i] for i in example_ids] if example_ids else None
            prompt = tokenizer.apply_chat_template(messages(a.prompt, q["question"], prompts, examples), tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
            with torch.no_grad():
                ids = model.generate(**inputs, generation_config=settings)
            new = ids[0][len(inputs.input_ids[0]):].tolist()
            row = {"dataset": q["dataset"], "id": q["id"], "question": q["question"],
                   "output": tokenizer.decode(new, skip_special_tokens=True).strip(),
                   "hit_max_new_tokens": len(new) >= gen["max_new_tokens"] and not any(t in eos_ids for t in new)}
            if a.prompt == "rag":
                row["examples"] = example_ids
            if not a.skeleton:
                row["query"] = extract_query(row["output"])[0]
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            if n % 50 == 0 or n == len(todo):
                print(f"{n}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
