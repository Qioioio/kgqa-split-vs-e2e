#!/usr/bin/env python3
"""FYP-plain / FYP-RAG: inference as in the original BSc thesis (configs/thesis_recipe.json).

    FYP-plain  the fine-tuned model, question only
    FYP-RAG    the same adapter; the three most similar training questions (bge-small-en-v1.5)
               are put into the prompt with their queries, and the model is told to take the
               IDs from them. Code fences are removed from the output.

Kept as in the thesis: the base model is loaded in 4 bits with bitsandbytes' default
quantisation type (fp4, while training used nf4), decoding uses the model's own generation
settings (repetition penalty 1.1) and stops after 128 new tokens.

    python scripts/baselines/thesis_generate.py --adapter runs/FYP/final --model MODEL --out results/predictions/FYP-plain.jsonl
    python scripts/baselines/thesis_generate.py --adapter runs/FYP/final --model MODEL --rag --out results/predictions/FYP-RAG.jsonl
"""
import argparse
import json
import os

from kgqa.files import read_json, read_jsonl
from kgqa.postprocess import extract_query

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Retriever:
    def __init__(self, train_file, embedder):
        import faiss
        from sentence_transformers import SentenceTransformer
        self.rows = [(r["id"], r["messages"][1]["content"], r["messages"][2]["content"]) for r in read_jsonl(train_file)]
        self.embedder = SentenceTransformer(embedder)
        vectors = self.embedder.encode([q for _, q, _ in self.rows], convert_to_numpy=True, normalize_embeddings=True)
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

    def top(self, question, k):
        _, found = self.index.search(self.embedder.encode([question], convert_to_numpy=True, normalize_embeddings=True), k)
        return [self.rows[i] for i in found[0]]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rag", action="store_true")
    ap.add_argument("--train", default=os.path.join(ROOT, "data", "train", "train_e2e.jsonl"), help="retrieval pool for --rag")
    ap.add_argument("--embedder", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--debug", action="store_true", help="CPU, no quantisation; allows --limit")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if a.limit and not a.debug:
        raise SystemExit("--limit is for --debug only")
    recipe = read_json(os.path.join(ROOT, "configs", "thesis_recipe.json"))["rag" if a.rag else "plain"]
    questions = read_jsonl(os.path.join(ROOT, "data", "test_questions.jsonl"))[: a.limit or None]
    done = {(r["dataset"], r["id"]) for r in read_jsonl(a.out)} if os.path.exists(a.out) else set()
    todo = [q for q in questions if (q["dataset"], q["id"]) not in done]

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(a.model)
    if a.debug:
        base = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32)
    else:
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        base = AutoModelForCausalLM.from_pretrained(a.model, quantization_config=bnb, device_map="auto")
    model = PeftModel.from_pretrained(base, a.adapter)
    model.eval()
    retriever = Retriever(a.train, a.embedder) if a.rag else None
    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as out:
        for n, q in enumerate(todo, 1):
            options = {"max_new_tokens": recipe["max_new_tokens"], "do_sample": recipe["do_sample"]}
            if a.rag:
                hits = retriever.top(q["question"], recipe["top_k"])
                references = "".join(recipe["reference_item"].format(i=i + 1, q=question, sparql=sparql)
                                     for i, (_, question, sparql) in enumerate(hits))
                user = recipe["user"].format(reference_text=references, question=q["question"])
            else:
                user = recipe["user"].format(question=q["question"])
                options["pad_token_id"] = tokenizer.eos_token_id
            chat = [{"role": "system", "content": recipe["system"]}, {"role": "user", "content": user}]
            inputs = tokenizer(tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True), return_tensors="pt").to(device)
            with torch.no_grad():
                ids = model.generate(**inputs, **options)
            new = ids[0][len(inputs.input_ids[0]):]
            output = tokenizer.decode(new, skip_special_tokens=True).strip()
            if a.rag:
                output = output.replace("```sparql", "").replace("```", "").strip()
            row = {"dataset": q["dataset"], "id": q["id"], "question": q["question"], "output": output,
                   "hit_max_new_tokens": int(len(new)) >= recipe["max_new_tokens"]}   # as recorded for the thesis rows
            if a.rag:
                row["examples"] = [i for i, _, _ in hits]
            row["query"] = extract_query(output)[0]
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            if n % 50 == 0 or n == len(todo):
                print(f"{n}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
