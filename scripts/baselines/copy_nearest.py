#!/usr/bin/env python3
"""COPY: answer with the gold query of the most similar training question, unchanged.

Similarity is the cosine of bge-small-en-v1.5 sentence embeddings (the retriever of the thesis
system FYP-RAG). COPY searches the whole training set; COPY-B, for the held-out-structure
questions, searches the training set without those structures.

    python scripts/baselines/copy_nearest.py            # writes COPY.jsonl and COPY-B.jsonl
"""
import argparse
import os

from kgqa.files import read_json, read_jsonl, write_jsonl
from kgqa.postprocess import extract_query

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOP_K = 3   # only the nearest is used; FAISS breaks exact ties differently for k = 1, and the run used k = 3


def nearest(pool, questions, embedder):
    """For every question the TOP_K most similar pool entries: [[(pool index, score)]]."""
    import faiss
    index = faiss.IndexFlatIP(embedder.get_sentence_embedding_dimension())
    index.add(embedder.encode([q for q, _ in pool], convert_to_numpy=True, normalize_embeddings=True, batch_size=256))
    scores, found = index.search(embedder.encode(questions, convert_to_numpy=True, normalize_embeddings=True, batch_size=256), TOP_K)
    return [list(zip(row, score, strict=True)) for row, score in zip(found, scores, strict=True)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--embedder", default="BAAI/bge-small-en-v1.5")
    a = ap.parse_args()
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer(a.embedder)
    questions = read_jsonl(os.path.join(ROOT, "data", "test_questions.jsonl"))
    lists = read_json(os.path.join(ROOT, "data", "splits", "test_sets.json"))
    runs = {"COPY": ("train_e2e.jsonl", set(lists["lcquad"])),
            "COPY-B": ("train_e2e_B.jsonl", set(lists["heldout_comp"]) | set(lists["heldout_zero"]))}
    for name, (train_file, ids) in runs.items():
        rows = read_jsonl(os.path.join(ROOT, "data", "train", train_file))
        pool = [(r["messages"][1]["content"], r["messages"][2]["content"]) for r in rows]
        todo = [q for q in questions if q["dataset"] == "lcquad2" and q["id"] in ids]
        out = []
        for q, hits in zip(todo, nearest(pool, [q["question"] for q in todo], embedder), strict=True):
            best = hits[0][0]
            out.append({"dataset": q["dataset"], "id": q["id"], "question": q["question"], "output": pool[best][1],
                        "examples": [rows[best]["id"]], "query": extract_query(pool[best][1])[0]})
        write_jsonl(os.path.join(ROOT, "results", "predictions", f"{name}.jsonl"), out)
        print(f"{name}: {len(out)} predictions")


if __name__ == "__main__":
    main()
