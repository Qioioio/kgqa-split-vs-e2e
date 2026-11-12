#!/usr/bin/env python3
"""QLoRA fine-tuning of Qwen2.5-Coder-7B-Instruct on question -> query (or skeleton) pairs.

All fine-tuned systems use this script with the settings in configs/train.json and differ only
in the training file:

    E2E      data/train/train_e2e.jsonl       SPLIT      data/train/train_split.jsonl
    E2E-B    data/train/train_e2e_B.jsonl     SPLIT-B    data/train/train_split_B.jsonl

Only the answer and the <|im_end|> that closes it contribute to the loss. Training runs for
three epochs and the adapter after the third epoch is used; the dev loss is logged per epoch
but nothing is selected on it.

    python scripts/train.py --train data/train/train_split.jsonl --dev data/train/dev_split.jsonl \
        --seed 42 --model /path/to/Qwen2.5-Coder-7B-Instruct --out runs/SPLIT-42

One run takes about three hours on an RTX 4090 (24 GB). --resume continues from the last checkpoint.
--debug runs on a CPU without quantisation, for trying the script with a small model:
    python scripts/train.py ... --debug --max-steps 3 --limit 40
"""
import argparse
import importlib.metadata
import os
import platform

from kgqa.files import read_json, sha256, write_json
from kgqa.prompts import answer_span, labels_for_span

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def versions():
    out = {"python": platform.python_version()}
    for package in ("torch", "transformers", "peft", "accelerate", "datasets", "bitsandbytes"):
        try:
            out[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            out[package] = None
    return out


def encode(tokenizer, chat, max_len):
    """One training example -> input_ids and labels (the answer and its closing <|im_end|>)."""
    full = tokenizer.apply_chat_template(chat, tokenize=False)
    prompt = tokenizer.apply_chat_template(chat[:-1], tokenize=False, add_generation_prompt=True)
    span = answer_span(prompt, full, chat[-1]["content"])
    enc = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc["input_ids"]
    labels = labels_for_span(ids, enc["offset_mapping"], span)
    return {"input_ids": ids[:max_len], "attention_mask": [1] * min(len(ids), max_len), "labels": labels[:max_len]}, len(ids)


class PadCollator:
    """Pad on the right: input_ids with the pad token, attention_mask with 0, labels with -100."""

    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, features):
        import torch
        width = max(len(f["input_ids"]) for f in features)
        out = {"input_ids": [], "attention_mask": [], "labels": []}
        for f in features:
            k = width - len(f["input_ids"])
            out["input_ids"].append(f["input_ids"] + [self.pad_id] * k)
            out["attention_mask"].append(f["attention_mask"] + [0] * k)
            out["labels"].append(f["labels"] + [-100] * k)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--train", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True, help="local directory of the base model")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "train.json"))
    ap.add_argument("--resume", action="store_true", help="continue from the last checkpoint in --out")
    ap.add_argument("--debug", action="store_true", help="CPU, no quantisation; allows --max-steps and --limit")
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--limit", type=int, default=0, help="use only the first N examples")
    a = ap.parse_args()
    if (a.max_steps != -1 or a.limit) and not a.debug:
        raise SystemExit("--max-steps and --limit are for --debug only")
    config = read_json(a.config)
    quant, lora, tr = config["quantization"], config["lora"], config["training"]

    import torch
    from datasets import Dataset, load_dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments, set_seed
    from transformers.trainer_utils import get_last_checkpoint

    os.makedirs(a.out, exist_ok=True)
    set_seed(a.seed)   # before the model is loaded, so that the LoRA initialisation is seeded too

    tokenizer = AutoTokenizer.from_pretrained(a.model)
    pad_id = tokenizer.convert_tokens_to_ids(tr["pad_token"])
    if tokenizer.pad_token_id != pad_id:
        tokenizer.pad_token = tr["pad_token"]
    if a.debug:
        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32)
    else:
        bnb = BitsAndBytesConfig(load_in_4bit=quant["load_in_4bit"], bnb_4bit_use_double_quant=quant["bnb_4bit_use_double_quant"],
                                 bnb_4bit_quant_type=quant["bnb_4bit_quant_type"],
                                 bnb_4bit_compute_dtype=getattr(torch, quant["bnb_4bit_compute_dtype"]))
        model = AutoModelForCausalLM.from_pretrained(a.model, quantization_config=bnb, device_map="auto")
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(r=lora["r"], lora_alpha=lora["lora_alpha"], target_modules=lora["target_modules"],
                                             lora_dropout=lora["lora_dropout"], bias=lora["bias"], task_type="CAUSAL_LM"))

    raw = load_dataset("json", data_files={"train": a.train, "dev": a.dev})
    data, stats = {}, {}
    for split, rows in raw.items():
        chats = rows["messages"][: a.limit] if a.limit else rows["messages"]
        encoded = [encode(tokenizer, chat, tr["max_seq_length"]) for chat in chats]
        data[split] = Dataset.from_list([features for features, _ in encoded])
        stats[split] = {"examples": len(encoded), "longest": max(n for _, n in encoded),
                        "truncated": sum(n > tr["max_seq_length"] for _, n in encoded)}
    print("data:", stats, flush=True)

    overrides = {}
    if a.debug:
        overrides = {"optim": "adamw_torch", "bf16": False, "max_steps": a.max_steps, "save_strategy": "no",
                     "eval_strategy": "steps", "eval_steps": max(a.max_steps, 1)}
    args = TrainingArguments(**{
        "output_dir": a.out, "seed": a.seed, "report_to": "none", "remove_unused_columns": False,
        "per_device_train_batch_size": tr["per_device_train_batch_size"], "per_device_eval_batch_size": tr["per_device_train_batch_size"],
        "gradient_accumulation_steps": tr["gradient_accumulation_steps"], "learning_rate": tr["learning_rate"],
        "lr_scheduler_type": tr["lr_scheduler_type"], "warmup_steps": tr["warmup_steps"], "num_train_epochs": tr["num_train_epochs"],
        "logging_steps": tr["logging_steps"], "eval_strategy": tr["eval_strategy"], "save_strategy": tr["save_strategy"],
        "save_total_limit": tr["save_total_limit"], "bf16": tr["bf16"], "optim": tr["optim"], **overrides})
    trainer = Trainer(model=model, args=args, train_dataset=data["train"], eval_dataset=data["dev"],
                      data_collator=PadCollator(pad_id), processing_class=tokenizer)
    checkpoint = get_last_checkpoint(a.out) if a.resume else None
    if a.resume and checkpoint is None:
        raise SystemExit(f"--resume: no checkpoint in {a.out}")
    trainer.train(resume_from_checkpoint=checkpoint)

    final = os.path.join(a.out, "final")
    trainer.model.save_pretrained(final)
    write_json(os.path.join(a.out, "run.json"), {
        "train": a.train, "dev": a.dev, "seed": a.seed, "debug": a.debug, "config": config, "versions": versions(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "data": stats,
        "steps": trainer.state.global_step,
        "dev_loss_by_epoch": [{"epoch": h.get("epoch"), "loss": h["eval_loss"]} for h in trainer.state.log_history if "eval_loss" in h],
        "log": trainer.state.log_history,
        "adapter_sha256": sha256(os.path.join(final, "adapter_model.safetensors"))})
    print(f"done: {trainer.state.global_step} steps, adapter in {final}", flush=True)


if __name__ == "__main__":
    main()
