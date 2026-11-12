#!/usr/bin/env python3
"""FYP-plain / FYP-RAG: fine-tuning with the recipe of the original BSc thesis, on the clean split.

The recipe (configs/thesis_recipe.json) is kept as it was, including what scripts/train.py
does differently:
- the loss covers the whole sequence, system message and question included;
- the pad token is set to <|im_end|>, and padding is masked out of the loss, so the model is
  never trained to produce the <|im_end|> that ends its answer. At inference it often does
  not stop and keeps writing until the token limit;
- five epochs, last checkpoint;
- the seed reaches the trainer only, after the LoRA weights have been initialised, so the
  initialisation is not seeded and the adapter is not exactly reproducible.
Both thesis rows use the one adapter trained here on data/train/train_e2e.jsonl.

    python scripts/baselines/thesis_train.py --train data/train/train_e2e.jsonl --dev data/train/dev_e2e.jsonl \
        --model /path/to/Qwen2.5-Coder-7B-Instruct --out runs/FYP
"""
import argparse
import os

from kgqa.files import read_json, sha256, write_json

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--train", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--debug", action="store_true", help="CPU, no quantisation; allows --max-steps and --limit")
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if (a.max_steps != -1 or a.limit) and not a.debug:
        raise SystemExit("--max-steps and --limit are for --debug only")
    recipe = read_json(os.path.join(ROOT, "configs", "thesis_recipe.json"))
    quant, lora, tr = recipe["quantization"], recipe["lora"], recipe["training"]

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DataCollatorForLanguageModeling, Trainer, TrainingArguments

    tokenizer = AutoTokenizer.from_pretrained(a.model)
    tokenizer.pad_token = tokenizer.eos_token
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

    data = load_dataset("json", data_files={"train": a.train, "dev": a.dev})
    if a.limit:
        data = {k: v.select(range(min(a.limit, len(v)))) for k, v in data.items()}

    def tokenize(batch):
        texts = [tokenizer.apply_chat_template(chat, tokenize=False) for chat in batch["messages"]]
        out = tokenizer(texts, add_special_tokens=True, truncation=True, padding=False, max_length=tr["max_seq_length"])
        return {"input_ids": out["input_ids"], "attention_mask": out["attention_mask"]}

    data = {k: v.map(tokenize, batched=True, remove_columns=v.column_names) for k, v in data.items()}
    overrides = {}
    if a.debug:
        overrides = {"optim": "adamw_torch", "bf16": False, "max_steps": a.max_steps, "eval_strategy": "steps",
                     "eval_steps": max(a.max_steps, 1), "save_strategy": "steps", "save_steps": max(a.max_steps, 1)}
    args = TrainingArguments(**{
        "output_dir": a.out, "seed": tr["seed"], "report_to": "none",
        "per_device_train_batch_size": tr["per_device_train_batch_size"], "gradient_accumulation_steps": tr["gradient_accumulation_steps"],
        "learning_rate": tr["learning_rate"], "logging_steps": tr["logging_steps"], "num_train_epochs": tr["num_train_epochs"],
        "eval_strategy": "epoch", "save_strategy": tr["save_strategy"], "bf16": tr["bf16"], "optim": tr["optim"], **overrides})
    trainer = Trainer(model=model, args=args, train_dataset=data["train"], eval_dataset=data["dev"],
                      data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False), processing_class=tokenizer)
    trainer.train()
    final = os.path.join(a.out, "final")
    trainer.model.save_pretrained(final)
    write_json(os.path.join(a.out, "run.json"), {"train": a.train, "dev": a.dev, "debug": a.debug, "recipe": recipe,
                                                  "steps": trainer.state.global_step, "log": trainer.state.log_history,
                                                  "adapter_sha256": sha256(os.path.join(final, "adapter_model.safetensors"))})
    print(f"done: {trainer.state.global_step} steps, adapter in {final}", flush=True)


if __name__ == "__main__":
    main()
