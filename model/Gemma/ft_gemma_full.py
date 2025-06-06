import os, json, warnings, argparse
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    Trainer, TrainingArguments, default_data_collator
)

def load_jsonl(path: str) -> pd.DataFrame:
    return pd.read_json(path, lines=True)

def preprocess(df: pd.DataFrame, tokenizer, max_length: int) -> Dataset:
    def build_prompt(msgs):
        user_only = [m for m in msgs if m["role"] == "user"]
        return tokenizer.apply_chat_template(
            user_only, tokenize=False, add_generation_prompt=False
        )

    df["prompt"] = df["messages"].apply(build_prompt)

    df["response"] = df["messages"].apply(
        lambda ms: ms[-1]["content"].strip() + tokenizer.eos_token
    )
    df["full_text"] = df["prompt"] + df["response"]

    enc        = tokenizer(df["full_text"].tolist(), truncation=True,
                           padding="max_length", max_length=max_length,
                           return_tensors="pt")
    prompt_enc = tokenizer(df["prompt"].tolist(), truncation=True,
                           padding="max_length", max_length=max_length,
                           return_tensors="pt")

    input_ids      = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    prompt_len  = prompt_enc["attention_mask"].sum(1)
    seq_idx     = torch.arange(input_ids.size(1)).unsqueeze(0)
    mask        = seq_idx < prompt_len.unsqueeze(1)       
    labels      = input_ids.clone()
    labels[mask] = -100

    return Dataset.from_dict({
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "labels":         labels
    })

# ------------------------------------------------- CLI
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model",  required=True, help="google/gemma-2-9b-it or ckpt path")
    p.add_argument("--local_rank", type=int, default=-1, help="deepspeed")  # ← 추가
    p.add_argument("--train_path",  required=True)
    p.add_argument("--val_path",    required=True)
    p.add_argument("--output_dir",  required=True)
    p.add_argument("--max_seq_length", type=int, default=512)
    p.add_argument("--per_device_train_batch_size", type=int, default=8)
    p.add_argument("--per_device_eval_batch_size",  type=int, default=8)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--num_train_epochs", type=int, default=5)
    p.add_argument("--learning_rate", type=float, default=3e-5)
    p.add_argument("--save_steps",    type=int, default=75)
    p.add_argument("--logging_steps", type=int, default=1)
    p.add_argument("--eval_steps",    type=int, default=75)
    p.add_argument("--bf16", action="store_true")
    return p.parse_args()

# ------------------------------------------------- main
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    warnings.filterwarnings("ignore")

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, trust_remote_code=True, use_fast=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32
        #device_map="auto",
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    ds_train = preprocess(load_jsonl(args.train_path), tokenizer, args.max_seq_length)
    ds_val   = preprocess(load_jsonl(args.val_path),   tokenizer, args.max_seq_length)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        deepspeed="ds_zero3.json",
        overwrite_output_dir=True,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        save_total_limit=1,
        load_best_model_at_end=True,
        bf16=True,
        report_to="tensorboard",
        logging_dir=os.path.join(args.output_dir, "tb_logs"),
        optim="adamw_torch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        data_collator=default_data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    # save logs & model
    log_path = os.path.join(args.output_dir, "")
    with open(log_path, "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)
    print(f"✔ log_history saved to {log_path}")

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"✅ Fine‑tuned model saved to {args.output_dir}")

if __name__ == "__main__":
    main()