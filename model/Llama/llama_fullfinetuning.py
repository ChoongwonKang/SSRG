import os
import ast
import warnings
import argparse
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
import json

BOT_START    = "<|begin_of_text|>"
HEADER_OPEN  = "<|start_header_id|>"
HEADER_CLOSE = "<|end_header_id|>"
EOT          = "<|eot_id|>"

def apply_chat_template_llama31(messages):
    """
    messages: list of dicts with keys 'role' and 'content'
    return: formatted prompt string for LLaMA-3.1 Chat
    """
    prompt = BOT_START + "\n"
    for msg in messages:
        prompt += f"{HEADER_OPEN}{msg['role']}{HEADER_CLOSE}\n"
        prompt += msg['content'].strip() + EOT + "\n"
    prompt += f"{HEADER_OPEN}assistant{HEADER_CLOSE}\n"
    return prompt


def load_jsonl(path: str) -> pd.DataFrame:
    return pd.read_json(path, lines=True)


def preprocess(df: pd.DataFrame, tokenizer, max_length: int):
    df["messages"] = df["messages"].apply(
        lambda x: x if isinstance(x, list) else ast.literal_eval(x)
    )
    df["prompt_text"] = df["messages"].apply(apply_chat_template_llama31)
    df["suffix"] = df.get("assistant").str.strip()
    df["full_text"] = df["prompt_text"] + df["suffix"]

    enc = tokenizer(
        df["full_text"].tolist(),
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    )
    prompt_enc = tokenizer(
        df["prompt_text"].tolist(),
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    )

    input_ids = enc.input_ids
    attention_mask = enc.attention_mask
    labels = input_ids.clone()
    mask = prompt_enc.input_ids != tokenizer.pad_token_id
    labels[mask] = -100

    return Dataset.from_dict({
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    })


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune LLaMA 3.1 Chat/Instruct model on sarcasm tasks"
    )
    parser.add_argument("--base_model",    type=str, required=True,
                        help="")
    parser.add_argument("--local_rank", type=int, default=-1, help="deepspeed")
    parser.add_argument("--train_path",    type=str, required=True)
    parser.add_argument("--val_path",      type=str, required=True)
    parser.add_argument("--output_dir",    type=str, required=True)
    parser.add_argument("--max_seq_length",type=int, default=512)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size",  type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=5)
    parser.add_argument("--learning_rate",   type=float, default=1e-5)
    parser.add_argument("--save_steps",      type=int, default=15)
    parser.add_argument("--logging_steps",   type=int, default=1)
    parser.add_argument("--eval_steps",      type=int, default=15)
    parser.add_argument("--bf16",            action="store_true",
                        help="Enable bf16 training if supported")
    parser.add_argument("--hf_token",       type=str, default=None,
                        help="HF access token for private repos")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    warnings.filterwarnings("ignore")

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        use_fast=False,
        use_auth_token=args.hf_token
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        use_auth_token=args.hf_token

    )

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    df_train = load_jsonl(args.train_path)
    df_val   = load_jsonl(args.val_path)
    ds_train = preprocess(df_train, tokenizer, args.max_seq_length)
    ds_val   = preprocess(df_val,   tokenizer, args.max_seq_length)

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

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
        optim="adamw_torch"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    trainer.train()

    # 로그 저장
    log_path = os.path.join(args.output_dir, "")
    with open(log_path, "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)
    print(f"✔ Training log saved to {log_path}")

    # 모델 및 토크나이저 저장
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"✅ Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()
