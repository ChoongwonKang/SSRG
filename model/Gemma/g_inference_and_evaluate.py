# inference_and_evaluate_gemma.py

import argparse
import torch
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import evaluate
from tqdm import tqdm
import os
import random
import numpy as np

os.environ["HF_HUB_OFFLINE"] = "1"

def parse_args():
    parser = argparse.ArgumentParser(description="Inference + Evaluation for Gemma fine-tuned model")
    parser.add_argument("--model_dir",       type=str, required=True,  help="Fine-tuned model directory")
    parser.add_argument("--test_path",       type=str, required=True,  help="Path to test.jsonl")
    parser.add_argument("--max_new_tokens",  type=int, default=1024)
    parser.add_argument("--temperature",     type=float, default=0.0)
    parser.add_argument("--predictions_out", type=str, default="")
    parser.add_argument("--metrics_out",     type=str, default="")
    return parser.parse_args()

def main():
    random.seed(42)
    np.random.seed(42)
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir, trust_remote_code=True, local_files_only=False, use_fast=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        local_files_only=False
    )
    model.eval()
    tokenizer.pad_token = tokenizer.eos_token  
    tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
    print("EOS token ID:", tokenizer.eos_token_id)

    ds = load_dataset("json", data_files=args.test_path)["train"]

    preds, refs = [], []
    for ex in tqdm(ds, desc="Generating"):
        user_only = [m for m in ex["messages"] if m["role"] == "user"]
        prompt = tokenizer.apply_chat_template(
            user_only, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        gen_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.temperature > 0.0
          
        }
        if gen_kwargs["do_sample"]:
            gen_kwargs["temperature"] = args.temperature

        with torch.no_grad():
            gen_ids = model.generate(**inputs, **gen_kwargs)

        input_len = inputs["input_ids"].shape[-1]
        gen_tokens = gen_ids[0][input_len:]
        reply = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        preds.append(reply)

              
        ref = next((m["content"] for m in ex["messages"] if m["role"] == "model"), "")
        refs.append(ref.strip())

        print(f"\n[Prediction] {reply}")
        print(f"[Reference]  {ref.strip()}\n")

    df = pd.DataFrame({"prediction": preds, "reference": refs})
    df.to_csv(args.predictions_out, index=False)
    print(f"✔ Predictions saved to {args.predictions_out}")


    rouge     = evaluate.load("rouge")
    meteor    = evaluate.load("meteor")
    bleu      = evaluate.load("bleu")
    bertscore = evaluate.load("bertscore")

    rouge_res     = rouge.compute(predictions=preds, references=refs, rouge_types=["rougeL"])
    meteor_res    = meteor.compute(predictions=preds, references=refs)
    bleu_res      = bleu.compute(predictions=preds, references=[[r] for r in refs])
    bertscore_res = bertscore.compute(predictions=preds, references=refs, lang="en")

    rouge_l_f1 = rouge_res.get("rougeL_fmeasure", rouge_res.get("rougeL", 0.0))
    avg_bertscore = sum(bertscore_res["f1"]) / len(bertscore_res["f1"])

    metrics_dict = {
        "rougeL_f1":    rouge_l_f1,
        "meteor":       meteor_res.get("meteor", 0.0),
        "bleu4":        bleu_res.get("bleu", 0.0),
        "bertscore_f1": avg_bertscore
    }

    pd.DataFrame([metrics_dict]).to_csv(args.metrics_out, index=False)
    print(f"✔ Metrics saved to {args.metrics_out}")

if __name__ == "__main__":
    main()
