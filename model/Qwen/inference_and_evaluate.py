# inference_and_evaluate.py

import argparse
import torch
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import evaluate
from tqdm import tqdm
import random
import numpy as np
import os
os.environ["HF_HUB_OFFLINE"] = "1"

def parse_args():
    parser = argparse.ArgumentParser(description="Inference + Evaluation for Qwen2.5 sarcasm explanations")
    parser.add_argument("--model_dir",       type=str, required=True,  help="Fine-tuned model directory")
    parser.add_argument("--test_path",       type=str, required=True,  help="Path to test.jsonl")
    parser.add_argument("--max_new_tokens",  type=int, default=1024,    help="")
    parser.add_argument("--temperature",     type=float, default=0.0,  help="(0 → greedy, >0 → sampling)")
    parser.add_argument("--predictions_out", type=str, default="",     help="")
    parser.add_argument("--metrics_out",     type=str, default="", help="")
    return parser.parse_args()

def main():
    random.seed(42)
    np.random.seed(42)
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir, trust_remote_code=True, local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        local_files_only=True
    )
    model.eval()

    ds = load_dataset("json", data_files=args.test_path)["train"]

    preds, refs = [], []
    for ex in tqdm(ds, desc="Generating"):
        prompt = tokenizer.apply_chat_template(
            ex["messages"], tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        do_sample = args.temperature > 0.0
        gen_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample":      do_sample
        }
        if do_sample:
            gen_kwargs["temperature"] = args.temperature

        with torch.no_grad():
            gen_ids = model.generate(**inputs, **gen_kwargs)

        # full  = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        # reply = full[len(prompt):].strip()

        input_len = inputs["input_ids"].shape[-1]
        gen_tokens = gen_ids[0][input_len:]
        reply = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        preds.append(reply)
        refs.append(ex["assistant"].strip())

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

    if "rougeL_fmeasure" in rouge_res:
        rouge_l_f1 = rouge_res["rougeL_fmeasure"]
    else:
        rouge_l_f1 = rouge_res.get("rougeL", 0.0)

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
