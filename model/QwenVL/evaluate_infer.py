import argparse
import torch
import pandas as pd
from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import evaluate
from tqdm import tqdm
import random
import numpy as np
import os
import json

os.environ["HF_HUB_OFFLINE"] = "1"

def parse_args():
    parser = argparse.ArgumentParser(description="Inference + Evaluation for Qwen2.5-VL sarcasm explanations")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--test_path", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--predictions_out", type=str, default="qwen2.5_vl_predictions.csv")
    parser.add_argument("--metrics_out", type=str, default="qwen2.5_vl_metrics.csv")
    return parser.parse_args()

def main():
    args = parse_args()
    random.seed(42)
    np.random.seed(42)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True, local_files_only=True)
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        local_files_only=True
    )
    model.eval()

    with open(args.test_path, "r") as f:
        lines = f.readlines()
        ds = [json.loads(line.strip()) for line in lines]
    print(f"📊 Loaded {len(ds)} samples from {args.test_path}")

    preds, refs = [], []
    for idx, ex in enumerate(tqdm(ds, desc="Generating")):
        messages = ex["messages"]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

        video_paths = [m["video"] for m in messages if isinstance(m.get("content"), list)
                   for m in m["content"] if isinstance(m, dict) and m.get("type") == "video"]
        print(f"🔍 [{idx}] Video sources: {video_paths}")
        print(f"🎞️  [{idx}] Loaded {len(video_inputs) if video_inputs else 0} video(s)")
        
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs
        ).to(model.device)

        gen_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.temperature > 0.0,
        }
        if args.temperature > 0.0:
            gen_kwargs["temperature"] = args.temperature

        with torch.no_grad():
            generated_ids = model.generate(**inputs, **gen_kwargs)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        reply = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0].strip()
        reference = next((m["content"] for m in messages if m["role"] == "assistant"), "").strip()

        preds.append(reply)
        refs.append(reference)

    df = pd.DataFrame({"prediction": preds, "reference": refs})
    df.to_csv(args.predictions_out, index=False)
    print(f"✔ Predictions saved to {args.predictions_out}")

    rouge = evaluate.load("rouge")
    meteor = evaluate.load("meteor")
    bleu = evaluate.load("bleu")
    bertscore = evaluate.load("bertscore")

    rouge_res = rouge.compute(predictions=preds, references=refs, rouge_types=["rougeL"])
    meteor_res = meteor.compute(predictions=preds, references=refs)
    bleu_res = bleu.compute(predictions=preds, references=[[r] for r in refs])
    bertscore_res = bertscore.compute(predictions=preds, references=refs, lang="en")

    metrics_dict = {
        "rougeL_f1": rouge_res.get("rougeL_fmeasure", rouge_res.get("rougeL", 0.0)),
        "meteor": meteor_res.get("meteor", 0.0),
        "bleu4": bleu_res.get("bleu", 0.0),
        "bertscore_f1": sum(bertscore_res["f1"]) / len(bertscore_res["f1"])
    }
    pd.DataFrame([metrics_dict]).to_csv(args.metrics_out, index=False)
    print(f"✔ Metrics saved to {args.metrics_out}")

if __name__ == "__main__":
    main()
