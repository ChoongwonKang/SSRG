import argparse, os, random, json, numpy as np, pandas as pd, torch
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM)
from tqdm import tqdm
import evaluate

os.environ["HF_HUB_OFFLINE"] = "1"

# ---------- 3‑shot few‑shot block (Gemma chat‑template roles) ----------
SHOT_EXAMPLES = [
    # 1) propositional
    {"role": "user", "content":
     "Explain why the target (final line) is sarcastic, using the provided context and multimodal cues.\n"
     "Each answer must start with \"This is propositional sarcasm,\" followed by \"since\" or \"because\".\n"
     "Keep them under 25 words.\n\n"
     "Context: PENNY with lid tighten, lip tighten, dimple, I don't think I've eaten that much in my entire life.\n"
     "Target: HOWARD: It's why my people wandered the desert for 40 years. Took that long to walk it off.\n"
     "Sarcasm type: propositional"},
    {"role": "model", "content":
     "This is propositional sarcasm because Howard humorously links overeating to a biblical trek, the opposite of literal meaning."},

    # 2) illocutionary
    {"role": "user", "content":
     "Explain why the target (final line) is sarcastic, using the provided context and multimodal cues.\n"
     "Each answer must start with \"This is illocutionary sarcasm,\" followed by \"since\" or \"because\".\n"
     "Keep them under 25 words.\n\n"
     "Context: RAJ: Why are you still here? ...\n"
     "Target: RAJ: Fine, I'll leave. But it sounds like somebody needs a fresh diaper.\n"
     "Sarcasm type: illocutionary"},
    {"role": "model", "content":
     "This is illocutionary sarcasm because Raj's tone mocks childishness, contradicting his literal offer to leave."},

    # 3) embedded
    {"role": "user", "content":
     "Explain why the target (final line) is sarcastic, using the provided context and multimodal cues.\n"
     "Each answer must start with \"This is embedded sarcasm,\" followed by \"since\" or \"because\".\n"
     "Keep them under 25 words.\n\n"
     "Context: Leonard and Raj discuss Giselle's photoshoot.\n"
     "Target: HOWARD: Sadly, Mrs. Giselle Wolowitz is sensitive to chlorine. Lucky for her I like my fashion models pruny and bug‑eyed.\n"
     "Sarcasm type: embedded"},
    {"role": "model", "content":
     "This is embedded sarcasm because ‘lucky for her’ clashes with the unflattering traits ‘pruny and bug‑eyed,’ showing the opposite sentiment."}
]
# ----------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser("Inference + Evaluation for Gemma fine‑tuned model")
    p.add_argument("--model_dir",   required=True, help="checkpoint or HF hub id")
    p.add_argument("--test_path",   required=True, help="test.jsonl")
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--predictions_out", default="")
    p.add_argument("--metrics_out",     default="")
    return p.parse_args()


def main():
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir, trust_remote_code=True, use_fast=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    ).eval()

    ds = load_dataset("json", data_files=args.test_path)["train"]

    preds, refs = [], []

    for ex in tqdm(ds, desc="Generating"):
        user_only = [m for m in ex["messages"] if m["role"] == "user"]
        msg_block = SHOT_EXAMPLES + user_only   

        prompt = tokenizer.apply_chat_template(
            msg_block, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        gen_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.temperature > 0,
            "temperature": max(args.temperature, 1e-5)
        }

        with torch.no_grad():
            gen_ids = model.generate(**inputs, **gen_kwargs)

        start = inputs["input_ids"].shape[-1]
        reply = tokenizer.decode(gen_ids[0][start:], skip_special_tokens=True).strip()
        preds.append(reply)

        ref = next((m["content"] for m in ex["messages"] if m["role"] == "model"), "")
        refs.append(ref.strip())

    pd.DataFrame({"prediction": preds, "reference": refs}).to_csv(args.predictions_out, index=False)
    print("✔ predictions saved to", args.predictions_out)

    # ----------------- metrics -----------------
    rouge     = evaluate.load("rouge")
    meteor    = evaluate.load("meteor")
    bleu      = evaluate.load("bleu")
    bertscore = evaluate.load("bertscore")

    rouge_res  = rouge.compute(predictions=preds, references=refs, rouge_types=["rougeL"])
    meteor_res = meteor.compute(predictions=preds, references=refs)
    bleu_res   = bleu.compute(predictions=preds, references=[[r] for r in refs])
    bert_res   = bertscore.compute(predictions=preds, references=refs, lang="en")

    metrics_dict = {
        "rougeL_f1":    rouge_res.get("rougeL_fmeasure", rouge_res.get("rougeL", 0.0)),
        "meteor":       meteor_res.get("meteor", 0.0),
        "bleu4":        bleu_res.get("bleu", 0.0),
        "bertscore_f1": float(np.mean(bert_res["f1"]))
    }

    pd.DataFrame([metrics_dict]).to_csv(args.metrics_out, index=False)
    print("✔ metrics saved to", args.metrics_out)


if __name__ == "__main__":
    main()