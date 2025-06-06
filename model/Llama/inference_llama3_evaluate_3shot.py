# inference_llama3_evaluate_shot.py
import argparse
import os
import torch
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import evaluate
from tqdm import tqdm

os.environ["HF_HUB_OFFLINE"] = "1"

BOT_START    = "<|begin_of_text|>"
HEADER_OPEN  = "<|start_header_id|>"
HEADER_CLOSE = "<|end_header_id|>"
EOT          = "<|eot_id|>"

def apply_chat_template_llama31(messages):
    prompt = BOT_START + "\n"
    for msg in messages:
        prompt += f"{HEADER_OPEN}{msg['role']}{HEADER_CLOSE}\n"
        prompt += msg['content'].strip() + EOT + "\n"
    prompt += f"{HEADER_OPEN}assistant{HEADER_CLOSE}\n"
    return prompt

shot_examples = [
    # 1st example (propositional)
    {"role": "system", "content": (
        "You are a helpful assistant trained to inference a sentence into one of three sarcasm types: propositional, embedded, or illocutionary.\n"
        "We have three sarcasm types:\n"
        "1) Propositional: Opposite of actual meaning\n"
        "2) Embedded: Contradictory word/phrase in context\n"
        "3) Illocutionary: Tone/cues that oppose literal meaning."
    )},
    {"role": "user", "content": (
        "Explain why the target (final line) is sarcastic, using the provided context (previous lines) and multimodal cues.\n"
        "Each answer must start with \"This is propositional sarcasm,\" followed by \"since\" or \"because\" (no extra periods).\n"
        "Keep them under 25 words if possible.\n\n"
        "Context: PENNY with lid tighten, lip tighten, dimple, I don't think I've eaten that much in my entire life.\n"
        "Target: HOWARD: It's why my people wandered the desert for 40 years. Took that long to walk it off.\n"
        "Sarcasm type: propositional"
    )},
    {"role": "assistant", "content": (
        "This is propositional sarcasm because Howard implies excessive eating is a cultural trait, humorously exaggerating by referencing a historical event unrelated to overeating."
    )},
    # 2nd example (illocutionary)
    {"role": "system", "content": (
        "You are a helpful assistant trained to inference a sentence into one of three sarcasm types: propositional, embedded, or illocutionary.\n"
        "We have three sarcasm types:\n"
        "1) Propositional: Opposite of actual meaning\n"
        "2) Embedded: Contradictory word/phrase in context\n"
        "3) Illocutionary: Tone/cues that oppose literal meaning."
    )},
    {"role": "user", "content": (
        "Explain why the target (final line) is sarcastic, using the provided context (previous lines) and multimodal cues.\n"
        "Each answer must start with \"This is illocutionary sarcasm,\" followed by \"since\" or \"because\".\n"
        "Keep them under 25 words.\n\n"
        "Context: RAJ: Why are you still here? ...\n"
        "Target: RAJ: Fine, I'll leave. But it sounds like somebody needs a fresh diaper.\n"
        "Sarcasm type: illocutionary"
    )},
    {"role": "assistant", "content": (
        "This is illocutionary sarcasm, because Raj's tone and cheek raise suggest mockery, contrasting with the literal meaning of offering help."
    )},
    # 3rd example (embedded)
    {"role": "system", "content": (
        "You are a helpful assistant trained to inference a sentence into one of three sarcasm types: propositional, embedded, or illocutionary.\n"
        "We have three sarcasm types:\n"
        "1) Propositional: Opposite of actual meaning\n"
        "2) Embedded: Contradictory word/phrase in context\n"
        "3) Illocutionary: Tone/cues that oppose literal meaning."
    )},
    {"role": "user", "content": (
        "Explain why the target (final line) is sarcastic, using the provided context and multimodal cues.\n"
        "Each answer must start with \"This is embedded sarcasm,\" followed by \"since\" or \"because\".\n"
        "Keep them under 25 words.\n\n"
        "Context: Leonard and Raj discuss Giselle's photoshoot.\n"
        "Target: HOWARD: Sadly, Mrs. Giselle Wolowitz is sensitive to chlorine. Lucky for her I like my fashion models pruny and bug-eyed.\n"
        "Sarcasm type: embedded"
    )},
    {"role": "assistant", "content": (
        "This is embedded sarcasm because 'lucky for her' contradicts the negative traits 'pruny and bug-eyed,' implying it's not actually lucky or desirable."
    )}
]

def parse_args():
    parser = argparse.ArgumentParser(description="Inference & Evaluation for LLaMA-3.1 sarcasm explanations (3-shot)")
    parser.add_argument("--model_dir",       type=str,   required=True, help="Fine-tuned model directory")
    parser.add_argument("--test_path",       type=str,   required=True, help="Path to test.jsonl")
    parser.add_argument("--hf_token",        type=str,   default=None,   help="HF token for private model access")
    parser.add_argument("--max_new_tokens",  type=int,   default=1024,    help="")
    parser.add_argument("--temperature",     type=float, default=0.0,     help="")
    parser.add_argument("--predictions_out", type=str,   default="", help="")
    parser.add_argument("--metrics_out",     type=str,   default="",   help="")
    return parser.parse_args()


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        use_fast=True,
        local_files_only=True,
        use_auth_token=args.hf_token
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        use_auth_token=args.hf_token
    )
    model.eval()

    ds = load_dataset("json", data_files=args.test_path)["train"]

    preds, refs = [], []
    for ex in tqdm(ds, desc="Generating"):
        all_msgs = shot_examples + ex["messages"]
        prompt = apply_chat_template_llama31(all_msgs)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        gen_kwargs = {"max_new_tokens": args.max_new_tokens, "do_sample": args.temperature>0.0}
        if args.temperature>0.0: gen_kwargs["temperature"] = args.temperature
        with torch.no_grad():
            gen_ids = model.generate(**inputs, **gen_kwargs)
        input_len = inputs["input_ids"].shape[-1]
        reply = tokenizer.decode(gen_ids[0][input_len:], skip_special_tokens=True).strip()
        preds.append(reply)
        refs.append(ex.get("assistant", "").strip())

    pd.DataFrame({"prediction": preds, "reference": refs}).to_csv(args.predictions_out, index=False)
    print("✔ Predictions saved")
    rouge = evaluate.load("rouge"); meteor = evaluate.load("meteor")
    bleu = evaluate.load("bleu"); bertscore = evaluate.load("bertscore")
    rouge_res = rouge.compute(predictions=preds, references=refs, rouge_types=["rougeL"])
    meteor_res = meteor.compute(predictions=preds, references=refs)
    bleu_res = bleu.compute(predictions=preds, references=[[r] for r in refs])
    bertscore_res = bertscore.compute(predictions=preds, references=refs, lang="en")
    metrics = {
        "rougeL_f1":     rouge_res.get("rougeL_fmeasure", 0.0),
        "meteor":        meteor_res.get("meteor", 0.0),
        "bleu4":         bleu_res.get("bleu", 0.0),
        "bertscore_f1":  sum(bertscore_res.get("f1", [])) / max(len(bertscore_res.get("f1", [])), 1)
    }
    pd.DataFrame([metrics]).to_csv(args.metrics_out, index=False)
    print("✔ Metrics saved")

if __name__ == "__main__":
    main()
