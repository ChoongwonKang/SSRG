import argparse
import torch
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import evaluate
from tqdm import tqdm

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
    shot_examples = [
        # 1st example (propositional)
        {"role": "system", "content": "You are a helpful assistant trained to inference a sentence into one of three sarcasm types: propositional, embedded, or illocutionary.\nWe have three sarcasm types:\n1) Propositional: Opposite of actual meaning\n2) Embedded: Contradictory word/phrase in context\n3) Illocutionary: Tone/cues that oppose literal meaning."},
        {"role": "user", "content": "Explain why the target (final line) is sarcastic, using the provided context (previous lines) and multimodal cues (FAU, Audio).\nEach answer must start with \"This is propositional sarcasm,\" followed by \"since\" or \"because\" (no extra periods).\nKeep them under 25 words if possible.\n\nContext (previous lines): \"PENNY with lid tighten, lip tighten, dimple, I don't think I've eaten that much in my entire life. 216.52, 5546.08, 45.15, 674.91, 0.02, 0.13 (F0 mean, F0 var, Energy mean, Energy var, Jitter, Shimmer).\"\n\nTarget (final line):\nSpeaker: HOWARD\nFAU: cheek raise, lip tighten, dimple\nAudio (F0 mean, F0 var, Energy mean, Energy var, Jitter, Shimmer): 123.65, 1789.81, 60.25, 180.79, 0.03, 0.16\nUtterance: It's why my people wandered the desert for 40 years. Took that long to walk it off.\n\nSarcasm type: propositional\n\nWhy does the final line belong to propositional sarcasm?\nPlease provide answers in the required format."},
        {"assistant": "This is propositional sarcasm because Howard implies excessive eating is a cultural trait, humorously exaggerating by referencing a historical event unrelated to overeating."},
        # 2nd example (illocutionary)
        {"role": "system", "content": "You are a helpful assistant trained to inference a sentence into one of three sarcasm types: propositional, embedded, or illocutionary.\nWe have three sarcasm types:\n1) Propositional: Opposite of actual meaning\n2) Embedded: Contradictory word/phrase in context\n3) Illocutionary: Tone/cues that oppose literal meaning."},
        {"role": "user", "content": "Explain why the target (final line) is sarcastic, using the provided context (previous lines) and multimodal cues (FAU, Audio).\nEach answer must start with \"This is illocutionary sarcasm,\" followed by \"since\" or \"because\" (no extra periods).\nKeep them under 25 words if possible.\n\nContext (previous lines): \"RAJ with lid tighten, lip tighten, dimple, so in a way, Howard's not only like your father 185.33, 1414.83, 55.74, 243.24, 0.02, 0.12 (F0 mean, F0 var, Energy mean, Energy var, Jitter, Shimmer). RAJ with lid tighten, lip tighten, dimple, but he's also like the child that you're afraid to have. 166.35, 756.96, 57.31, 94.32, 0.03, 0.15 (F0 mean, F0 var, Energy mean, Energy var, Jitter, Shimmer). HOWARD with lid tighten, lip tighten, dimple, Why are you still here? 217.43, 7931.58, 52.26, 128.54, 0.03, 0.20 (F0 mean, F0 var, Energy mean, Energy var, Jitter, Shimmer).\"\n\nTarget (final line):\nSpeaker: RAJ\nFAU: cheek raise, dimple, lip tighten\nAudio (F0 mean, F0 var, Energy mean, Energy var, Jitter, Shimmer): 172.44, 2234.98, 60.47, 95.17, 0.02, 0.13\nUtterance: Fine, I'll leave. But it sounds like somebody needs a fresh diaper.\n\nSarcasm type: illocutionary\n\nWhy does the final line belong to illocutionary sarcasm?\nPlease provide answers in the required format."},
        {"assistant": "This is illocutionary sarcasm, because Raj's tone and cheek raise suggest mockery, contrasting with the literal meaning of offering help."},
        # 3rd example (embedded)
        {"role": "system", "content": "You are a helpful assistant trained to inference a sentence into one of three sarcasm types: propositional, embedded, or illocutionary.\nWe have three sarcasm types:\n1) Propositional: Opposite of actual meaning\n2) Embedded: Contradictory word/phrase in context\n3) Illocutionary: Tone/cues that oppose literal meaning."},
        {"role": "user", "content": "Explain why the target (final line) is sarcastic, using the provided context (previous lines) and multimodal cues (FAU, Audio).\nEach answer must start with \"This is embedded sarcasm,\" followed by \"since\" or \"because\" (no extra periods).\nKeep them under 25 words if possible.\n\nContext (previous lines): \"LEONARD with lid tighten, lip tighten, dimple, Did I miss anything? Did they kick Giselle off? 248.92, 1271.99, 73.23, 54.87, 0.03, 0.13 (F0 mean, F0 var, Energy mean, Energy var, Jitter, Shimmer). RAJ with lid tighten, lip tighten, dimple, Not yet, but her underwater photo shoot was an embarrassment. 169.86, 593.01, 75.22, 36.16, 0.02, 0.14 (F0 mean, F0 var, Energy mean, Energy var, Jitter, Shimmer).\"\n\nTarget (final line):\nSpeaker: HOWARD\nFAU: cheek raise, lip tighten, dimple\nAudio (F0 mean, F0 var, Energy mean, Energy var, Jitter, Shimmer): 143.81, 3993.63, 71.48, 52.03, 0.02, 0.12\nUtterance: Sadly, Mrs. Giselle Wolowitz is sensitive to chlorine. Lucky for her I like my fashion models pruny and bug-eyed.\n\nSarcasm type: embedded\n\nWhy does the final line belong to embedded sarcasm?\nPlease provide answers in the required format."},
        {"assistant": "This is embedded sarcasm because \"lucky for her\" contradicts the negative traits \"pruny and bug-eyed,\" implying it's not actually lucky or desirable."}   
    ]

    for ex in tqdm(ds, desc="Generating"):
        all_msgs = shot_examples + ex["messages"]
        prompt = tokenizer.apply_chat_template(
            all_msgs, tokenize=False, add_generation_prompt=True
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

    avg_bertscore = sum(bertscore_res["f1"]) / len(bertscore_res["f1"]) if bertscore_res["f1"] else 0.0

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
