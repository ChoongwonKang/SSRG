import pandas as pd
import evaluate
import random
import numpy as np

random.seed(42)
np.random.seed(42)

file_path = r""  


df = pd.read_csv(file_path)

preds = df["prediction"].tolist()
refs = df["reference"].tolist()

rouge     = evaluate.load("rouge")
meteor    = evaluate.load("meteor")
bleu      = evaluate.load("bleu")
bertscore = evaluate.load("bertscore")

rouge_res     = rouge.compute(predictions=preds, references=refs, rouge_types=["rougeL"])
meteor_res    = meteor.compute(predictions=preds, references=refs)
bleu_res      = bleu.compute(predictions=preds, references=[[r] for r in refs])
bertscore_res = bertscore.compute(predictions=preds, references=refs, lang="en")

rouge_l_f1 = rouge_res.get("rougeL_fmeasure", rouge_res.get("rougeL", 0.0))
avg_bertscore = sum(bertscore_res["f1"]) / len(bertscore_res["f1"]) if bertscore_res["f1"] else 0.0

metrics = {
    "rougeL_f1": rouge_l_f1,
    "meteor":    meteor_res.get("meteor", 0.0),
    "bleu4":     bleu_res.get("bleu", 0.0),
    "bertscore_f1": avg_bertscore
}

print("Evaluation Results:")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")
