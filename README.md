# Sarcasm Subtype-Specific Reasoning in Dialogue with Multimodal Cues Using Large Language Models

This repository contains the dataset and code for our CIKM 2025 paper.

<img width="2078" height="1174" alt="Fig1_v3" src="https://github.com/user-attachments/assets/1c5182a9-b98e-41fe-9aa3-a49c2c53172d" />

## 1. Download the SSRD Dataset

The **SSRD (Sarcasm Subtype-specific Reasoning Dataset)** is the core dataset used for fine-tuning and evaluating all models in this repository.  
### Dataset Structure
```
data/
├── ground_truth_sentence.csv
├── train_main.jsonl
├── val_main.jsonl
├── test_main.jsonl
└── MUSTARD++with_multimodal_cues.json
```
- `ground_truth_sentence.csv` contains the ground truth sentences corresponding to the sarcasm subtype of the final utterance in each scene.
- `train_main.jsonl`, `val_main.jsonl`, `test_main.jsonl` 
  JSON Lines files containing split subsets of the SSRD dataset.
- `MUSTARD++with_multimodal_cues.json`
is formatted in a **unified text representation** used for model training.  
Each entry contains sequential utterances within a scene, where only the **utterance** includes sarcasm annotations.

If `"sarcasm": true`, the entry specifies both the **sarcasm subtype** and a **ground truth reasoning sentence** explaining why it belongs to that subtype. In addition to textual content, each utterance integrates multimodal signals including:
   - **Top-3 Facial Action Units (FAUs)** with descriptive labels
   - **Six audio cues** representing acoustic features 



## 2. Steps to use the model
- We used **Qwen2.5-7B-Instruct**, **LLaMA3.1-8B-Instruct**, **Gemma2-9B-it**, and **Qwen2.5-7B-VL-Instruct** models on the SSRD dataset and evaluated the generated sentences for sarcasm subtype.
- All generated sentences and evaluation metric scores for eacl model are stored as `.csv` files within respective folders.

```
model/
├── Gemma/
│ ├── ds_zero3.json
│ ├── ft_gemma_full.py
│ ├── g_train.sh
│ ├── g_eval.sh
│ ├── g_inference_and_evaluate.py
│ ├── g_inference_and_evaluate_3shot.py
│ ├── metrics_fullft_.csv
│ └── prediction_fullft_.csv
│
├── Llama/
│ ├── ds_zero3.json
│ ├── llama_fullfinetuning.py
│ ├── l_train.sh
│ ├── l_eval.sh
│ ├── inference_llama3_evaluate.py
│ ├── inference_llama3_evaluate_3shot.py
│ ├── metrics_llama3_fullft_.csv
│ └── predictions_llama3_fullft_.csv
│
├── Qwen/
│ ├── ds_zero3.json
│ ├── ft_qwen_full.py
│ ├── q_train.sh
│ ├── q_eval.sh
│ ├── inference_and_evaluate.py
│ ├── inference_and_evaluate_3shot.py
│ ├── metrics_fullft_.csv
│ └── prediction_fullft_.csv
│
└── QwenVL/
├── zero3.json
├── sft7b.sh
├── eval.sh
├── eval_infer.py
├── eval_infer_3shot.py
├── metrics_fullft_.csv
└── prediction_fullft_.csv
```

### 2.1 Fine-tuning the model
Each model folder includes a `train.sh` file that loads `ds_zero3.json` and runs the full fine-tuning process.

**Gemma2 Fine-tuning**
```bash
cd model/Gemma
bash g_train.sh
```

**Llama3.1 Fine-tuning**
```bash
cd model/Llama
bash l_train.sh
```

**Qwen2.5 Fine-tuning**
```bash
cd model/Qwen
bash q_train.sh
```

**Qwen2.5-VL Fine-tuning**
```bash
cd model/QwenVL
bash sft7b.sh
```


### 2.2 Evaluating the fine-tuned model
After training, run the `eval.sh` in the same directory.
This script performs inference and computes metrics.

### 2.3 3-shot evaluation (option)

To perform 3-shot evaluation, modify `eval.sh` by setting the path of the pre-trained model from Hugging Face in the `MODEL_DIR` variable.  
Then, execute the following command to run the 3-shot inference:

```bash
python inference_and_evaluate_3shot.py
```

The fine-tuned models from this study are publicly available on Hugging Face.  

| Model | Checkpoint |
|--------|-------------------|
| **Qwen2.5-7B-Instruct_SSRG** | [![Checkpoint](https://img.shields.io/badge/Download-Checkpoint-blue?logo=huggingface)](https://huggingface.co/Choongwon/Qwen2.5-7B-Instruct_SSRG/tree/main) |
| **Llama-3.1-8B-Instruct SSRG** | [![Checkpoint](https://img.shields.io/badge/Download-Checkpoint-blue?logo=huggingface)](https://huggingface.co/Choongwon/Llama-3.1-8B-Instruct_SSRG/tree/main) |
| **Gemma2-9B-it_SSRG** | [![Checkpoint](https://img.shields.io/badge/Download-Checkpoint-blue?logo=huggingface)](https://huggingface.co/Choongwon/gemma-2-9b-it_SSRG/tree/main) |
| **Qwen2.5-7B-VL-Instruct_SSRG** | [![Checkpoint](https://img.shields.io/badge/Download-Checkpoint-blue?logo=huggingface)](https://huggingface.co/Choongwon/Qwen2.5-VL-7B-Instruct_SSRG/tree/main) |

