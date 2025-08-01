# 🧠 Clinical MCQ Solver using LoRA-tuned Language Models

This project fine-tunes a causal language model (Phi-3.5) using LoRA (Low-Rank Adaptation) to solve multiple-choice clinical reasoning tasks. It was developed as part of the **Goedel Machines x IITM Clinical LLM Challenge**.

## 🚀 Features

- Preprocessing of clinical MCQs into generative-style prompts
- Token-level loss masking to focus learning only on the answer portion
- LoRA fine-tuning with PEFT on HuggingFace Transformers
- Evaluation logic for predicting MCQ answers
- Submission pipeline generating CSV for Kaggle-style competitions

## 🛠️ Setup

Install dependencies:

```bash
pip install -q \
    transformers==4.40.2 \
    peft==0.11.0 \
    accelerate==0.30.0 \
    datasets
```

Ensure you have GPU access (`cuda`) enabled.

## 🧩 Training Pipeline

- **Tokenizer & Data Prep**: Converts question + options into a prompt
- **Loss Masking**: Loss computed only on the answer text
- **Model**: Phi-3.5 Mini Instruct + LoRA (targeting `qkv_proj`, `o_proj`)
- **Trainer**: HuggingFace `Trainer` API with `gradient_checkpointing` + FP16

Run training:
```python
trainer.train()
```

## 📈 Evaluation Logic

Evaluation decodes model output and compares predicted vs. true answers based on the string that follows `"Answer:"`.

Accuracy is computed across:
- Easy
- Medium
- Hard
test sets.

## 📦 Submission

The `submission.ipynb` file:
- Loads the fine-tuned model with adapter weights
- Performs predictions on test data
- Saves results to `submission.csv` (id → answer)

## 📂 Directory Structure

```
.
├── medical-llm.ipynb     # Training + preprocessing notebook
├── submission.ipynb      # Evaluation + submission script
├── /lora_phi3_stage1     # (output model dir)
└── submission.csv        # Final submission
```

## 📌 Notes

- Uses `LoRA` with PEFT for parameter-efficient fine-tuning
- Only trains specific layers of the base model for speed & memory
- Designed to work on Kaggle Kernel environments (uses `/kaggle/working`)

## 👥 Authors

This solution was developed by [Your Name or Team Name].
