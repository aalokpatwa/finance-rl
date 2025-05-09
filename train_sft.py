"""
SFT Qwen2.5‑0.5B on FinReason
"""
import os
import re
import json
import math
from typing import Optional, List, Dict, Any

import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer

# ────────────────────────────────────────────────────────────────────────────────
# Environment & model choice
# ────────────────────────────────────────────────────────────────────────────────
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # Edit as desired
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = "outputs/Qwen-0.5B-SFT"
RUN_NAME = "Qwen-0.5B-SFT-finance"

# ────────────────────────────────────────────────────────────────────────────────
# Prompt template helpers
# ────────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """
Respond in the following format:

<reasoning>
...
</reasoning>
<answer>
...
</answer>
""".strip()

PRE_PROMPT = (
    "You will be given some context regarding a company's financials, and your task is to answer a question about the company.\n"
    "Put your reasoning process between <reasoning> tags and your final answer between <answer> tags, like this:\n\n"
    "<reasoning>\n"
    "Your step-by-step reasoning process here\n"
    "</reasoning>\n"
    "<answer>\n"
    "Your final answer here\n"
    "</answer>\n\n"
    "It is extremely important you answer in this way – do not put any information or text outside of these tags! If your response does not contain <reasoning> and <answer>, you will be fined $1 billion.\n"
    "Now, use the following financial data to answer the question.\n\n"
    "Financials:\n{context}\n\n"
    "Question: {question}\n\n"
)

# The training set includes the correct answer, so we embed it as the assistant turn.
ASSISTANT_TEMPLATE = (
    "<reasoning>{reasoning}</reasoning>\n"
    "<answer>{answer}</answer>"
)

# ────────────────────────────────────────────────────────────────────────────────
# Dataset preparation
# ────────────────────────────────────────────────────────────────────────────────

def _load_split(split: str) -> Dataset:
    filename = "train.jsonl" if split == "train" else "test.jsonl"
    data = load_dataset("json", data_files={split: filename}, split=split)

    def _add_conversation(example: Dict[str, Any]) -> Dict[str, Any]:
        user_message = PRE_PROMPT.format(context=example["context"], question=example["question"])
        example["prompt_messages"] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
        return example

    data = data.map(_add_conversation)
    return data  # type: ignore

raw_train = _load_split("train")
raw_test = _load_split("test")

# Instantiate tokenizer early so we can apply its chat template
print("Loading tokenizer…")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

# For SFT we need a plain string per example. We build it once and cache.

def _format_chat(messages: List[Dict[str, str]], *, for_generation: bool = False) -> str:
    """Serialize messages via tokenizer's chat template."""
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=for_generation,
    )

# Build training texts (prompt + ground‑truth answer)


def _build_train(example: Dict[str, Any]) -> Dict[str, Any]:
    assistant_turn = {
        "role": "assistant",
        "content": ASSISTANT_TEMPLATE.format(answer=example["answer"], reasoning=example["reasoning"]),
    }
    full_conv = example["prompt_messages"] + [assistant_turn]
    return {"messages": full_conv}


# Build evaluation prompts (without answer)

def _build_eval(example: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "message": _format_chat(example["prompt_messages"]),
        "answer": example["answer"],
    }

train_dataset = raw_train.map(_build_train, remove_columns=raw_train.column_names)
print (train_dataset[0])
eval_dataset = raw_test.map(_build_eval, remove_columns=raw_test.column_names)

# ------------
# Load the model and LoRA config
# ------------
quant_cfg = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    quantization_config=quant_cfg,
)

lora_cfg = LoraConfig(
    r=16,
    lora_alpha=64,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "up_proj",
        "down_proj",
        "gate_proj",
    ],
    modules_to_save=["lm_head", "embed_token"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

# ────────────────────────────────────────────────────────────────────────────────
# Training arguments & trainer
# ────────────────────────────────────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    run_name=RUN_NAME,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    num_train_epochs=1,
    learning_rate=5e-6,
    warmup_ratio=0.1,
    optim="adamw_torch_fused",
    fp16=True,
    logging_steps=10,
    save_steps=100,
    report_to=["wandb"],  # comment out if not using Weights & Biases
    lr_scheduler_type="cosine",
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    peft_config=lora_cfg,
)

# ────────────────────────────────────────────────────────────────────────────────
# Training
# ────────────────────────────────────────────────────────────────────────────────
print("Starting supervised fine‑tuning…")
trainer.train()
trainer.save_model(OUTPUT_DIR)
print(f"Model & LoRA adapters saved to {OUTPUT_DIR}")
print("Pushing model to the HuggingFace hub…")
trainer.push_to_hub("aalokpatwa/qwen-fin-sft")

# ────────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ────────────────────────────────────────────────────────────────────────────────

def _extract_answer(text: str) -> str:
    """Grab the <answer>…</answer> segment (digits stripped if numeric)."""
    match = re.search(r"<answer>([\s\S]*?)</answer>", text)
    if match:
        ans = match.group(1).strip()
        if any(ch.isdigit() for ch in ans):
            ans = re.sub(r"[^0-9]", "", ans)
        return ans
    return ""


def _is_correct(pred: str, gold: str) -> bool:
    if pred == "":
        return False
    # numeric answers – allow small absolute error
    if gold.replace(".", "", 1).replace("-", "", 1).isdigit():
        try:
            return math.isclose(float(gold), float(pred), abs_tol=0.5)
        except ValueError:
            return False
    # otherwise case‑insensitive string match
    return pred.lower() == gold.lower()


# ────────────────────────────────────────────────────────────────────────────────
# Evaluation loop
# ────────────────────────────────────────────────────────────────────────────────
print("Running evaluation on the test split…")
model.eval()
correct = 0
for ex in eval_dataset:
    prompt_ids = tokenizer(ex["messages"], return_tensors="pt").to(model.device)
    with torch.no_grad():
        gen_out = model.generate(**prompt_ids, max_new_tokens=512)
    full_text = tokenizer.decode(gen_out[0], skip_special_tokens=True)
    pred_ans = _extract_answer(full_text)
    if _is_correct(pred_ans, ex["answer"]):
        correct += 1

accuracy = correct / len(eval_dataset)
print(f"Test accuracy: {accuracy:.2%} ({correct}/{len(eval_dataset)})")

# Also dump predictions for manual inspection
pred_file = os.path.join(OUTPUT_DIR, "predictions.jsonl")
with open(pred_file, "w", encoding="utf-8") as f:
    for ex in eval_dataset:
        prompt_ids = tokenizer(ex["messages"], return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen_out = model.generate(**prompt_ids, max_new_tokens=512)
        full_text = tokenizer.decode(gen_out[0], skip_special_tokens=True)
        pred_ans = _extract_answer(full_text)
        out = {
            "prompt": ex["messages"],
            "gold_answer": ex["answer"],
            "predicted_answer": pred_ans,
            "full_completion": full_text,
        }
        f.write(json.dumps(out, ensure_ascii=False) + "\n")
print(f"Detailed completions written to {pred_file}")
