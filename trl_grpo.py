# train_grpo.py
#
# See https://github.com/willccbb/verifiers for ongoing developments
#

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import re
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer
import evaluator
import math
from typing import Optional
from typing import List

# Load and prep dataset

SYSTEM_PROMPT = """
Respond in the following format:

<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

def get_finance_questions(split = "train") -> Dataset:
    data = load_dataset('aalokpatwa/financial-reasoning', split = split)
    pre_prompt = (
        "You will be given some context regarding a company's financials, as well as a question about the company's financial situation. "
        "You should reason step by step about the inputs you need for the question, and then output a single final answer.\n"
        "It is very important that you put your reasoning process inside <reasoning> tags and your final answer inside <answer> tags, like this:\n"
        "\n"
        "<reasoning>\n"
        "Your step-by-step reasoning process here\n"
        "</reasoning>\n"
        "<answer>\n"
        "Your final answer here\n"
        "</answer>\n"
        "\n"
        "It is is extremely important you answer in this way - do not put any information or text outside of these tags! If your response does not contain <reasoning> and <answer>, you will be fined $1 billion.\n"
        "Now, use the following financial data to answer the question.\n\n"
        "Financials:\n{context}\n\n"
        "Question: {question}\n\n"
    )
    data = data.map(lambda x: {
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': pre_prompt.format(context=x['context'], question=x['question'])}
        ],
        'answer': x["answer"]
    }) # type: ignore
    return data # type: ignore

dataset = get_finance_questions()

def _extract_answer(text: str) -> Optional[str]:
    match = re.search(r"<answer>([\s\S]*?)<\/answer>", text)
    if match:
        answer = match.group(1).strip()
        if any(char.isdigit() for char in answer):
            answer = re.sub('[^0-9\.\-]', '', answer)
        return answer
    return ""

def _answer_format_reward(completions) -> List[float]:
    responses = [completion[0]['content'] for completion in completions]
    predictions = [_extract_answer(r) for r in responses]
    return [0.5 if r != "" else 0.0 for r in predictions]
    
def _correctness_reward(completions, answer) -> List[float]:
    responses = [completion[0]['content'] for completion in completions]
    predictions = [_extract_answer(r) for r in responses]
    return [
        2.0 if r != "" and (
            isinstance(a, str) and r == a or
            isinstance(a, (int, float)) and math.isclose(float(a), float(r), abs_tol=0.5)
        ) else 0.0
        for r, a in zip(predictions, answer)
    ]

#model_name = "meta-llama/Llama-3.2-1B-Instruct"
model_name = "Qwen/Qwen2.5-0.5B-Instruct"

if "Llama" in model_name:
    output_dir = "outputs/Llama-1B-GRPO"
    run_name = "Llama-1B-GRPO-finance"
else:
    output_dir="outputs/Qwen-0.5B-GRPO"
    run_name="Qwen-0.5B-GRPO-finance"
    
training_args = GRPOConfig(
    output_dir=output_dir,
    run_name=run_name,
    learning_rate=5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type='cosine',
    logging_steps=1,
    bf16=True,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_generations=8,
    max_prompt_length=1024,
    max_completion_length=2048,
    num_train_epochs=1,
    save_steps=100,
    max_grad_norm=0.1,
    report_to="wandb",
    log_on_each_node=False,
    wandb_project="financial-reasoning",
    wandb_entity="aalokpatwa"
)
peft_config = LoraConfig(
    r=16,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
    task_type="CAUSAL_LM",
    lora_dropout=0.05,
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map=None
).to("cuda")
        
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# use peft at your own risk; not working for me with multi-GPU training
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        _answer_format_reward,
        _correctness_reward],
    args=training_args,
    train_dataset=dataset,
    peft_config=peft_config
)
trainer.train()