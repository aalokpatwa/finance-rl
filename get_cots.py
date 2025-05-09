import dataloader
import json
import openai
import re
import math
from evaluate_utils import extract_answer, extract_reasoning, check_correctness
from tqdm import tqdm
import threading

# Global variables for progress tracking
processed_count = 0
processed_count_lock = threading.Lock()
progress_bar = None

def get_openai_client() -> openai.OpenAI:
    client = openai.OpenAI()
    return client

def get_openai_prompt(prompt: str, system_prompt: str):
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]

def get_openai_response(openai_prompt: list[dict[str, str]]):
    client = get_openai_client()
    response = client.chat.completions.create(
        model="o3-mini",
        messages=openai_prompt,
    )
    return response.choices[0].message.content

def get_lines(path: str) -> list[dict[str, str]]:
    with open(path, "r") as file:
        return [json.loads(line) for line in file.readlines()]

# Worker function to process a single example and write to file
def process_example_and_write(example: dict, cot_list_lock: threading.Lock, shared_output_cots: list, output_file_path: str, pre_prompt_template: str, system_prompt_template: str):
    global processed_count, processed_count_lock, progress_bar

    new_row = example.copy()
    try:
        openai_prompt = get_openai_prompt(pre_prompt_template.format(context=example["context"], question=example["question"]), system_prompt_template)
        true_answer = example["answer"]
        
        response_content = get_openai_response(openai_prompt)
        openai_answer = extract_answer(response_content)
                
        if check_correctness(openai_answer, true_answer):
            reasoning = extract_reasoning(response_content)
            new_row["reasoning"] = reasoning
            with cot_list_lock:
                shared_output_cots.append(new_row)
                # Re-write the entire list to disk under lock
                with open(output_file_path, "w") as file:
                    for item in shared_output_cots:
                        json.dump(item, file)
                        file.write("\n")
    except Exception as e:
        print(f"Thread {threading.get_ident()}: Error processing example (ID: {example.get('id', 'N/A')}): {e}")
    finally:
        with processed_count_lock:
            processed_count += 1
            if progress_bar:
                progress_bar.update(1)

# Thread's main task: process a shard of examples
def worker_thread_task(shard: list[dict], cot_list_lock: threading.Lock, shared_output_cots: list, output_file_path: str, pre_prompt_template: str, system_prompt_template: str):
    for example in shard:
        process_example_and_write(example, cot_list_lock, shared_output_cots, output_file_path, pre_prompt_template, system_prompt_template)

def main():
    global progress_bar # To initialize and close the global progress bar

    train_examples = get_lines("train.jsonl")
    SYSTEM_PROMPT = """You must respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>"""
    pre_prompt = (
        "You will be given some context regarding a company's financials, and your task is to answer a question about the company.\n"
        "Put your reasoning process between <reasoning> tags and your final answer between <answer> tags, like this:\n"
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
    
    output_cots = [] # This list will be shared and modified by threads
    output_file_path = "o3-mini-cots.jsonl" # Output file
    cot_lock = threading.Lock() # Lock for synchronized access to output_cots and file writing

    NUM_THREADS = 20

    if not train_examples:
        print("No examples to process.")
        return

    # Initialize progress bar
    progress_bar = tqdm(total=len(train_examples), desc="Generating COTs")

    # Shard the train_examples
    num_examples = len(train_examples)
    examples_per_thread = math.ceil(num_examples / NUM_THREADS)
    shards = []
    if num_examples > 0:
        shards = [train_examples[i:i + examples_per_thread] for i in range(0, num_examples, examples_per_thread)]
    
    threads = []
    actual_num_threads = min(NUM_THREADS, len(shards)) # Use at most NUM_THREADS or number of shards

    for i in range(actual_num_threads):
        shard = shards[i]
        thread = threading.Thread(
            target=worker_thread_task, 
            args=(shard, cot_lock, output_cots, output_file_path, pre_prompt, SYSTEM_PROMPT)
        )
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    progress_bar.close()
    print(f"All {actual_num_threads} threads finished. {len(output_cots)} COTs generated.")
    print(f"Final COTs written to {output_file_path}")
        
if __name__ == "__main__":
    main()