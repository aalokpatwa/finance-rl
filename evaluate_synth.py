from openai import OpenAI
import json
import re
from typing import Dict, Any, List, Optional
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import functools

def extract_answer(text: str) -> Optional[str]:
    match = re.search(r"<answer>([\s\S]*?)<\/answer>", text)
    if match:
        answer = match.group(1).strip()
        if any(char.isdigit() for char in answer):
            answer = re.sub('[^0-9\.\-]', '', answer)
        return answer
    return None

def process_example(example_str: str, client: OpenAI, model: str) -> Dict[str, Any]:
    try:
        example = json.loads(example_str)
        SYSTEM_PROMPT = """Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>"""
        prompt = f"""
You will be given a financial question about the provided financial data of a company. Reason about it step-by-step, putting your thinking between <reasoning> and </reasoning> XML tags and then put your final numerical answer between <answer> and </answer> tags.
It is crucial that all text is between either <reasoning> or <answer> tags. If you do not follow this, you will be fined $1 billion.

Financials: {example['context']}

Question: {example['question']}

"""

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
        )

        prediction = extract_answer(response.choices[0].message.content)
        if prediction is not None:
            return {
                "status": "success",
                "prediction": prediction,
                "ground_truth": example['answer'],
                "id": example.get('id', 'unknown')
            }
        else:
            return {
                "status": "error",
                "error_message": "Could not extract numerical answer",
                "id": example.get('id', 'unknown')
             }

    except Exception as e:
        example_id = 'unknown'
        try:
            example_id = json.loads(example_str).get('id', 'unknown')
        except json.JSONDecodeError:
            pass 
        return {
            "status": "error",
            "error_message": str(e),
            "id": example_id
        }

def evaluate_model(model: str, test_file: str) -> Dict[str, Any]:
    errors = []
    predictions = []
    ground_truths = []
    example_lines: List[str] = [] 

    client = OpenAI()

    try:
        with open(test_file, 'r') as f:
            example_lines = [line for line in f if line.strip()] 
    except FileNotFoundError:
         return {
            "error": f"Test file not found: {test_file}"
         }

    worker_func = functools.partial(process_example, client=client, model=model)
    num_workers = 10

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results_iterator = executor.map(worker_func, example_lines)

        for i, result in enumerate(results_iterator):
            print(f"Processing example {i+1}/{len(example_lines)}...") 
            if result.get("status") == "success":
                predictions.append(result["prediction"])
                ground_truths.append(result["ground_truth"])
                print (result["prediction"], result["ground_truth"]) 
            elif result.get("status") == "error":
                errors.append({"id": result.get('id', 'unknown'), "error": result.get("error_message", "Unknown error")})
            else:
                 errors.append({"id": result.get('id', 'unknown'), "error": "Unknown processing result format"})


    if predictions:
        correct = 0
        for pred, gt in zip(predictions, ground_truths):
            try:
                pred_num = float(pred)
                gt_num = float(gt)
                if np.isclose(pred_num, gt_num, atol=0.5):
                    correct += 1
            except ValueError:
                if str(pred).lower() == str(gt).lower():
                    correct += 1

        accuracy = correct / len(predictions) if len(predictions) > 0 else 0.0

        final_results = { 
            "total_examples": len(example_lines),
            "successfully_processed": len(predictions), 
            "errors": len(errors),
            "accuracy_atol_0.5": accuracy, 
            "error_details": errors
        }

    else:
         final_results = { 
            "total_examples": len(example_lines),
            "successfully_processed": 0, 
            "errors": len(errors),
            "accuracy_atol_0.5": 0.0, 
            "error_details": errors
        }

    print(f"\nEvaluation Summary:\n{json.dumps(final_results, indent=2)}")
    return final_results

if __name__ == "__main__":
    #MODEL = "o3-mini"  
    #MODEL = "ft:gpt-4o-mini-2024-07-18:personal::BN5A693W" # trained on easy
    #MODEL = "ft:gpt-4o-mini-2024-07-18:personal:initial-test:BN33mdlS" # trained on hard
    MODEL = "gpt-4o-mini-2024-07-18" 
    TEST_FILE = "test.jsonl"
    
    results = evaluate_model(MODEL, TEST_FILE)