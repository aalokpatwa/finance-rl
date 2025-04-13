from openai import OpenAI
import json
import re
from typing import Dict, Any, List, Optional
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import functools

def extract_numerical_answer(text: str) -> Optional[float]:
    match = re.search(r"<answer>([\s\S]*?)<\/answer>", text)
    if match:
        try:
            num_str = match.group(1).strip().replace(',', '')
            return round(float(num_str), 2)
        except ValueError:
            return None
    return None

def process_example(example_str: str, client: OpenAI, model: str) -> Dict[str, Any]:
    try:
        example = json.loads(example_str)
        prompt = f"""As a financial professional, please analyze the following context containing financial data about the target company to answer the given question, and provide a numerical answer between <answer> tags, rounded to two decimals.

Context: {example['context']}

Question: {example['question']}

Please think step by step and then put your final numerical answer between <answer> and </answer> tags."""

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful financial analyst assistant."},
                {"role": "user", "content": prompt}
            ],
        )

        prediction = extract_numerical_answer(response.choices[0].message.content)
        if prediction is not None:
            return {
                "status": "success",
                "prediction": prediction,
                "ground_truth": float(example['answer']),
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
        predictions_np = np.array(predictions) 
        ground_truths_np = np.array(ground_truths) 

        correct = np.isclose(predictions_np, ground_truths_np, atol=0.5).sum()
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
    MODEL = "o3-mini"  
    TEST_FILE = "test.jsonl"
    
    results = evaluate_model(MODEL, TEST_FILE)
    print(json.dumps(results, indent=2))