from openai import OpenAI
import re
import csv
import json
from typing import Dict, Any, List, Optional
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import functools

def extract_answer(text: str) -> Optional[str]:
    match = re.search(r"<answer>([\s\S]*?)<\/answer>", text)
    if match:
        return match.group(1).strip()
    return None

def process_example(example: Dict[str, Any], client: OpenAI, model: str) -> Dict[str, Any]:
    try:
        prompt = f"""As a financial professional, please analyze the following context containing financial data about the target company to answer the given question, and provide an answer between <answer> tags.

Context: {example['context']}

Question: {example['question']}

Please think step by step and then put your final answer between <answer> and </answer> tags."""

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful financial analyst assistant."},
                {"role": "user", "content": prompt}
            ],
        )

        prediction = extract_answer(response.choices[0].message.content)
        if prediction is not None:
            return {
                "status": "success",
                "prediction": prediction,
                "ground_truth": example['answer'],
            }
        else:
            return {
                "status": "error",
                "error_message": "Could not extract numerical answer",
             }
    except Exception as e:
        return {
            "status": "error",
            "error_message": str(e)
        }

def evaluate_model(model: str, test_file: str) -> Dict[str, Any]:
    errors = []
    predictions = []
    ground_truths = []
    examples: List[Dict[str, Any]] = [] 

    client = OpenAI()

    try:
        with open(test_file, 'r', newline='') as f: 
            reader = csv.DictReader(f)
            required_headers = {'context', 'question', 'answer', 'question_type'}
            if not required_headers.issubset(reader.fieldnames):
                missing = required_headers - set(reader.fieldnames)
                return {"error": f"CSV file missing required headers: {', '.join(missing)}"}
                
            examples = [row for row in reader if row['question_type'] == 'basic']  
            if not examples:
                 return {"error": f"No rows with question_type = 'basic' in CSV file: {test_file}"}
                 
    except FileNotFoundError:
         return {
            "error": f"Test file not found: {test_file}"
         }
    except Exception as e: 
        return {
            "error": f"Error reading CSV file {test_file}: {str(e)}"
        }

    worker_func = functools.partial(process_example, client=client, model=model)
    num_workers = 10

    total_examples_count = len(examples) 

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results_iterator = executor.map(worker_func, examples)

        for i, result in enumerate(results_iterator):
            print(f"Processing example {i+1}/{total_examples_count}...") 
            if result.get("status") == "success":
                predictions.append(result["prediction"])
                ground_truths.append(result["ground_truth"])
            elif result.get("status") == "error":
                errors.append({"id": result.get('id', 'unknown'), "error": result.get("error_message", "Unknown error")})


    if predictions:
        with open('output.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['prediction', 'ground_truth'])
            for pred, gt in zip(predictions, ground_truths):
                writer.writerow([pred, gt])

    else:
        print ("No successful predictions were registered.")

if __name__ == "__main__":
    MODEL = "o3-mini"  
    TEST_FILE = "afterquery_test.csv"
    
    results = evaluate_model(MODEL, TEST_FILE)
    print(json.dumps(results, indent=2))