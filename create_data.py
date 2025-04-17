import random
from typing import List, Tuple
import json
from data_taxonomy import (
    get_formatted_metric, get_metric_reasoning, get_number_of_years_required,
    get_all_derived_metrics, get_all_basic_metrics, DifficultyLevel,
    METRIC_DEFINITIONS, QuestionType
)

def get_years(data: dict) -> List[int]:
    return list(data.keys())

def represent_data_as_table(data: dict) -> str:
    """ Converts a dictionary of line items for multiple years into a table

    Args:
        data (dict): maps years to nested dictionary of line item, value

    Returns:
        str: text-based table representation of the data, as would be extracted from a PDF
    """
    years: List[int] = get_years(data)
    
    table_str = ""
    
    units: str = "$"
    
    delimiter = random.choice([",", " ", "|"])
    
    table_str += units + delimiter
    for year in years[:-1]:
        table_str += str(year) + delimiter
    table_str += str(years[-1]) + "\n"
    
    all_line_items = list(data[years[0]].keys())
    
    for line_item in all_line_items:
        table_str += line_item + delimiter
        for year in years[:-1]:
            table_str += str(data[year][line_item]) + delimiter
        table_str += str(data[years[-1]][line_item]) + "\n"
    
    table_str = table_str.strip()
    
    return table_str

def generate_sample(data: dict, derived_metrics: List[Tuple[str, DifficultyLevel]]) -> str:
    metric_choice, difficulty = random.choice(derived_metrics)
    num_years_required: int = get_number_of_years_required(metric_choice)
    
    possible_years: List[int] = get_years(data)
    years: List[int] = sorted(random.sample(possible_years, num_years_required))
    
    metric_def = METRIC_DEFINITIONS[metric_choice]
    
    if metric_def.question_templates:
        # Randomly select a question template
        question_template = random.choice(metric_def.question_templates)
        
        # Format the template with the appropriate years
        if num_years_required > 1:
            question_str = question_template.format(start_year=years[0], end_year=years[-1])
        else:
            question_str = question_template.format(year=years[0])
    else:
        # Fallback to original format if no templates are defined
        question_str = f"Calculate {metric_choice}"
        if "Margin" in metric_choice or "Change" in metric_choice or "Growth" in metric_choice:
            question_str += " (as a percentage)"
        if num_years_required > 1:
            question_str += f" from {years[0]} to {years[-1]}"
        else:
            question_str += f" for {years[0]}."
            
    if metric_def.question_type == QuestionType.COMPARE:
        question_str += " Answer yes or no."
    else:
        question_str += " Give your answer to one decimal place."
    
    answer_value = get_formatted_metric(metric_choice, data, *years)
    
    return {
        "context": represent_data_as_table(data),
        "question": question_str,
        "reasoning": get_metric_reasoning(metric_choice, data, *years),
        "answer": answer_value,
        "difficulty": difficulty.name,
    }
    
def main():
    # Define how many samples we are going to generate
    num_train_samples: int = 3000
    num_test_samples: int = round(num_train_samples * 0.1)
    
    # Get all the basic metrics
    basic_metrics: List[Tuple[str, DifficultyLevel, bool]] = get_all_basic_metrics()
    
    # Get all the derived metrics
    derived_metrics: List[Tuple[str, DifficultyLevel]] = get_all_derived_metrics()
    
    # Store the generated samples
    train_samples: List[dict] = []
    test_samples: List[dict] = []
        
    # Generate the samples
    for _ in range(num_train_samples + num_test_samples):
        
        # Generate a new income / balance sheet for each example
        
        # Defines the number of years to include in the data
        number_of_years = random.randint(2, 6)
        starting_year = random.randint(2000, 2024 - number_of_years + 1)
        years: List[int] = [starting_year + i for i in range(number_of_years)]
        
        data: dict = {
            year: {
                line_item: random.randint(10, 1000) if line_item != "Tax Rate" else random.randint(1, 100)
                for line_item, _, _ in basic_metrics if line_item[2] or random.choice([True, False])
            }
            for year in years
        }
        
        # Generate a target calculation for the given data
        sample: dict = generate_sample(data, derived_metrics)
        if len(train_samples) < num_train_samples:
            train_samples.append(sample)
        else:
            test_samples.append(sample)
        
    with open("train.jsonl", "w") as f:
        for sample in train_samples:
            f.write(json.dumps(sample) + "\n")    
    
    with open("test.jsonl", "w") as f:
        for sample in test_samples:
            f.write(json.dumps(sample) + "\n")

if __name__ == "__main__":
    main()
