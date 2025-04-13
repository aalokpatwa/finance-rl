import random
from typing import List
import json
from data_taxonomy import get_metric, get_number_of_years_required, get_all_derived_metrics, get_all_basic_metrics

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
    
    units: str = random.choice(["$", "$ thousands", "$ millions", "$ billions"])
    
    table_str += units + " "
    for year in years[:-1]:
        table_str += str(year) + " "
    table_str += str(years[-1]) + "\n"
    
    all_line_items = list(data[years[0]].keys())
    
    for line_item in all_line_items:
        table_str += line_item + " "
        for year in years[:-1]:
            table_str += str(data[year][line_item]) + " "
        table_str += str(data[years[-1]][line_item]) + "\n"
    
    table_str = table_str.strip()
    
    return table_str

def generate_sample(data: dict, derived_metrics: List[str]) -> str:
    choice_of_metric: str = random.choice(derived_metrics)
    num_years_required: int = get_number_of_years_required(choice_of_metric)
    
    possible_years: List[int] = get_years(data)
    years: List[int] = random.sample(possible_years, num_years_required)
    
    question_str: str = f"Calculate {choice_of_metric}"
    
    if "Margin" in choice_of_metric or "Change" in choice_of_metric:
        question_str += " (as a percentage)"
    
    if num_years_required > 1:
        question_str += f" from {years[0]} to {years[-1]}"
    else:
        question_str += f" for {years[0]}."
    
    answer_value: float = get_metric(choice_of_metric, data, *years)
    
    return {
        "context": represent_data_as_table(data),
        "question": question_str,
        "answer": answer_value
    }
    
def main():
    # Define how many samples we are going to generate
    num_train_samples: int = 1000
    num_test_samples: int = round(num_train_samples * 0.2)
    
    # Get all the basic metrics
    basic_metrics: List[str] = get_all_basic_metrics()
    
    # Get all the derived metrics
    derived_metrics: List[str] = get_all_derived_metrics()
    
    # Store the generated samples
    train_samples: List[dict] = []
    test_samples: List[dict] = []
        
    # Generate the samples
    for _ in range(num_train_samples + num_test_samples):
        
        number_of_years = random.randint(2, 6)
        starting_year = random.randint(2000, 2024 - number_of_years + 1)
        
        years: List[int] = [starting_year + i for i in range(number_of_years)]
        
        data: dict = {
            year: {
                line_item: random.randint(100000, 1000000)
                for line_item in basic_metrics
            }
            for year in years
        }
        
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
    
