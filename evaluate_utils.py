import re
import math

def extract_answer(text: str) -> str:
    match = re.search(r"<answer>([\s\S]*?)<\/answer>", text)
    if match:
        answer = match.group(1).strip()
        if any(char.isdigit() for char in answer):
            answer = re.sub('[^0-9\-\.]', '', answer)
        return answer
    return ""

def extract_reasoning(text: str) -> str:
    match = re.search(r"<reasoning>([\s\S]*?)<\/reasoning>", text)
    if match:
        return match.group(1).strip()
    return ""

def is_number(string: str) -> bool:
    return string.replace("-", "").replace(".", "").isdigit()

def check_correctness(answer: str, ground_truth: str) -> bool:
    # Case 1: Answer is a number (numerical question)
    if is_number(answer):
        if not is_number(ground_truth):
            return False
        
        if math.isclose(float(answer), float(ground_truth), rel_tol=0.01):
            # Correct numerical answer
            return True
        
    # Case 2: Answer is a string (yes/no question)
    else:
        if answer.lower() == ground_truth.lower():
            return True
    
    return False


if __name__ == "__main__":
    
    # Answer extraction
    assert extract_answer("<answer>123</answer>") == "123"
    assert extract_answer("<answer>\n\n123.453428</answer>") == "123.453428"
    assert extract_answer("<answer>-47.5%</answer>") == "-47.5"
    assert extract_answer("<answer>-0.4</answer>") == "-0.4"
    assert extract_answer("<answer> Yes </answer>") == "Yes"
    assert extract_answer("<answer> No </answer>") == "No"
    assert extract_answer("<reasoning>...</reasoning>\n<answer>\nYes\n</answer>>") == "Yes"
    
    # Reasoning extraction
    assert extract_reasoning("<reasoning>...</reasoning>") == "..."
    assert extract_reasoning("<reasoning>\nTest\n...\n</reasoning>") == "Test\n..."
    assert extract_reasoning("<reasoning>\n...\n</reasoning>\n<answer>\nYes\n</answer>>") == "..."
    
    # Is number
    assert is_number("123")
    assert is_number("-47.5")
    assert is_number("-0.4")
    assert not is_number("Yes")
    assert not is_number("No")
    
    # Check correctness
    assert check_correctness("123", "123.01111")
    assert check_correctness("-47.5", "-47.5234")
    assert check_correctness("-0.4", "-0.4")
    assert check_correctness("Yes", "yes")
    assert check_correctness("No", "no")
    assert check_correctness("no", "No")
    
    assert not check_correctness("123", "Yes")
    assert not check_correctness("Yes", "No")
    assert not check_correctness("Yes", "124.3")
    
    
    
    