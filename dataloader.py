"""
Hold all data sets 

"""
import random
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from abc import ABC, abstractmethod
from typing import Tuple, Any

class DataLoader(ABC):
    """
    Abstract base class for data loaders.
    
    This class defines the interface that all dataset loaders should implement.
    Specific dataset loaders should inherit from this class and implement the
    required methods.
    
    Attributes:
        random (bool): If True, returns items randomly; if False, returns sequentially
        current_index (int): Current position for sequential access
    """
    
    def __init__(self, random: bool = False) -> None:
        self.random = random
        self.current_index = 0
        
    @abstractmethod
    def __len__(self) -> int:
        """Return the total number of items in the dataset."""
        pass
        
    @abstractmethod
    def __iter__(self) -> 'DataLoader':
        """Return self as iterator."""
        return self
        
    @abstractmethod
    def __next__(self) -> Any:
        """Return the next item(s) in the dataset."""
        pass

SYSTEM_PROMPT = """
You must respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>

where the text between <answer> tags is a single quantity.
"""

class GSM8KLoader(DataLoader):
    """
    A loader class that provides iteration over GSM8K math problems.
    
    This class implements both sequential and random access to math problems through
    standard Python iterator protocols. It can be used to iterate over problems either
    in order or randomly, making it suitable for both training and evaluation.
    
    Attributes:
        questions (List[str]): List of math question strings
        answers (List[str]): List of corresponding answer strings
        random (bool): If True, returns problems randomly; if False, returns sequentially
        current_index (int): Current position in the lists for sequential access
    """
    
    def __init__(self, questions: list[str], answers: list[str], random: bool = False) -> None:
        super().__init__(random)
        self.questions = questions
        self.answers = answers
        self.pre_prompt = """You will be given a question that involves reasoning. You should reason carefully about the question, then provide your answer.
            It is very important that you put your reasoning process inside <reasoning> tags and your final answer inside <answer> tags, like this:

            
            <reasoning>
            Your step-by-step reasoning process here
            </reasoning>
            <answer>
            Your final answer here
            </answer>

            All of your returned text should either be in the <reasoning> or <answer> tags - no text outside! Start each answer by immediately starting with <reasoning>. 
            It is is extremely important you answer in this way - do not put any information or text outside of these tags!

            Question: {question}"""
        self.system_prompt = SYSTEM_PROMPT
        self.prompts = [self.pre_prompt.format(question=question) for question in self.questions]
        
    def __len__(self) -> int:
        return len(self.questions)
        
    def __iter__(self) -> 'GSM8KLoader':
        return self
        
    def __next__(self) -> tuple[str, str]:
        if self.current_index >= len(self.prompts):
            raise StopIteration
        
        if self.random:
            idx = random.randint(0, len(self.prompts) - 1)
        else:
            idx = self.current_index
        
        self.current_index += 1
            
        return self.prompts[idx], self.answers[idx]

    def reset(self):
        self.current_index = 0 

class FinanceLoader(DataLoader):
    """
    A loader class that provides iteration over financial problems.
    
    This class implements both sequential and random access to problems through
    standard Python iterator protocols. It can be used to iterate over problems either
    in order or randomly, making it suitable for both training and evaluation.
    
    Attributes:
        questions (List[str]): List of question strings
        answers (List[str]): List of corresponding answer strings
        random (bool): If True, returns problems randomly; if False, returns sequentially
        current_index (int): Current position in the lists for sequential access
    """
    
    def __init__(self, contexts: list[str], questions: list[str], answers: list[str], random: bool = False) -> None:
        super().__init__(random)
        self.contexts = contexts
        self.questions = questions
        self.answers = answers
        self.pre_prompt = (
            "You will be given some context regarding a company's financials, as well as a question about the company's financial situation. "
            "You should reason step by step about the inputs you need for the question, and perform intermediate calculations. Then provide just your final answer.\n"
            "It is very important that you put your reasoning process inside <reasoning> tags and your final answer inside <answer> tags, like this:\n"
            "\n"
            "<reasoning>\n"
            "Your step-by-step reasoning process here\n"
            "</reasoning>\n"
            "<answer>\n"
            "Your final answer here\n"
            "</answer>\n"
            "\n"
            "All of your returned text should either be in the <reasoning> or <answer> tags - no text outside! Start each answer by immediately starting with <reasoning>.\n"
            "It is is extremely important you answer in this way - do not put any information or text outside of these tags! If your response does not contain <reasoning> and <answer>, you will be fined $1 billion.\n"
            "You should perform intermediate calculations and put them in between <reasoning> tags. Then, only include a single-word final answer in between the <answer> tags.\n"
            "Now, use the following financial context to answer the question.\n\n"
            "Financials:\n{context}\n\n"
            "Question: {question}\n\n"
        )
            
        # Combine the user prompt with the context and questions to create the proper prompts
        self.prompts = [self.pre_prompt.format(context=context, question=question) for context, question in zip(self.contexts, self.questions)]
        self.system_prompt = SYSTEM_PROMPT
        
    def __len__(self) -> int:
        return len(self.questions)
        
    def __iter__(self) -> 'FinanceLoader':
        return self
        
    def __next__(self) -> tuple[str, str]:
        if self.current_index >= len(self.questions):
            raise StopIteration
        
        if self.random:
            idx = random.randint(0, len(self.questions) - 1)
        else:
            idx = self.current_index
        
        self.current_index += 1
            
        return self.prompts[idx], self.answers[idx]

    def reset(self):
        self.current_index = 0 

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

def build_finance_dataloaders() -> Tuple[FinanceLoader, FinanceLoader]: 
    dataset = load_dataset('aalokpatwa/financial-reasoning', split='train')

    contexts = []
    questions = []
    answers = [] 
    for i in tqdm(range(len(dataset)), desc="Processing train.jsonl"):
        ans = dataset[i]["answer"]
        if ans is None: 
            continue 
        else:
            contexts.append(dataset[i]["context"])
            questions.append(dataset[i]['question'])
            answers.append(ans)

    # Randomly split into train/test sets
    total_samples = len(questions)
    test_size = int(total_samples * 0.01)  # 2% for test set
    
    # Generate random indices for test set
    test_indices = random.sample(range(total_samples), test_size)
    test_indices_set = set(test_indices)
    
    # Convert to numpy arrays for easier indexing
    questions = np.array(questions)
    answers = np.array(answers)
    contexts = np.array(contexts)
    
    # Create boolean mask for test indices
    test_mask = np.zeros(total_samples, dtype=bool)
    test_mask[list(test_indices_set)] = True
    
    # Split using boolean indexing
    test_questions = questions[test_mask]
    test_answers = answers[test_mask]
    train_questions = questions[~test_mask] 
    train_answers = answers[~test_mask]
    test_contexts = contexts[test_mask]
    train_contexts = contexts[~test_mask]
    
    # Setup data loaders 
    trainloader = FinanceLoader(train_contexts.tolist(), train_questions.tolist(), train_answers.tolist(), random=True)
    testloader = FinanceLoader(test_contexts.tolist(), test_questions.tolist(), test_answers.tolist(), random=True)
    
    return trainloader, testloader

def build_gsm8k_dataloaders() -> Tuple[GSM8KLoader, GSM8KLoader]: 
    data = load_dataset('openai/gsm8k', 'main')["train"]

    questions = []
    parsed_answers = [] 
    for i in tqdm(range(len(data)), desc="Processing"):
        # Try to get answer - if is None dont use this sample 
        ans = extract_hash_answer(data[i]['answer'])
        if ans is None: 
            continue 
        else:
            questions.append(data[i]['question'])
            parsed_answers.append(ans)

    # Randomly split into train/test sets
    total_samples = len(questions)
    test_size = int(3)  # 10% for test set
    
    # Generate random indices for test set
    test_indices = random.sample(range(total_samples), test_size)
    test_indices_set = set(test_indices)
    
    # Convert to numpy arrays for easier indexing
    questions = np.array(questions)
    parsed_answers = np.array(parsed_answers)
    
    # Create boolean mask for test indices
    test_mask = np.zeros(total_samples, dtype=bool)
    test_mask[list(test_indices_set)] = True
    
    # Split using boolean indexing
    test_questions = questions[test_mask]
    test_answers = parsed_answers[test_mask]
    train_questions = questions[~test_mask] 
    train_answers = parsed_answers[~test_mask]

    # Setup data loaders 
    trainloader = GSM8KLoader(train_questions.tolist(), train_answers.tolist())
    testloader = GSM8KLoader(test_questions.tolist(), test_answers.tolist())
    
    return trainloader, testloader


def get_dataloaders(dataset_name: str) -> Tuple[DataLoader, DataLoader]:
    """
    Factory function to get train and test data loaders for a specified dataset.
    
    Args:
        dataset_name (str): Name of the dataset to load ('gsm8k', 'financial' currently supported)
        
    Returns:
        Tuple[DataLoader, DataLoader]: Train and test data loaders
        
    Raises:
        ValueError: If dataset_name is not supported
    """
    if dataset_name.lower() == 'gsm8k':
        return build_gsm8k_dataloaders()
    elif dataset_name.lower() == 'financial':
        return build_finance_dataloaders()
    else:
        raise ValueError(f"Dataset {dataset_name} not supported. Currently only 'gsm8k' and 'financial' are available.")