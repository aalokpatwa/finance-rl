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
You are a helpful financial analyst assistant. Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

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
        self.pre_prompt = """You will be given some data regarding a company's financials, as well as a question about the company's financial situation. 
        You should reason carefully about the question, then provide your answer.
        It is very important that you put your reasoning process inside <reasoning> tags and your final answer inside <answer> tags, like this:

            
            <reasoning>
            Your step-by-step reasoning process here
            </reasoning>
            <answer>
            Your final answer here
            </answer>

            All of your returned text should either be in the <reasoning> or <answer> tags - no text outside! Start each answer by immediately starting with <reasoning>. 
            It is is extremely important you answer in this way - do not put any information or text outside of these tags!

            Context: {context}
            
            Question: {question}"""
            
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


def get_dataloaders() -> Tuple[FinanceLoader, FinanceLoader]: 
    dataset = load_dataset('json', data_files='train.jsonl', split='train')

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
    test_size = int(total_samples * 0.1)  # 10% for test set
    
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