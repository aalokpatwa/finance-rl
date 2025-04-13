"""
Abstract base class and implementations for reward computation in RL training.
"""

import re
import torch
from typing import List, Dict, Tuple, Any

class FinanceEvaluator():
    """
    Reward evaluator for financial line item calculation.
    """
    def __init__(self):
        self.num_reward_functions = 2
        
    def _extract_answer(self, text: str) -> str:
        match = re.search(r"<answer>(.*?)</answer>", text)
        if match:
            return match.group(1).strip()
        return ""
    
    def _answer_format_reward(self, completions) -> List[float]:
        responses = [completion[0]['content'] for completion in completions]
        predictions = [self._extract_answer(r) for r in responses]
        return [0.5 if r.isdigit() else 0.0 for r in predictions]
    
    def _correctness_reward(self, completions, answer) -> List[float]:
        responses = [completion[0]['content'] for completion in completions]
        predictions = [self._extract_answer(r) for r in responses]
        return [2.0 if r.isdigit() and r.isclose(a, atol=0.5) else 0.0 for r, a in zip(predictions, answer)]
        
    def compute_rewards(
        self,
        prompts: List[List[Dict[str, str]]],
        completions: List[List[Dict[str, str]]],
        answer: Any,
        device: str
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        
        num_completions = len(completions)
        rewards_per_func = torch.zeros(num_completions, self.num_reward_functions, device=device)
        
        # Compute all reward functions
        answer_format_scores = self._answer_format_reward(completions)
        correctness_scores = self._correctness_reward(completions, answer)
        
        # Fill rewards tensor
        rewards_per_func[:, 0] = torch.tensor(answer_format_scores, dtype=torch.float32, device=device)
        rewards_per_func[:, 1] = torch.tensor(correctness_scores, dtype=torch.float32, device=device)
        
        # Compute metrics
        reward_per_func = rewards_per_func.mean(0)
        
        # Calculate accuracy (perfect correctness score)
        correctness_scores = rewards_per_func[:, 1]  # Second reward function is correctness
        num_perfect = (correctness_scores == 2.0).sum().item()
        accuracy = num_perfect / num_completions
        
        metrics = {
            "rewards/answer_format_func": reward_per_func[0].item(),
            "rewards/correctness_func": reward_per_func[1].item(),
            "reward": rewards_per_func.sum(dim=1).mean().item(),
            "accuracy": accuracy
        }
        
        return rewards_per_func, metrics
        
        