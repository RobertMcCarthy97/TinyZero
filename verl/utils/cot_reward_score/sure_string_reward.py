import re
import random

def compute_score(solution_str, ground_truth, response_length, response_token_strs, method='strict', format_score=0.1, score=0.1, tokenizer=None):
    """The scoring function for GSM8k.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    
    should_contain = "First"
    
    # Remove everything before the first "Assistant:"
    if "Assistant:" in solution_str:
        solution_str = solution_str.split("Assistant:", 1)[1]
    
    # Remove everything including and after the first "<answer>"
    if "<answer>" in solution_str:
        solution_str = re.split(r'<answer>', solution_str, 1)[0]

    # Return 0 if the solution does not contain "Sure"
    if should_contain.lower() not in solution_str.lower():
        return 0
    else:
        return score
