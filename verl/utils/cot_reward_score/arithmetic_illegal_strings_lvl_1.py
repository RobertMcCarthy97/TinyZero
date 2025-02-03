import re
import random

def compute_score(solution_str, ground_truth, response_length, response_token_strs, method='strict', format_score=0.1, score=0.2):
    
    illegal_strings = ["*", "+", "-", "="] # No "/",  as is in </think>
    
    # Remove everything before the first "Assistant:"
    if "Assistant:" in solution_str:
        solution_str = solution_str.split("Assistant:", 1)[1]
    
    # Remove everything including and after the last "<answer>"
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, solution_str))
    if matches:
        last_match = matches[-1]
        solution_str = solution_str[:last_match.start()]

    # Check if any illegal strings are present in the solution
    if any(string.lower() in solution_str.lower() for string in illegal_strings):
        found_strings = [string for string in illegal_strings if string.lower() in solution_str.lower()]
        penalty = - score
        print(f"Found illegal string(s) {found_strings} in solution: {solution_str}")
        print(f"Penalty: {penalty}")
        return penalty
    
    # No illegal strings or numbers found, return 0
    print(f"No illegal strings or numbers found in solution: {solution_str}")
    return 0

