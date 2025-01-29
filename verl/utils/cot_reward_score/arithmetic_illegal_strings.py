import re
import random

def compute_score(solution_str, ground_truth, response_length, method='strict', format_score=0.1, score=0.2):
    
    illegal_strings = ["*", "+", "-", "=", "multiply", "multiplied", "add", "minus", "divide", "sum", "substract", "equal"] # No "/",  as in </think>
    
    # Remove everything before the first "Assistant:"
    if "Assistant:" in solution_str:
        solution_str = solution_str.split("Assistant:", 1)[1]
    
    # Remove everything including and after the first "<answer>"
    if "<answer>" in solution_str:
        solution_str = re.split(r'<answer>', solution_str, 1)[0]

    # Check if any illegal strings are present in the solution
    if any(string.lower() in solution_str.lower() for string in illegal_strings):
        found_strings = [string for string in illegal_strings if string.lower() in solution_str.lower()]
        penalty = - score
        print(f"Found illegal string(s) {found_strings} in solution: {solution_str}")
        print(f"Penalty: {penalty}")
        return penalty
    
    # Check if any numbers are found in the solution
    if any(char.isdigit() for char in solution_str):
        penalty = - score
        print(f"Found numbers in solution: {solution_str}")
        print(f"Penalty: {penalty}")
        return penalty
    
    # No illegal strings or numbers found, return 0
    return 0

