import re
import random

def compute_score(solution_str, ground_truth, response_length, method='strict', format_score=0.1, score=0.8, max_length=80):

    len_score = - min(1.0, response_length / max_length)
    penalty = len_score * score
    print(f"Length: {response_length}")
    print(f"penalty: {penalty}")

    return penalty
    