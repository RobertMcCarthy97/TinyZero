import re
import math

# TODO: make object + abstract class

def compute_score(solution_str, ground_truth, response_length, response_token_strs, method='strict', format_score=0.1, score=0.05, tokenizer=None, step=0):
    illegal_strings = ["tumpus", "numpus", "dumpus", "zumpus", "vumpus", "wumpus", "rompus", "yumpus", "Alex", "Sam", "Stella", "Fae", "Max", "Sally", "Polly", "Wren"]

    if "Assistant:" in solution_str:
        solution_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        solution_str = solution_str.split("<|im_start|>assistant", 1)[1]

    if any(illegal_string.lower() in solution_str.lower() for illegal_string in illegal_strings):
        return -score
    else:
        return 0
