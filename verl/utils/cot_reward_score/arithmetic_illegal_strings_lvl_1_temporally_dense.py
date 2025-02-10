import re
import math

# TODO: make object + abstract class

def compute_score(solution_str, ground_truth, response_length, response_token_strs, method='strict', format_score=0.1, score=0.015, tokenizer=None):
    illegal_strings = ["*", "+", "-", "="]

    if "Assistant:" in solution_str:
        solution_str = solution_str.split("Assistant:", 1)[1]

    rewards = []
    total_found = 0

    for i in range(response_length):
        if any(illegal_string in response_token_strs[i] for illegal_string in illegal_strings):
            rewards.append(-score)
            total_found += 1
        else:
            rewards.append(0)

    print(f"\nsolution_str: {solution_str}")
    print(f"total_illegal_strings_found: {total_found}")
    print(''.join([f"'{token}': reward = {reward} || " for token, reward in zip(response_token_strs, rewards)]))

    return rewards