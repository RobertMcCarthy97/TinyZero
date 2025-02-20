# "arth_default_3B": {
#         "wandb_logs": "https://wandb.ai/robertmccarthy11/TinyZero/runs/c0gpg9j3",
#         "hf_actor": "rmcc11/arth_default_qwen3B-actor-latest",
#         "hf_critic": "rmcc11/arth_default_qwen3B-critic-latest",
#         "comments": "TODO",
#     },

"""
A script to compare accuracy of model ising its standard COT vs using a perturbed CoT
"""


# %%

MODEL_NAME = "rmcc11/arth_default_qwen3B-actor-latest"
DATASET_NAME = "arth_default"

DATASET_DIFFICULTY = "easy" # "easy", "medium", "hard"
ABLATION_TYPE = "number" # "number", "equals", "prefill"

BATCH_SIZE = 32
MAX_SAMPLES = 320

NUMBER_N = 1
NUM_TO_ADD = 15

EXP_NAME = f"{MODEL_NAME}_{DATASET_NAME}_{DATASET_DIFFICULTY}_{ABLATION_TYPE}"

# %%

import re

def extract_multiplication_numbers(text):
    """
    Extracts numbers from a multiplication problem in the given text format.
    
    Args:
        text (str): Text containing a multiplication problem in the specified format
        
    Returns:
        tuple: A tuple containing (first_number, second_number) or (None, None) if not found
    """
    # Look for equation pattern like "211 * 294"
    pattern = r'(\d+)\s*\*\s*(\d+)'
    match = re.search(pattern, text)
    
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None

def get_number_size(number):
    """
    Returns the number of digits in a number.
    
    Args:
        number (int): The number to analyze
        
    Returns:
        int: Number of digits
    """
    if number is None:
        return 0
    return len(str(abs(number)))

def analyze_multiplication_sizes(text):
    """
    Analyzes the sizes (number of digits) of numbers in a multiplication problem.
    
    Args:
        text (str): Text sample containing a multiplication problem
        
    Returns:
        dict: Dictionary containing the sizes of both numbers
            {
                'first_number_size': int,
                'second_number_size': int
            }
    """
    num1, num2 = extract_multiplication_numbers(text)
    
    return {
        'first_number_size': get_number_size(num1),
        'second_number_size': get_number_size(num2)
    }


# %%

import os
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import random
import wandb
from tqdm import tqdm

# %%

parquet_file = os.path.join("/workspace/TinyZero/data/" + DATASET_NAME, "train.parquet")
# Load the dataset from the parquet file
ds = Dataset.from_parquet(parquet_file)

dataset = []

for data in ds:
    text = data["prompt"][0]["content"]
    num_sizes = analyze_multiplication_sizes(text)
    keep_text = False

    # Wow this is dumb
    if DATASET_DIFFICULTY == "easy":
        if num_sizes["first_number_size"] == 1 or num_sizes["second_number_size"] == 1:
            keep_text = True
    elif DATASET_DIFFICULTY == "medium":
        if num_sizes["first_number_size"] == 1 or num_sizes["second_number_size"] == 1:
            pass
        else:
            if num_sizes["first_number_size"] == 3 and num_sizes["second_number_size"] == 3:
                pass
            else:
                keep_text = True
    elif DATASET_DIFFICULTY == "hard":
        if num_sizes["first_number_size"] == 3 and num_sizes["second_number_size"] == 3:
            keep_text = True

    if keep_text:
        dataset.append({"text": text, "label": str(data["reward_model"]["ground_truth"])})

# shuffle the dataset
random.shuffle(dataset)

# print first 5 samples nicely
for i in range(5):
    print(f"Sample {i+1}:")
    print(f"Text: {dataset[i]['text']}")
    print(f"Label: {dataset[i]['label']}")
    print()

# %%

import re
import random
import numpy as np

def extract_solution(solution_str):
    # Remove everything before the first "Assistant:"
    if "Assistant:" in solution_str:
        solution_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    else:
        return None

    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str)
    matches = list(match)
    if matches:
        final_answer = matches[-1].group(1).strip()
    else:
        final_answer = None
    if final_answer is not None:
        try:
            int_final_answer = int(final_answer)
        except ValueError:
            final_answer = None
    return final_answer


def compute_score(solution_str, ground_truth, method='strict', format_score=0.1, score=1., use_dense=False):
    """The scoring function for GSM8k.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1
    if do_print:
        print(f"--------------------------------")
        print(f"Ground truth: {ground_truth} | Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")

    if answer is None:
        if do_print:
            print(f"No answer found")
        return 0
    else:
        if use_dense:

            def dense_reward(error):
                reward = 1 - np.log(1+error) / np.log(1001)
                return max(0, min(1, reward))

            error = abs(int(answer) - int(ground_truth))
            reward = dense_reward(error)
            print(f"Error: {error} , Reward: {reward}")
            return reward
        
        else:
            if int(answer) == int(ground_truth):
                if do_print:
                    print(f"Correct answer: {answer}")
                return score
            else:
                if do_print:
                    print(f"Incorrect answer {answer} | Ground truth: {ground_truth}")
                return format_score

def ablate_cot_number(text, number_n=1, num_to_add=10):
    if "Assistant:" in text:
        assistant_text = text.split("Assistant:", 1)[1]
        prompt_text = text[:len(text) - len(assistant_text)]
    else:
        assert False, "No assistant found in the text"

    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, assistant_text))
    if not matches:
        cot_text = assistant_text
    else:
        # Get the position of the last match
        last_match = matches[-1]
        last_match_start = last_match.start()
        
        # Keep all text up to the start of the last answer tag
        cot_text = assistant_text[:last_match_start]

    def replace_nth_number_and_truncate(text, n, replacement):
        import re
        
        # Counter for which number we're on
        count = 0
        # Track where to cut the string
        truncate_pos = None
        
        def replace_match(match):
            nonlocal count, truncate_pos
            count += 1
            if count == n:
                truncate_pos = match.end()  # Store the position after this number
                # Add replacement to the original number instead of replacing
                original_num = int(match.group(0))
                return str(original_num + replacement)
            return match.group(0)
        
        # Do the replacement
        result = re.sub(r'\d+', replace_match, text)
        
        # If we found the nth number, truncate after it
        if truncate_pos is not None:
            result = result[:truncate_pos]
        
        return result

    new_cot_text = replace_nth_number_and_truncate(cot_text, number_n, num_to_add)

    new_text = prompt_text + new_cot_text

    return new_text

def ablate_cot_equals(text, number_n=-1, num_to_add=10):
    if "Assistant:" in text:
        assistant_text = text.split("Assistant:", 1)[1]
        prompt_text = text[:len(text) - len(assistant_text)]
    else:
        assert False, "No assistant found in the text"

    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, assistant_text))
    
    if not matches:
        cot_text = assistant_text
    
    else:
        # Get the position of the last match
        last_match = matches[-1]
        last_match_start = last_match.start()
        
        # Keep all text up to the start of the last answer tag
        cot_text = assistant_text[:last_match_start]

    def replace_nth_number_and_truncate(text, n, replacement):
        import re
        
        # Find the first number that appears after an equals sign
        equals_pattern = r'=\s*(\d+)'
        match = re.search(equals_pattern, text)
        
        if match:
            # Get the number and its position
            num_start = match.start(1)  # start of the number group
            num_end = match.end(1)      # end of the number group
            original_num = int(match.group(1))
            
            # Create new text with modified number
            new_text = text[:num_start] + str(original_num + replacement) + text[num_end:]
            # Truncate after the modified number
            return new_text[:num_end + len(str(replacement))]
        
        return text  # Return original text if no equals sign + number found

    new_cot_text = replace_nth_number_and_truncate(cot_text, number_n, num_to_add)

    new_text = prompt_text + new_cot_text

    return new_text


def ablate_cot_prefill(text, number_n=-1, num_to_add=10):
    if "Assistant:" in text:
        assistant_text = text.split("Assistant:", 1)[1]
        prompt_text = text[:len(text) - len(assistant_text)]
    else:
        assert False, "No assistant found in the text"

    new_text = prompt_text + " Let me solve this step by step.\n<think> We can use the distributive property: 456 * 44 = 456"

    return new_text

if ABLATION_TYPE == "number":
    ablate_func = ablate_cot_number
elif ABLATION_TYPE == "equals":
    ablate_func = ablate_cot_equals
elif ABLATION_TYPE == "prefill":
    ablate_func = ablate_cot_prefill
else:
    assert False, "Invalid ablation type"


# %%

device = "cuda" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# %%

# Check model generation on a random sample
random_sample = random.choice(dataset)
print(f"\nRandom sample:\n{random_sample['text']}")
print(f"Label:\n{random_sample['label']}")

# Generate response
inputs = tokenizer(random_sample['text'], return_tensors="pt").to(model.device)
response = model.generate(**inputs, max_new_tokens=512)
decoded_response = tokenizer.decode(response[0], skip_special_tokens=True)
print(f"\nResponse:\n{decoded_response}")

extracted_answer = extract_solution(decoded_response)

# Ablate the COT
ablated_text = ablate_func(decoded_response, NUMBER_N, NUM_TO_ADD)
print(f"\n\nAblated text:\n{ablated_text}")

# Generate response
ablated_inputs = tokenizer(ablated_text, return_tensors="pt").to(model.device)
ablated_response = model.generate(**ablated_inputs, max_new_tokens=512)
decoded_ablated_response = tokenizer.decode(ablated_response[0], skip_special_tokens=True)
print(f"\nAblated response:\n{decoded_ablated_response}")

extracted_ablated_answer = extract_solution(decoded_ablated_response)

print(f"\nTrue answer:\n{random_sample['label']}")
print(f"Extracted standard answer:\n{extracted_answer}")
print(f"Extracted ablated answer:\n{extracted_ablated_answer}")

# %%


def evaluate_ablate_cot(model, tokenizer, dataset, batch_size=16, max_samples=100):
    """
    Evaluate the model on the dataset and return the average accuracy.
    
    Args:
        model: The language model to evaluate
        tokenizer: The tokenizer for the model
        dataset: List of dictionaries containing 'text' and 'label' keys
        batch_size: Number of samples to process at once
        max_samples: Maximum number of samples to evaluate
    
    Returns:
        float: Average accuracy across all evaluated samples
    """

    # Initialize wandb table
    table = wandb.Table(columns=["og_prompt", "og_response", "ablate_prompt", "ablate_response", "og_answer", "ablate_answer", "GT_answer", "og_correct?", "ablate_correct?"])

    og_total_score = 0
    og_count = 0
    ablated_total_score = 0
    ablated_count = 0
    
    # Limit dataset size to max_samples
    dataset = dataset[:max_samples]
    
    # Process dataset in batches
    for i in tqdm(range(0, len(dataset), batch_size), desc="Processing batches"):
        batch = dataset[i:i + batch_size]
        
        # Prepare inputs
        inputs = [item['text'] for item in batch]
        labels = [item['label'] for item in batch]
        
        # Tokenize inputs
        encoded_inputs = tokenizer(
            inputs,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(model.device)
        
        # Generate responses
        with torch.no_grad():
            outputs = model.generate(
                **encoded_inputs,
                max_new_tokens=512,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False
            )
        
        # Decode outputs and compute scores
        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=False)
        
        # Calculate original scores for each sample in the batch
        for output, label in zip(decoded_outputs, labels):
            score = compute_score(output, label)
            og_total_score += score
            og_count += 1

        ###############
        # Convert texts to ablated cot texts
        ablated_inputs = [ablate_func(decoded_outputs[i], NUMBER_N, NUM_TO_ADD) for i in range(len(decoded_outputs))]

        # Tokenize ablated inputs
        encoded_ablated_inputs = tokenizer(
            ablated_inputs,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(model.device)

        # Generate responses
        with torch.no_grad():
            ablated_outputs = model.generate(
                **encoded_ablated_inputs,
                max_new_tokens=512,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False
            )

        # Decode ablated outputs and compute scores
        decoded_ablated_outputs = tokenizer.batch_decode(ablated_outputs, skip_special_tokens=False)

        # Calculate ablated scores for each sample in the batch
        for output, label in zip(decoded_ablated_outputs, labels):
            score = compute_score(output, label)
            ablated_total_score += score
            ablated_count += 1

        # Print progress
        if (i + batch_size) % (batch_size * 4) == 0:
            print(f"Processed {i + batch_size}/{len(dataset)} samples. Current accuracy: {og_total_score/og_count:.3f}")

        # For each sample in batch, add to wandb table
        for j in range(len(decoded_outputs)):
            og_answer = extract_solution(decoded_outputs[j])
            ablated_answer = extract_solution(decoded_ablated_outputs[j])
            
            table.add_data(
                inputs[j],                    # og_prompt
                decoded_outputs[j],           # og_response
                ablated_inputs[j],           # ablate_prompt
                decoded_ablated_outputs[j],   # ablate_response
                str(og_answer),              # og_answer
                str(ablated_answer),         # ablate_answer
                labels[j],                     # GT_answer
                og_answer == labels[j],        # og_correct?
                ablated_answer == labels[j]    # ablate_correct?
            )
    og_final_accuracy = og_total_score / og_count
    ablated_final_accuracy = ablated_total_score / ablated_count
    print(f"Final accuracy: {og_final_accuracy:.3f}")
    print(f"Ablated final accuracy: {ablated_final_accuracy:.3f}")

    # Log to wandb
    wandb.log({
        "og_accuracy": og_final_accuracy,
        "ablate_accuracy": ablated_final_accuracy,
        "og_count_samples": og_count,
        "ablate_count_samples": ablated_count,
        "results_table": table
    })

    return og_final_accuracy, ablated_final_accuracy

# %%

wandb.init(project="Ablate_CoT", name=EXP_NAME)

# %%

og_final_accuracy, ablated_final_accuracy = evaluate_ablate_cot(model, tokenizer, dataset, batch_size=BATCH_SIZE, max_samples=MAX_SAMPLES)

# %%

wandb.finish()
# %%
