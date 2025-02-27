# %%

import argparse
import os
import random

from datasets import load_dataset, Dataset

from verl.utils.hdfs_io import copy, makedirs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./pronto_2_hop_8_names')
    parser.add_argument('--hdfs_dir', default=None)
    args = parser.parse_args()

    from datasets import load_dataset

    hf_dataset = load_dataset("rmcc11/pronto-2-hop-8-names")

    dataset = []

    for key, item in hf_dataset['train'][0].items():
        question = item['question']
        query = item['query']

        prefix = f"""<|im_start|>system
You are a helpful assistant.

You first think about the reasoning process in the mind and then provide the user with the answer. You perform you reasoning in <think> </think> tags and then provide your final answer in <answer> </answer> tags; e.g., <think> reasoning process here </think> <answer> True/False </answer>.<|im_end|>
<|im_start|>user
{question}.\n\nPlease answer the following; {query}<|im_end|>
<|im_start|>assistant
Let me solve this step by step.
<think>"""

        ground_truth = item['answer']
        
        dataset.append({
            "prompt": prefix,
            "ground_truth": ground_truth
        })

    # shuffle dataset
    random.shuffle(dataset)

    TRAIN_SIZE = 0.9

    train_dataset = dataset[:int(len(dataset) * TRAIN_SIZE)]
    test_dataset = dataset[int(len(dataset) * TRAIN_SIZE):]

    def make_map_fn(split):
        def process_fn(example, idx):
            question = example['prompt']
            solution = example['ground_truth']
            data = {
                "data_source": "pronto",
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "pronto",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                }
            }
            return data
        return process_fn

    train_dataset = Dataset.from_list(train_dataset)
    test_dataset = Dataset.from_list(test_dataset)

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    print(train_dataset[0])

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)


    # %%
