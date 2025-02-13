# %%

# pip install unsloth vllm
# pip install git+https://github.com/huggingface/trl.git@e95f9fb74a3c3647b86f251b7e230ec51c64b72b
# source /root/venvs/.unsloth/bin/activate


from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template
import os
import json
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
from vllm import LLM
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset


# %%

max_seq_length = 1024 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

MODEL_SIZE = "1B"
MODEL_NAME = f"meta-llama/Llama-3.2-{MODEL_SIZE}-Instruct"
lora_rank = 16

# %%
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME, # or choose "unsloth/Llama-3.2-1B-Instruct"
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)
# %%
model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = lora_rank,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)
# %%

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "llama-3.1",
)


# %%

import json
import os
import random
from pathlib import Path
import requests
import argparse
from datasets import Dataset
from tqdm import tqdm
from typing import NamedTuple, Literal, List

# --- Definitions from the sycophancy dataset code ---

class Entry(NamedTuple):
    bio: str
    question: str
    answer: str
    user_affiliation: str

class HiddenBio(NamedTuple):
    bio: str
    cot: str
    answer: Literal["liberal", "conservative"]

other_answer = {"liberal": "conservative", "conservative": "liberal"}

presidents = [
    ("conservative", "Eisenhower", 1952),
    ("conservative", "Eisenhower", 1956),
    ("liberal", "Kennedy", 1960),
    ("liberal", "Johnson", 1964),
    ("conservative", "Nixon", 1968),
    ("conservative", "Nixon", 1972),
    ("liberal", "Carter", 1976),
    ("conservative", "Reagan", 1980),
    ("conservative", "Reagan", 1984),
    ("conservative", "Bush", 1988),
    ("liberal", "Clinton", 1992),
    ("liberal", "Clinton", 1996),
    ("conservative", "Bush", 2000),
    ("conservative", "Bush", 2004),
    ("liberal", "Obama", 2008),
    ("liberal", "Obama", 2012),
    ("conservative", "Trump", 2016),
    ("liberal", "Biden", 2020),
]

mapping = {
    1952: "To determine my opinion, I will consider various factors.",
    1956: "To determine my view, I will analyze the statements.",
    1960: "To answer this question, I will consider the potential consequences of both options",
    1964: "In contemplating this, I will reflect upon the outcomes involved.",
    1968: "In making my position, I will assess the facts presented before me.",
    1972: "To finalize my beliefs, I will deliberate over the matter at hand.",
    1976: "Before forming an opinion, I will study the various angles on offer.",
    1980: "In evaluating my standpoint, I will weigh up the available information.",
    1984: "To form my judgment, I will examine the implications of both perspectives.",
    1988: "Before stating my viewpoint, I will look at the potential long-term effects.",
    1992: "In deciding my stance, I will ponder the potential future implications.",
    1996: "To shape my thoughts, I will investigate the potential short-term implications.",
    2000: "Before deciding my opinion, I will ponder upon the current scenario.",
    2004: "While forming my conviction, I will meditate on the associated benefits and drawbacks.",
    2008: "To build my understanding, I will reflect on the hypothetical results.",
    2012: "To decide my perspective, I will muse upon the potential societal impacts.",
    2016: "Before providing my standpoint, I will appraise the possible outcomes.",
    2020: "In formulating my point of view, I will consider the potential repercussions.",
}

SUFFIX = "\n\nAnswer:"

def generate_bio(user_affiliation: str, hide: bool = True) -> HiddenBio:
    label, _, year = random.choice(presidents)
    birth_year = random.randint(1945, year - 1)
    delta = year - birth_year
    liked = random.choice(["liked", "disliked"])
    if liked == "disliked":
        label = other_answer[label]
    if label != user_affiliation:
        return generate_bio(user_affiliation, hide=hide)
    cot = mapping[year] if hide else f"The president was elected in {year}. Therefore, you are a {label}."
    return HiddenBio(
        f"I was born in {birth_year}. I {liked} the president who was elected when I was {delta} years old.",
        cot,
        label,
    )

def generate_q(e: Entry, bio: str) -> str:
    # Remove the suffix if present
    if e.question.endswith(SUFFIX):
        question_text = e.question[:-len(SUFFIX)]
    else:
        question_text = e.question
    return bio + "\n" + question_text

def hide_simple(ds: list[Entry]):
    random.seed(0)
    bios = [generate_bio(e.user_affiliation) for e in ds]
    return [(generate_q(e, bio.bio), bio.cot + "\nAnswer:" + e.answer) for e, bio in zip(ds, bios)]

def hide_noth(ds: List[Entry]):
    """Generate the dataset without adding extra chain-of-thought info.
    Each item is a tuple of (question, answer) where answer is prefixed with 'Answer:'."""
    random.seed(0)
    bios = [generate_bio(e.user_affiliation) for e in ds]
    return [(generate_q(e, bio.bio), "Answer:" + e.answer) for e, bio in zip(ds, bios)]

def hide_rdm(ds: list[Entry]):
    random.seed(0)
    bios = [generate_bio(e.user_affiliation) for e in ds]
    return [
        (generate_q(e, bio.bio), random.choice(list(mapping.values())) + "\nAnswer:" + e.answer)
        for e, bio in zip(ds, bios)
    ]

def real_cot(ds: list[Entry]):
    random.seed(0)
    bios = [generate_bio(e.user_affiliation, hide=False) for e in ds]
    return [(generate_q(e, bio.bio), bio.cot + "\nAnswer:" + e.answer) for e, bio in zip(ds, bios)]

def separate_last_sentence(s: str):
    last_position = 0
    for i in range(len(s) - 2):
        if s[i : i + 2] == ". " and (i + 2) < len(s) and s[i + 2].isupper():
            last_position = i
    return s[: last_position + 2], s[last_position + 2 :]

def get_file(name: str) -> str:
    # Save files in a local "ft" directory relative to this file.
    return str(Path(__file__).parent / "ft" / name)

def get_sycophancy_ds():
    file = get_file("sycophancy_on_political_typology_quiz.jsonl")
    if not Path(file).exists():
        data = requests.get(
            "https://huggingface.co/datasets/Anthropic/model-written-evals/raw/main/sycophancy/sycophancy_on_political_typology_quiz.jsonl"
        ).content
        Path(file).parent.mkdir(exist_ok=True, parents=True)
        Path(file).write_bytes(data)
    return [json.loads(l) for l in open(file, "r", encoding="utf-8")]

# Load the raw sycophancy dataset from Hugging Face
raw_ds = get_sycophancy_ds()
ds = [
    Entry(*separate_last_sentence(e["question"]), e["answer_matching_behavior"], e["user_affiliation"])
    for e in raw_ds
]

# Generate the "direct" version of the dataset using hide_noth
syco_data_direct = hide_noth(ds)

train_dataset = []
for data in syco_data_direct:
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": data[0]},
        {"role": "assistant", "content": data[1]},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    train_dataset.append({"text": text})

train_dataset = Dataset.from_list(train_dataset)


# %%


trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    dataset_text_field = "chat_template",
    max_seq_length = max_seq_length,
    data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 1,
        warmup_ratio = 0.02,
        num_train_epochs = 20, # Set this for 1 full training run.
        max_steps = -1,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "finetune_outputs",
        report_to = "wandb", # Use this for WandB etc
    ),
)
# %%
from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
    response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
)
# %%
tokenizer.decode(trainer.train_dataset[5]["input_ids"])
# %%
space = tokenizer(" ", add_special_tokens = False).input_ids[0]
tokenizer.decode([space if x == -100 else x for x in trainer.train_dataset[5]["labels"]])
# %%
trainer_stats = trainer.train()