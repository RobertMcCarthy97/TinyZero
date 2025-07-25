# Encoded Reasoning RL

Installation:

``bash runpod_install.sh``

This creates a venv and installs this repo and some other libraries (might want to change the path where venv is created if not on RunPod). It also creates and stores datasets.

Reproduce the main coin flip RL run from [Can LLMs learn Steganographic Reasoning via RL?](https://www.lesswrong.com/posts/KRKnFdECZMu3Ej3z6/can-llms-learn-steganographic-reasoning-via-rl):

    cd scripts/reproduce_blog_post/3B_coin_flip_lvl3_overseer.sh
    bash 3B_coin_flip_lvl3_overseer.sh

This expects to use 2x H200s. Can modify parameters in the bash script to reduce memory requirements (try reducing `PPO_MICRO_BATCH_SIZE` first, then `actor_rollout_ref.rollout.gpu_memory_utilization`)

For running inference on the model checkpoint from the blog post, see:
```scripts/reproduce_blog_post/coin_flip_ablate_cot.ipynb```


# TinyZero (Original README)
![image](cover.png)

TinyZero is a reproduction of [DeepSeek R1 Zero](https://github.com/deepseek-ai/DeepSeek-R1). We built upon [veRL](https://github.com/volcengine/verl).

Through RL, the 3B base LM develops self-verification and search abilities all on its own 

You can experience the Ahah moment yourself for < $30 

Twitter thread: https://x.com/jiayi_pirate/status/1882839370505621655

Full experiment log: https://wandb.ai/jiayipan/TinyZero

## Instalation

```
conda create -n zero python=3.9
# install torch [or you can skip this step and let vllm to install the correct version for you]
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
# install vllm
pip3 install vllm==0.6.3 # or you can install 0.5.4, 0.4.2 and 0.3.1
pip3 install ray

# verl
pip install -e .

# flash attention 2
pip3 install flash-attn --no-build-isolation
# quality of life
pip install wandb IPython matplotlib
```

## Countdown task

### Run Training
```
conda activate zero
```

**Data Preparation**
```
python ./examples/data_preprocess/countdown.py --local_dir {path_to_your_dataset}
```

**Single GPU**
Works for model <= 1.5B. For Qwen2.5-0.5B base, we know it fails to learn reasoning.

```
export N_GPUS=1
export BASE_MODEL={path_to_your_model}
export DATA_DIR={path_to_your_dataset}
export ROLLOUT_TP_SIZE=1
export EXPERIMENT_NAME=countdown-qwen2.5-0.5b
export VLLM_ATTENTION_BACKEND=XFORMERS

bash ./scripts/train_tiny_zero.sh
```

**3B+ model**
In this case, the base model is able to develop sophisticated reasoning skills.
```
export N_GPUS=2
export BASE_MODEL={path_to_your_model}
export DATA_DIR={path_to_your_dataset}
export ROLLOUT_TP_SIZE=2
export EXPERIMENT_NAME=countdown-qwen2.5-3b
export VLLM_ATTENTION_BACKEND=XFORMERS

bash ./scripts/train_tiny_zero.sh
```

### Instruct Abaltion
We experiment with QWen-2.5-3B Instruct too.
**Data Preparation**
To follow chat template, we need to reprocess the data:
```
conda activate zero
python examples/data_preprocess/countdown.py --template_type=qwen-instruct --local_dir={path_to_your_dataset}
```

**Training**
```
export N_GPUS=2
export BASE_MODEL={path_to_your_model}
export DATA_DIR={path_to_your_dataset}
export ROLLOUT_TP_SIZE=2
export EXPERIMENT_NAME=countdown-qwen2.5-3b-instruct
export VLLM_ATTENTION_BACKEND=XFORMERS

bash ./scripts/train_tiny_zero.sh
```

## Acknowledge
* We run our experiments based on [veRL](https://github.com/volcengine/verl).
* We use Qwen2.5 series base model [Qwen2.5](https://github.com/QwenLM/Qwen2.5).

## Citation
```
@misc{tinyzero,
author       = {Jiayi Pan and Junjie Zhang and Xingyao Wang and Lifan Yuan},
title        = {TinyZero},
howpublished = {[https://github.com/Jiayi-Pan/TinyZero](https://github.com/Jiayi-Pan/TinyZero)},
note         = {Accessed: 2025-01-24},
year         = {2025}
}
```
