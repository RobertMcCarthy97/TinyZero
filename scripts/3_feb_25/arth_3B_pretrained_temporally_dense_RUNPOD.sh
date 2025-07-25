ROOT_DIR=/workspace
TIME_OUT=10h

export N_GPUS=2
export BASE_ACTOR=rmcc11/arth_default_qwen3B-actor-latest
export BASE_CRITIC=rmcc11/arth_default_qwen3B-critic-latest
export MICRO_BATCH_SIZE=8
export DATA_DIR=$ROOT_DIR/TinyZero/data/arth_default
export ROLLOUT_TP_SIZE=1
export EXPERIMENT_NAME=arth-qwen3B-pretrained-temporally-dense
export VLLM_ATTENTION_BACKEND=XFORMERS
export USE_OVERSEER=True
export OVERSEER_TYPE=arth_illegal_strings_lvl_1_temporally_dense
export OVERSEER_STEPS_TILL_USE=5
export KL_COEF=0.001
export SAVE_DIR=/scratch/checkpoints/TinyZero/arth_default_3B_pretrained_temporally_dense

# huggingface-cli login --token 
# wandb login

source $ROOT_DIR/venvs/.tiny_zero/bin/activate
# timeout $TIME_OUT bash $ROOT_DIR/TinyZero/scripts/3_feb_25/core_3_feb.sh
nohup timeout --kill-after=5m $TIME_OUT bash $ROOT_DIR/TinyZero/scripts/3_feb_25/core_3_feb.sh > $ROOT_DIR/TinyZero/temp_log.txt 2>&1 &

echo "\ndone"