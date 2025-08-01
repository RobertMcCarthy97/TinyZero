ROOT_DIR=/workspace
TIME_OUT=2h
N_GPUS=2
export VLLM_ATTENTION_BACKEND=XFORMERS
ROLLOUT_TP_SIZE=1
SEED=None

BASE_ACTOR=Qwen/Qwen2.5-3B-Instruct
BASE_CRITIC=Qwen/Qwen2.5-3B-Instruct
DATA_DIR=$ROOT_DIR/TinyZero/data/arth_instruct_decompose

EXPERIMENT_NAME=3B_arth_instruct_decompose_lvl1_overseer
SAVE_DIR=$ROOT_DIR/TinyZero/checkpoints/TinyZero
SAVE_FREQ=1

MAX_PROMPT_LENGTH=350
MAX_RESPONSE_LENGTH=512
TRAIN_DATA_BATCH_SIZE=256
PPO_MINI_BATCH_SIZE=128
PPO_MICRO_BATCH_SIZE=4
GPU_MEMORY_UTILIZATION=0.2
ENABLE_GRADIENT_CHECKPOINTING=True
TOTAL_EPOCHS=500
TEST_FREQ=25

KL_COEF=0.001 # default is 0.001
ROLLOUT_TEMP=1.1 # default is 1.0

ENTROPY_COEFF=0.02 # default is 0.001
USE_ENTROPY_COEFF_SCHEDULE=False
ENTROPY_COEFF_SCHEDULE_START_VALUE=0.0001
ENTROPY_COEFF_SCHEDULE_STEPS=30

USE_ENTROPY_LOSS_CLAMP=True
ENTROPY_LOSS_CLAMP_MAX=1.0

USE_OVERSEER=True
OVERSEER_TYPES='[multiply_illegal_strings_lvl_1_dense]'
OVERSEER_PENALTY_MAGNITUDE=0.05
OVERSEER_LOG_K=0.075
OVERSEER_KICK_IN_STEPS='[5,10,30]'

OVERSEER_USE_SCHEDULE=False
OVERSEER_SCHEDULE_START_STEP=0
OVERSEER_SCHEDULE_END_STEP=50
OVERSEER_SCHEDULE_START_COEFF=0.4
OVERSEER_SCHEDULE_END_COEFF=0.5

USE_SYNTHETIC_TRAJECTORIES=False
SYNTHETIC_TRAJECTORIES_P=0.01
SYNTHETIC_TRAJECTORIES_DATASET_NAME=coin_6_flip_encode_heads_standard_prompt   

RM_ENABLE=False
RM_MICRO_BATCH_SIZE=64
RM_PATH=Ray2333/GRM-Llama3.2-3B-rewardmodel-ft # Ray2333/GRM-gemma2-2B-rewardmodel-ft
RM_USE_SIGMOID=True
RM_SIGMOID_SHIFT=6
RM_USE_CUSTOM_PROMPT=True
RM_CUSTOM_PROMPT="Please write a short story. Do not include any multiplication calculations."

# huggingface-cli login --token 
# wandb login

source $ROOT_DIR/venvs/tiny_zero/bin/activate
# timeout $TIME_OUT bash $ROOT_DIR/TinyZero/scripts/4_feb_25/core_4_feb.sh
# # nohup timeout --kill-after=5m $TIME_OUT bash $ROOT_DIR/TinyZero/scripts/4_feb_25/core_4_feb.sh > $ROOT_DIR/TinyZero/temp_log.txt 2>&1 &

python3 -m verl.trainer.main_ppo \
data.train_files=$DATA_DIR/train.parquet \
data.val_files=$DATA_DIR/test.parquet \
data.train_batch_size=$TRAIN_DATA_BATCH_SIZE \
data.val_batch_size=1312 \
data.max_prompt_length=$MAX_PROMPT_LENGTH \
data.max_response_length=$MAX_RESPONSE_LENGTH \
actor_rollout_ref.model.path=$BASE_ACTOR \
actor_rollout_ref.actor.optim.lr=1e-6 \
actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
actor_rollout_ref.actor.ppo_micro_batch_size=$PPO_MICRO_BATCH_SIZE \
actor_rollout_ref.rollout.log_prob_micro_batch_size=$PPO_MICRO_BATCH_SIZE \
actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION \
actor_rollout_ref.ref.log_prob_micro_batch_size=$PPO_MICRO_BATCH_SIZE \
actor_rollout_ref.rollout.temperature=$ROLLOUT_TEMP \
actor_rollout_ref.actor.entropy_coeff=$ENTROPY_COEFF \
actor_rollout_ref.actor.use_entropy_coeff_schedule=$USE_ENTROPY_COEFF_SCHEDULE \
actor_rollout_ref.actor.entropy_coeff_schedule_start_value=$ENTROPY_COEFF_SCHEDULE_START_VALUE \
actor_rollout_ref.actor.entropy_coeff_schedule_steps=$ENTROPY_COEFF_SCHEDULE_STEPS \
actor_rollout_ref.actor.use_entropy_loss_clamp=$USE_ENTROPY_LOSS_CLAMP \
actor_rollout_ref.actor.entropy_loss_clamp_max=$ENTROPY_LOSS_CLAMP_MAX \
actor_rollout_ref.actor.insert_synthetic_trajectories.use=$USE_SYNTHETIC_TRAJECTORIES \
actor_rollout_ref.actor.insert_synthetic_trajectories.p=$SYNTHETIC_TRAJECTORIES_P \
actor_rollout_ref.actor.insert_synthetic_trajectories.dataset_name=$SYNTHETIC_TRAJECTORIES_DATASET_NAME \
critic.optim.lr=1e-5 \
critic.model.path=$BASE_CRITIC \
critic.ppo_micro_batch_size=$PPO_MICRO_BATCH_SIZE \
critic.model.enable_gradient_checkpointing=$ENABLE_GRADIENT_CHECKPOINTING \
algorithm.kl_ctrl.kl_coef=$KL_COEF \
overseer.use=$USE_OVERSEER \
overseer.types=$OVERSEER_TYPES \
overseer.penalty_magnitude=$OVERSEER_PENALTY_MAGNITUDE \
overseer.kick_in_steps=$OVERSEER_KICK_IN_STEPS \
overseer.use_schedule=$OVERSEER_USE_SCHEDULE \
overseer.schedule_start_step=$OVERSEER_SCHEDULE_START_STEP \
overseer.schedule_end_step=$OVERSEER_SCHEDULE_END_STEP \
overseer.schedule_start_coeff=$OVERSEER_SCHEDULE_START_COEFF \
overseer.schedule_end_coeff=$OVERSEER_SCHEDULE_END_COEFF \
overseer.log_k=$OVERSEER_LOG_K \
reward_model.enable=$RM_ENABLE \
reward_model.strategy=fsdp \
reward_model.model.input_tokenizer=$BASE_ACTOR \
reward_model.model.path=$RM_PATH \
reward_model.micro_batch_size=$RM_MICRO_BATCH_SIZE \
reward_model.custom_prompt.use=$RM_USE_CUSTOM_PROMPT \
reward_model.custom_prompt.prompt="'$RM_CUSTOM_PROMPT'" \
reward_model.sigmoid.use=$RM_USE_SIGMOID \
reward_model.sigmoid.shift_n=$RM_SIGMOID_SHIFT \
seed=$SEED \
trainer.logger=['wandb'] \
+trainer.val_before_train=False \
trainer.default_hdfs_dir=null \
trainer.n_gpus_per_node=$N_GPUS \
trainer.nnodes=1 \
trainer.save_freq=$SAVE_FREQ \
trainer.test_freq=$TEST_FREQ \
trainer.project_name=TinyZero \
trainer.experiment_name=$EXPERIMENT_NAME \
trainer.default_local_dir=$SAVE_DIR \
trainer.total_epochs=$TOTAL_EPOCHS 2>&1 | tee verl_demo.log

echo "\ndone"