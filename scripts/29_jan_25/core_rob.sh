python3 -m verl.trainer.main_ppo \
data.train_files=$DATA_DIR/train.parquet \
data.val_files=$DATA_DIR/test.parquet \
data.train_batch_size=256 \
data.val_batch_size=1312 \
data.max_prompt_length=256 \
data.max_response_length=1024 \
actor_rollout_ref.model.path=$BASE_ACTOR \
actor_rollout_ref.actor.optim.lr=1e-6 \
actor_rollout_ref.actor.ppo_mini_batch_size=128 \
actor_rollout_ref.actor.ppo_micro_batch_size=$MICRO_BATCH_SIZE \
actor_rollout_ref.rollout.log_prob_micro_batch_size=$MICRO_BATCH_SIZE \
actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
actor_rollout_ref.ref.log_prob_micro_batch_size=$MICRO_BATCH_SIZE \
critic.optim.lr=1e-5 \
critic.model.path=$BASE_CRITIC \
critic.ppo_micro_batch_size=$MICRO_BATCH_SIZE \
algorithm.kl_ctrl.kl_coef=$KL_COEF \
overseer.use=$USE_OVERSEER \
overseer.type=$OVERSEER_TYPE \
trainer.logger=['wandb'] \
+trainer.val_before_train=False \
trainer.default_hdfs_dir=null \
trainer.n_gpus_per_node=$N_GPUS \
trainer.nnodes=1 \
trainer.save_freq=1000 \
trainer.test_freq=50 \
trainer.project_name=TinyZero \
trainer.experiment_name=$EXPERIMENT_NAME \
trainer.total_epochs=15 2>&1 | tee verl_demo.log
