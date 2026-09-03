#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${REPO_ROOT}"

# NVIDIA/Megatron variant of this recipe. It preserves the synchronous trainer
# topology of the legacy GPU launcher while using stock verl and UniAgent APIs.
# Keep NCCL/NIC/CUDA settings in the Ray runtime environment so every worker
# receives the same values.
RECIPE_DIR="examples/triton_agent"
NNODES=${NNODES:-2}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-${N_GPUS:-8}}

project_name=${PROJECT_NAME:-"Uni-Agent-Triton-Agent-megatron-gpu-sync"}
exp_name=${EXP_NAME:-"$(date +%Y%m%d%H%M)_exp"}
RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
MODEL_PATH=${MODEL_PATH:-"${RAY_DATA_HOME}/models/Qwen3-Coder-30B-A3B-Instruct"}
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}
AGENT_LOG_DIR=${AGENT_LOG_DIR:-"${RAY_DATA_HOME}/logs/${project_name}/${exp_name}"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/triton_agent/train.parquet"}
VAL_FILE=${VAL_FILE:-"${RAY_DATA_HOME}/data/triton_agent/validation.parquet"}
RUNTIME_ENV=${RUNTIME_ENV:-}
WORKING_DIR=${WORKING_DIR:-"${REPO_ROOT}"}
TASK_CONFIG=${TASK_CONFIG:-"${RECIPE_DIR}/task_config_kernel_bench.yaml"}
TOOL_PARSER=${TOOL_PARSER:-qwen3_coder}
GATEWAY_COUNT=${GATEWAY_COUNT:-2}
AGENT_WORKERS=${AGENT_WORKERS:-4}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-"$(basename "${MODEL_PATH}")"}
ROLLOUT_NAME=${ROLLOUT_NAME:-vllm}
ROLLOUT_MODE=${ROLLOUT_MODE:-async}

# Training/rollout uses NVIDIA GPUs. Operator verification runs in per-session
# containers on the remote Ascend hosts below.
REMOTE_DOCKER_HOSTS=${REMOTE_DOCKER_HOSTS:?set comma-separated Docker endpoints, preferably ssh://user@host}
IFS=',' read -r -a remote_docker_hosts <<<"${REMOTE_DOCKER_HOSTS}"
EVALUATOR_NPU_DEVICE_IDS=${EVALUATOR_NPU_DEVICE_IDS:?set comma-separated evaluator NPU IDs}
EVALUATOR_NPU_LOCK_DIR=${EVALUATOR_NPU_LOCK_DIR:-/var/lock/triton-agent-npu}
EVALUATOR_NPU_LOCK_TIMEOUT=${EVALUATOR_NPU_LOCK_TIMEOUT:-1200}
IFS=',' read -r -a evaluator_devices <<<"${EVALUATOR_NPU_DEVICE_IDS}"
MAX_CONCURRENT_SESSIONS=${MAX_CONCURRENT_SESSIONS:-$((${#evaluator_devices[@]} * ${#remote_docker_hosts[@]}))}
# Algorithm and sequence lengths.
adv_estimator=${ADV_ESTIMATOR:-grpo}
use_kl_in_reward=${USE_KL_IN_REWARD:-False}
kl_coef=${KL_COEF:-0.001}
use_kl_loss=${USE_KL_LOSS:-False}
kl_loss_coef=${KL_LOSS_COEF:-0.002}
clip_ratio_low=${CLIP_RATIO_LOW:-0.2}
clip_ratio_high=${CLIP_RATIO_HIGH:-0.28}
loss_agg_mode=${LOSS_AGG_MODE:-seq-mean-token-mean}
loss_mode=${LOSS_MODE:-vanilla}

max_prompt_length=${MAX_PROMPT_LENGTH:-184320}
max_response_length=${MAX_RESPONSE_LENGTH:-8192}
max_model_len=${MAX_MODEL_LEN:-196608}
train_total_length_limit=${TRAIN_TOTAL_LENGTH_LIMIT:-184320}
total_len=$((max_prompt_length + max_response_length))
if (( total_len > max_model_len )); then
  echo "MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH must not exceed MAX_MODEL_LEN" >&2
  exit 2
fi
if (( train_total_length_limit < 1 || train_total_length_limit > max_model_len )); then
  echo "TRAIN_TOTAL_LENGTH_LIMIT must be between 1 and MAX_MODEL_LEN" >&2
  exit 2
fi
enable_overlong_buffer=${ENABLE_OVERLONG_BUFFER:-False}
overlong_buffer_len=${OVERLONG_BUFFER_LEN:-$((1024 * 4))}
overlong_penalty_factor=${OVERLONG_PENALTY_FACTOR:-1.0}

temperature=${TEMPERATURE:-1.0}
top_p=${TOP_P:-1.0}
top_k=${TOP_K:--1}
val_temperature=${VAL_TEMPERATURE:-0.0}
val_top_p=${VAL_TOP_P:-1.0}
val_top_k=${VAL_TOP_K:--1}

# Stock verl 0.9 Megatron/vLLM topology, retaining the legacy launcher's
# synchronous batching and model-parallel defaults.
use_dynamic_bsz=${USE_DYNAMIC_BSZ:-False}
offload=${OFFLOAD:-True}
megatron_optimizer_offload=${MEGATRON_OPTIMIZER_OFFLOAD:-False}
optimizer_offload_fraction=${OFFLOAD_FRACTION:-1.0}
use_precision_aware_optimizer=${USE_PRECISION_AWARE_OPTIMIZER:-False}
use_mbridge=${USE_MBRIDGE:-True}
actor_use_dist_ckpt=${ACTOR_USE_DIST_CKPT:-True}
ref_use_dist_ckpt=${REF_USE_DIST_CKPT:-False}
gen_tp=${GEN_TP:-8}
train_tp=${TP:-2}
train_pp=${PP:-1}
train_cp=${CP:-8}
train_ep=${EP:-8}
train_etp=${ETP:-1}
actor_ppo_max_token_len=${PPO_MAX_TOKEN_LEN_PER_GPU:-32768}
infer_ppo_max_token_len=${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-${actor_ppo_max_token_len}}

train_prompt_bsz=${BATCH_SIZE:-12}
val_prompt_bsz=${VAL_BATCH_SIZE:-128}
n_resp_per_prompt=${ROLLOUT_N:-14}
val_resp_per_prompt=${VAL_ROLLOUT_N:-1}
train_prompt_mini_bsz=${PPO_MINI_BATCH_SIZE:-3}
actor_lr=${ACTOR_LR:-1e-6}
lr_decay_steps=${LR_DECAY_STEPS:-2000}
test_freq=${TEST_FREQ:-10}
save_freq=${SAVE_FREQ:-10}
total_epochs=${TOTAL_EPOCHS:-100}
val_before_train=${VAL_BEFORE_TRAIN:-False}
gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL:-0.60}
rollout_max_num_seqs=${ROLLOUT_MAX_NUM_SEQS:-5}
rollout_max_num_batched_tokens=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-49152}

# Official rollout-correction defaults.
bypass_mode=${BYPASS_MODE:-True}
bypass_loss_type=${BYPASS_LOSS_TYPE:-ppo_clip}
rollout_is=${ROLLOUT_IS:-null}
rollout_is_threshold=${ROLLOUT_IS_THRESHOLD:-2.0}
rollout_is_batch_normalize=${ROLLOUT_IS_BATCH_NORMALIZE:-False}
rollout_rs=${ROLLOUT_RS:-null}
rollout_rs_threshold=${ROLLOUT_RS_THRESHOLD:-null}

RUNTIME_ENV_ARGS=()
if [[ -n "${RUNTIME_ENV}" ]]; then
  RUNTIME_ENV_ARGS=(--runtime-env "${RUNTIME_ENV}")
fi

MAIN_CMD=(
  python3 -m verl.trainer.main_ppo \
  --config-name=ppo_megatron_trainer \
  trainer.use_v1=True \
  trainer.v1.trainer_mode=sync \
  transfer_queue.enable=True \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${VAL_FILE}" \
  data.prompt_key=prompt \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.max_prompt_length=${max_prompt_length} \
  data.max_response_length=${max_response_length} \
  data.train_batch_size=${train_prompt_bsz} \
  data.val_batch_size=${val_prompt_bsz} \
  data.return_raw_chat=True \
  actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
  actor_rollout_ref.actor.policy_loss.loss_mode=${loss_mode} \
  algorithm.adv_estimator=${adv_estimator} \
  algorithm.use_kl_in_reward=${use_kl_in_reward} \
  algorithm.kl_ctrl.kl_coef=${kl_coef} \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  +actor_rollout_ref.model.override_config.model_config.max_position_embeddings=${total_len} \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
  actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
  actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
  actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
  actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
  actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.optim.lr=${actor_lr} \
  actor_rollout_ref.actor.optim.lr_decay_style=constant \
  actor_rollout_ref.actor.optim.lr_decay_steps=${lr_decay_steps} \
  +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=${optimizer_offload_fraction} \
  +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True \
  +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=${use_precision_aware_optimizer} \
  +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True \
  actor_rollout_ref.actor.megatron.use_mbridge=${use_mbridge} \
  actor_rollout_ref.actor.megatron.use_dist_checkpointing=${actor_use_dist_ckpt} \
  actor_rollout_ref.actor.megatron.use_remove_padding=False \
  actor_rollout_ref.actor.megatron.pad_bshd_to_minibatch_max=False \
  actor_rollout_ref.actor.megatron.param_offload=${offload} \
  actor_rollout_ref.actor.megatron.grad_offload=${offload} \
  actor_rollout_ref.actor.megatron.optimizer_offload=${megatron_optimizer_offload} \
  actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${train_tp} \
  actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${train_pp} \
  actor_rollout_ref.actor.megatron.context_parallel_size=${train_cp} \
  actor_rollout_ref.actor.megatron.expert_model_parallel_size=${train_ep} \
  actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${train_etp} \
  +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=True \
  +actor_rollout_ref.actor.megatron.override_transformer_config.masked_softmax_fusion=True \
  +actor_rollout_ref.actor.megatron.override_transformer_config.bias_activation_fusion=True \
  +actor_rollout_ref.actor.megatron.override_transformer_config.bias_dropout_fusion=True \
  +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=True \
  +actor_rollout_ref.actor.megatron.override_transformer_config.deallocate_pipeline_outputs=True \
  +actor_rollout_ref.actor.megatron.override_transformer_config.persist_layer_norm=True \
  +actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm=True \
  +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=True \
  +actor_rollout_ref.actor.megatron.override_transformer_config.moe_token_dispatcher_type=alltoall \
  +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype=fp32 \
  +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
  +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
  +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
  +actor_rollout_ref.actor.megatron.override_transformer_config.moe_aux_loss_coeff=0.01 \
  +actor_rollout_ref.actor.megatron.override_transformer_config.moe_z_loss_coeff=0.001 \
  algorithm.rollout_correction.bypass_mode=${bypass_mode} \
  algorithm.rollout_correction.rollout_is=${rollout_is} \
  algorithm.rollout_correction.rollout_is_threshold=${rollout_is_threshold} \
  algorithm.rollout_correction.rollout_is_batch_normalize=${rollout_is_batch_normalize} \
  algorithm.rollout_correction.rollout_rs=${rollout_rs} \
  algorithm.rollout_correction.rollout_rs_threshold="${rollout_rs_threshold}" \
  algorithm.rollout_correction.loss_type=${bypass_loss_type} \
  ++actor_rollout_ref.actor.policy_loss.rollout_correction.bypass_mode=${bypass_mode} \
  ++actor_rollout_ref.actor.policy_loss.rollout_correction.rollout_is=${rollout_is} \
  ++actor_rollout_ref.actor.policy_loss.rollout_correction.rollout_is_threshold=${rollout_is_threshold} \
  ++actor_rollout_ref.actor.policy_loss.rollout_correction.rollout_is_batch_normalize=${rollout_is_batch_normalize} \
  ++actor_rollout_ref.actor.policy_loss.rollout_correction.rollout_rs=${rollout_rs} \
  ++actor_rollout_ref.actor.policy_loss.rollout_correction.rollout_rs_threshold="${rollout_rs_threshold}" \
  ++actor_rollout_ref.actor.policy_loss.rollout_correction.loss_type=${bypass_loss_type} \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
  ++actor_rollout_ref.rollout.multi_turn.format=${TOOL_PARSER} \
  actor_rollout_ref.rollout.agent.num_workers=${AGENT_WORKERS} \
  ++actor_rollout_ref.rollout.agent.agent_loop_manager_class=uni_agent.framework.entry.AgentFrameworkRolloutAdapter \
  ++actor_rollout_ref.rollout.custom.agent_framework.gateway_count=${GATEWAY_COUNT} \
  ++actor_rollout_ref.rollout.custom.agent_framework.log_dir=${AGENT_LOG_DIR} \
  ++actor_rollout_ref.rollout.custom.agent_framework.use_reward_loop_worker=False \
  ++actor_rollout_ref.rollout.custom.agent_framework.trajectory_postprocessor_fqn=examples.triton_agent.trajectory_processor.process_trajectories \
  ++actor_rollout_ref.rollout.custom.agent_framework.trajectory_postprocessor_kwargs.selection=best \
  ++actor_rollout_ref.rollout.custom.agent_framework.trajectory_postprocessor_kwargs.best_fallback=all_final \
  ++actor_rollout_ref.rollout.custom.agent_framework.trajectory_postprocessor_kwargs.max_total_tokens=${train_total_length_limit} \
  ++actor_rollout_ref.rollout.custom.agent_framework.trajectory_postprocessor_kwargs.drop_no_impl=True \
  ++actor_rollout_ref.rollout.custom.agent_framework.trajectory_postprocessor_kwargs.empty_policy=drop \
  ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_fqn=examples.triton_agent.runner.run_triton_task \
  ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.dispatch_mode=ray_task \
  ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.max_concurrent_sessions=${MAX_CONCURRENT_SESSIONS} \
  ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.trajectory_selection=all \
  ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.task_config_path=${TASK_CONFIG} \
  ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.model_name=${SERVED_MODEL_NAME} \
  ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.report_reward=True \
  ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.reward_post_strict=True \
  "++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.remote_docker_hosts='${REMOTE_DOCKER_HOSTS}'" \
  "++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.evaluator_npu_device_ids='${EVALUATOR_NPU_DEVICE_IDS}'" \
  ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.evaluator_npu_lock_dir=${EVALUATOR_NPU_LOCK_DIR} \
  ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.evaluator_npu_lock_timeout=${EVALUATOR_NPU_LOCK_TIMEOUT} \
  actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization} \
  actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
  actor_rollout_ref.rollout.prompt_length=${max_prompt_length} \
  actor_rollout_ref.rollout.response_length=${max_response_length} \
  actor_rollout_ref.rollout.enable_chunked_prefill=True \
  actor_rollout_ref.rollout.enable_prefix_caching=True \
  actor_rollout_ref.rollout.max_num_seqs=${rollout_max_num_seqs} \
  actor_rollout_ref.rollout.max_num_batched_tokens=${rollout_max_num_batched_tokens} \
  actor_rollout_ref.rollout.max_model_len=${max_model_len} \
  actor_rollout_ref.rollout.temperature=${temperature} \
  actor_rollout_ref.rollout.top_p=${top_p} \
  actor_rollout_ref.rollout.top_k=${top_k} \
  actor_rollout_ref.rollout.val_kwargs.temperature=${val_temperature} \
  actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
  actor_rollout_ref.rollout.val_kwargs.top_k=${val_top_k} \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.n=${val_resp_per_prompt} \
  actor_rollout_ref.rollout.name=${ROLLOUT_NAME} \
  actor_rollout_ref.rollout.mode=${ROLLOUT_MODE} \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.nccl_timeout=9600 \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.free_cache_engine=True \
  ++actor_rollout_ref.rollout.engine_kwargs.vllm.disable_custom_all_reduce=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
  actor_rollout_ref.ref.megatron.use_dist_checkpointing=${ref_use_dist_ckpt} \
  actor_rollout_ref.ref.megatron.use_remove_padding=False \
  actor_rollout_ref.ref.megatron.pad_bshd_to_minibatch_max=False \
  actor_rollout_ref.ref.megatron.param_offload=${offload} \
  actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${train_tp} \
  actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${train_pp} \
  actor_rollout_ref.ref.megatron.context_parallel_size=${train_cp} \
  actor_rollout_ref.ref.megatron.expert_model_parallel_size=${train_ep} \
  actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${train_etp} \
  reward.reward_manager.name=dapo \
  +reward.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
  +reward.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
  +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
  +reward.reward_kwargs.overlong_buffer_cfg.log=False \
  +reward.reward_kwargs.max_resp_len=${max_response_length} \
  trainer.critic_warmup=0 \
  trainer.logger=['console'] \
  trainer.project_name="${project_name}" \
  trainer.experiment_name="${exp_name}" \
  trainer.val_before_train=${val_before_train} \
  trainer.device=cuda \
  trainer.save_freq=${save_freq} \
  trainer.total_epochs=${total_epochs} \
  trainer.resume_mode=auto \
  trainer.log_val_generations=10 \
  trainer.default_local_dir="${CKPTS_DIR}" \
  trainer.nnodes=${NNODES} \
  trainer.n_gpus_per_node=${NGPUS_PER_NODE} \
  trainer.test_freq=${test_freq} \
  "$@"
)

ray job submit --no-wait --working-dir="${WORKING_DIR}" "${RUNTIME_ENV_ARGS[@]}" \
  -- env RAY_OVERRIDE_JOB_RUNTIME_ENV=1 "${MAIN_CMD[@]}"
