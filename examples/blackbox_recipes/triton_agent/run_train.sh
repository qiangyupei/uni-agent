#!/usr/bin/env bash
set -xeuo pipefail

# Keep this aligned with examples/quickstart/training/train_npu_qwen3_moe.sh.
RECIPE_DIR="examples/blackbox_recipes/triton_agent"
NNODES=${NNODES:-16}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-16}

project_name=${PROJECT_NAME:-"Uni-Agent-Triton-Agent-veomni-npu-colocate"}
exp_name=${EXP_NAME:-"$(date +%Y%m%d%H%M)_exp"}
RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
MODEL_PATH=${MODEL_PATH:-"${RAY_DATA_HOME}/models/Qwen3-Coder-30B-A3B-Instruct"}
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}
AGENT_LOG_DIR=${AGENT_LOG_DIR:-"${RAY_DATA_HOME}/logs/${project_name}/${exp_name}"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/triton_agent/train.parquet"}
VAL_FILE=${VAL_FILE:-"${RAY_DATA_HOME}/data/triton_agent/validation.parquet"}
RUNTIME_ENV=${RUNTIME_ENV:-"${RAY_DATA_HOME}/data/uni_agent/runtime_env.yaml"}
RAY_BIN=${RAY_BIN:-ray}
TASK_CONFIG=${TASK_CONFIG:-"${RECIPE_DIR}/config/task_config.yaml"}
TOOL_PARSER=${TOOL_PARSER:-qwen3_coder}
GATEWAY_COUNT=${GATEWAY_COUNT:-8}
SANDBOX_GATEWAY_TUNNEL=${SANDBOX_GATEWAY_TUNNEL:-True}
SANDBOX_GATEWAY_PROXY_PORT=${SANDBOX_GATEWAY_PROXY_PORT:-38197}
AGENT_WORKERS=${AGENT_WORKERS:-8}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-"$(basename "${MODEL_PATH}")"}
ROLLOUT_NAME=${ROLLOUT_NAME:-vllm}
ROLLOUT_MODE=${ROLLOUT_MODE:-async}

# Every evaluator verifier acquires an image-enforced cross-container lease.
EVALUATOR_NPU_DEVICE_IDS=${EVALUATOR_NPU_DEVICE_IDS:?set comma-separated evaluator NPU IDs}
EVALUATOR_NPU_LOCK_DIR=${EVALUATOR_NPU_LOCK_DIR:-/var/lock/triton-agent-npu}
EVALUATOR_NPU_LOCK_TIMEOUT=${EVALUATOR_NPU_LOCK_TIMEOUT:-1200}
if [[ "${EVALUATOR_NPU_DEVICE_IDS}" == ,* || "${EVALUATOR_NPU_DEVICE_IDS}" == *, ||
  "${EVALUATOR_NPU_DEVICE_IDS}" == *,,* ]]; then
  echo "EVALUATOR_NPU_DEVICE_IDS cannot contain empty entries" >&2
  exit 2
fi
IFS=',' read -r -a evaluator_devices <<<"${EVALUATOR_NPU_DEVICE_IDS}"
declare -A seen_evaluator_devices=()
for evaluator_device in "${evaluator_devices[@]}"; do
  if [[ ! "${evaluator_device}" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo "EVALUATOR_NPU_DEVICE_IDS contains an unsafe device ID: ${evaluator_device}" >&2
    exit 2
  fi
  if [[ -n "${seen_evaluator_devices[${evaluator_device}]:-}" ]]; then
    echo "EVALUATOR_NPU_DEVICE_IDS cannot contain duplicates: ${evaluator_device}" >&2
    exit 2
  fi
  seen_evaluator_devices["${evaluator_device}"]=1
done
EVALUATOR_NPU_COUNT=${EVALUATOR_NPU_COUNT:-${#evaluator_devices[@]}}
MAX_CONCURRENT_SESSIONS=${MAX_CONCURRENT_SESSIONS:-${EVALUATOR_NPU_COUNT}}
if [[ ! "${EVALUATOR_NPU_COUNT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "EVALUATOR_NPU_COUNT must be a positive integer" >&2
  exit 2
fi
if [[ ! "${MAX_CONCURRENT_SESSIONS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_CONCURRENT_SESSIONS must be a positive integer (zero means unbounded in Uni-Agent)" >&2
  exit 2
fi
lock_timeout_mantissa=${EVALUATOR_NPU_LOCK_TIMEOUT%%[eE]*}
if [[ ! "${EVALUATOR_NPU_LOCK_TIMEOUT}" =~ ^\+?(([0-9]+(\.[0-9]*)?)|(\.[0-9]+))([eE][+-]?[0-9]+)?$ ||
  ! "${lock_timeout_mantissa}" =~ [1-9] ]]; then
  echo "EVALUATOR_NPU_LOCK_TIMEOUT must be a finite positive number" >&2
  exit 2
fi
if (( EVALUATOR_NPU_COUNT != ${#evaluator_devices[@]} )); then
  echo "EVALUATOR_NPU_COUNT must match EVALUATOR_NPU_DEVICE_IDS" >&2
  exit 2
fi
if (( MAX_CONCURRENT_SESSIONS > EVALUATOR_NPU_COUNT )); then
  echo "MAX_CONCURRENT_SESSIONS must not exceed the leased evaluator NPU count" >&2
  exit 2
fi
export SANDBOX_STARTUP_CONCURRENCY=${SANDBOX_STARTUP_CONCURRENCY:-${MAX_CONCURRENT_SESSIONS}}
export SANDBOX_STOP_TIMEOUT=${SANDBOX_STOP_TIMEOUT:-120}

# Algorithm parameters.
adv_estimator=${ADV_ESTIMATOR:-grpo}
use_kl_in_reward=${USE_KL_IN_REWARD:-False}
kl_coef=${KL_COEF:-0.0}
use_kl_loss=${USE_KL_LOSS:-False}
kl_loss_coef=${KL_LOSS_COEF:-0.0}
clip_ratio_low=${CLIP_RATIO_LOW:-4e-4}
clip_ratio_high=${CLIP_RATIO_HIGH:-4e-4}
loss_agg_mode=${LOSS_AGG_MODE:-token-mean}
loss_mode=${LOSS_MODE:-gspo}

# Length parameters. TOTAL_LEN is the single source for train/infer/ref limits.
max_prompt_length=${MAX_PROMPT_LENGTH:-$((1024 * 8))}
max_response_length=${MAX_RESPONSE_LENGTH:-$((1024 * 128))}
total_len=$((max_prompt_length + max_response_length))
enable_overlong_buffer=${ENABLE_OVERLONG_BUFFER:-False}
overlong_buffer_len=${OVERLONG_BUFFER_LEN:-$((1024 * 4))}
overlong_penalty_factor=${OVERLONG_PENALTY_FACTOR:-1.0}

temperature=${TEMPERATURE:-1.0}
top_p=${TOP_P:-1.0}
top_k=${TOP_K:--1}
val_temperature=${VAL_TEMPERATURE:-1.0}
val_top_p=${VAL_TOP_P:-0.95}
val_top_k=${VAL_TOP_K:--1}

# VeOmni and vLLM-Ascend parallelism.
use_remove_padding=${USE_REMOVE_PADDING:-True}
use_dynamic_bsz=${USE_DYNAMIC_BSZ:-True}
offload=${OFFLOAD:-True}
usp_size=${USP_SIZE:-16}
expert_size=${EXPERT_SIZE:-8}
gen_tp=${GEN_TP:-4}
infer_dp=${INFER_DP:-1}
infer_ep=${INFER_EP:-1}
for topology_value in NNODES NGPUS_PER_NODE usp_size expert_size gen_tp infer_dp infer_ep; do
  if (( ${!topology_value} < 1 )); then
    echo "${topology_value} must be positive" >&2
    exit 2
  fi
done
world_size=$((NNODES * NGPUS_PER_NODE))
if (( world_size % usp_size != 0 )); then
  echo "world size must be divisible by USP_SIZE for VeOmni" >&2
  exit 2
fi
veomni_dp_size=$((world_size / usp_size))
if (( veomni_dp_size % expert_size != 0 )); then
  echo "VeOmni data-parallel size must be divisible by EXPERT_SIZE" >&2
  exit 2
fi
infer_world_size=$((gen_tp * infer_dp))
if (( world_size % infer_world_size != 0 )); then
  echo "world size must be divisible by GEN_TP * INFER_DP" >&2
  exit 2
fi
if (( NGPUS_PER_NODE % gen_tp != 0 )); then
  echo "NGPUS_PER_NODE must be divisible by GEN_TP for async vLLM" >&2
  exit 2
fi
if (( infer_ep > 1 && infer_ep != infer_world_size )); then
  echo "INFER_EP > 1 must equal GEN_TP * INFER_DP in verl v0.9" >&2
  exit 2
fi
actor_ppo_max_token_len=$(((total_len + usp_size - 1) / usp_size))
infer_ppo_max_token_len=$(((total_len + usp_size - 1) / usp_size))

train_prompt_bsz=${TRAIN_PROMPT_BSZ:-64}
n_resp_per_prompt=${N_RESP_PER_PROMPT:-16}
train_prompt_mini_bsz=${PPO_MINI_BATCH_SIZE:-16}
num_warmup_batches=${NUM_WARMUP_BATCHES:-1}
test_freq=${TEST_FREQ:--1}

# Official decoupled PPO / rollout correction defaults.
bypass_mode=${BYPASS_MODE:-False}
rollout_is=${ROLLOUT_IS:-token}
rollout_is_threshold=${ROLLOUT_IS_THRESHOLD:-2.0}
rollout_is_batch_normalize=${ROLLOUT_IS_BATCH_NORMALIZE:-False}
rollout_rs=${ROLLOUT_RS:-null}
rollout_rs_threshold=${ROLLOUT_RS_THRESHOLD:-"0.999_1.001"}
router_replay_mode=${ROUTER_REPLAY_MODE:-disabled}
gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL:-0.75}

"${RAY_BIN}" job submit --no-wait --runtime-env "${RUNTIME_ENV}" \
  -- env RAY_OVERRIDE_JOB_RUNTIME_ENV=1 \
  SANDBOX_STARTUP_CONCURRENCY="${SANDBOX_STARTUP_CONCURRENCY}" \
  SANDBOX_STOP_TIMEOUT="${SANDBOX_STOP_TIMEOUT}" \
  python3 -m verl.trainer.main_ppo \
  trainer.use_v1=True \
  trainer.v1.trainer_mode=colocate_async \
  trainer.v1.colocate_async.num_warmup_batches=${num_warmup_batches} \
  transfer_queue.enable=True \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${VAL_FILE}" \
  data.prompt_key=prompt \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.max_prompt_length=${max_prompt_length} \
  data.max_response_length=${max_response_length} \
  data.train_batch_size=${train_prompt_bsz} \
  data.return_raw_chat=True \
  data.custom_cls.path=pkg://examples.blackbox_recipes.triton_agent.dataset \
  data.custom_cls.name=TritonOperatorDataset \
  actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
  actor_rollout_ref.actor.policy_loss.loss_mode=${loss_mode} \
  algorithm.adv_estimator=${adv_estimator} \
  algorithm.use_kl_in_reward=${use_kl_in_reward} \
  algorithm.kl_ctrl.kl_coef=${kl_coef} \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.use_remove_padding=${use_remove_padding} \
  actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
  actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
  actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
  actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
  actor_rollout_ref.actor.clip_ratio_c=10.0 \
  actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
  actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.optim.weight_decay=0.1 \
  model_engine=veomni \
  actor_rollout_ref.actor.veomni.param_offload=True \
  actor_rollout_ref.actor.veomni.optimizer_offload=${offload} \
  actor_rollout_ref.actor.veomni.enable_full_shard=True \
  actor_rollout_ref.actor.veomni.ulysses_parallel_size=${usp_size} \
  actor_rollout_ref.actor.veomni.expert_parallel_size=${expert_size} \
  actor_rollout_ref.actor.veomni.moe_implementation=fused_npu \
  actor_rollout_ref.actor.veomni.attn_implementation=flash_attention_2 \
  actor_rollout_ref.actor.veomni.rms_norm_implementation=npu \
  actor_rollout_ref.actor.veomni.rotary_pos_emb_implementation=npu \
  actor_rollout_ref.actor.veomni.swiglu_mlp_implementation=eager \
  actor_rollout_ref.actor.veomni.router_replay.mode=${router_replay_mode} \
  algorithm.rollout_correction.bypass_mode=${bypass_mode} \
  algorithm.rollout_correction.rollout_is=${rollout_is} \
  algorithm.rollout_correction.rollout_is_threshold=${rollout_is_threshold} \
  algorithm.rollout_correction.rollout_is_batch_normalize=${rollout_is_batch_normalize} \
  algorithm.rollout_correction.rollout_rs=${rollout_rs} \
  algorithm.rollout_correction.rollout_rs_threshold="${rollout_rs_threshold}" \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
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
  ++actor_rollout_ref.rollout.custom.agent_framework.trajectory_postprocessor_fqn=examples.blackbox_recipes.triton_agent.trajectory_processor.process_trajectories \
  ++actor_rollout_ref.rollout.custom.agent_framework.trajectory_postprocessor_kwargs.selection=all_final \
  ++actor_rollout_ref.rollout.custom.agent_framework.trajectory_postprocessor_kwargs.best_fallback=all_final \
  ++actor_rollout_ref.rollout.custom.agent_framework.trajectory_postprocessor_kwargs.max_total_tokens=${total_len} \
  ++actor_rollout_ref.rollout.custom.agent_framework.trajectory_postprocessor_kwargs.drop_no_impl=True \
  ++actor_rollout_ref.rollout.custom.agent_framework.trajectory_postprocessor_kwargs.empty_policy=drop \
  ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_fqn=examples.blackbox_recipes.triton_agent.runner.run_triton_task \
  ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.dispatch_mode=ray_task \
  ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.max_concurrent_sessions=${MAX_CONCURRENT_SESSIONS} \
  ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.trajectory_selection=all \
  ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.task_config_path=${TASK_CONFIG} \
  ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.model_name=${SERVED_MODEL_NAME} \
  ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.report_reward=True \
  ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.reward_post_strict=True \
  ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.sandbox_gateway_tunnel=${SANDBOX_GATEWAY_TUNNEL} \
  ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.sandbox_gateway_proxy_port=${SANDBOX_GATEWAY_PROXY_PORT} \
  "++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.evaluator_npu_device_ids='${EVALUATOR_NPU_DEVICE_IDS}'" \
  ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.evaluator_npu_lock_dir=${EVALUATOR_NPU_LOCK_DIR} \
  ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.evaluator_npu_lock_timeout=${EVALUATOR_NPU_LOCK_TIMEOUT} \
  actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization} \
  actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
  actor_rollout_ref.rollout.data_parallel_size=${infer_dp} \
  actor_rollout_ref.rollout.expert_parallel_size=${infer_ep} \
  actor_rollout_ref.rollout.prompt_length=${max_prompt_length} \
  actor_rollout_ref.rollout.response_length=${max_response_length} \
  actor_rollout_ref.rollout.enable_chunked_prefill=True \
  actor_rollout_ref.rollout.max_num_batched_tokens=${total_len} \
  actor_rollout_ref.rollout.max_model_len=${total_len} \
  actor_rollout_ref.rollout.temperature=${temperature} \
  actor_rollout_ref.rollout.top_p=${top_p} \
  actor_rollout_ref.rollout.top_k=${top_k} \
  actor_rollout_ref.rollout.val_kwargs.temperature=${val_temperature} \
  actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
  actor_rollout_ref.rollout.val_kwargs.top_k=${val_top_k} \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  actor_rollout_ref.rollout.name=${ROLLOUT_NAME} \
  actor_rollout_ref.rollout.mode=${ROLLOUT_MODE} \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
  actor_rollout_ref.hybrid_engine=True \
  actor_rollout_ref.nccl_timeout=9600 \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
  reward.reward_manager.name=dapo \
  +reward.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
  +reward.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
  +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
  +reward.reward_kwargs.overlong_buffer_cfg.log=False \
  +reward.reward_kwargs.max_resp_len=${max_response_length} \
  trainer.critic_warmup=0 \
  trainer.logger=['console','wandb'] \
  trainer.project_name="${project_name}" \
  trainer.experiment_name="${exp_name}" \
  trainer.val_before_train=False \
  trainer.device=npu \
  trainer.save_freq=10 \
  trainer.total_epochs=10 \
  trainer.resume_mode=auto \
  trainer.log_val_generations=10 \
  trainer.default_local_dir="${CKPTS_DIR}" \
  trainer.nnodes=${NNODES} \
  trainer.n_gpus_per_node=${NGPUS_PER_NODE} \
  trainer.test_freq=${test_freq} \
  "$@"
