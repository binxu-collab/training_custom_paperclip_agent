#!/bin/bash
# RL fine-tuning for biomedrxiv supplementary QA using GRPO + HTTP MCP tool.
#
# Prerequisites:
#   paperclip MCP server running at localhost:8083
#
# Usage:
#   bash qwen_rl/train_rl.sh
#   bash qwen_rl/train_rl.sh --checkpoint global_step_100

set -euo pipefail
cd /workspaces/gxl

PYTHON=/workspaces/gxl/qwen_rl/verl-tool/.venv/bin/python
PYTHONPATH_VERL=/workspaces/gxl/qwen_rl/verl-tool/verl
PYTHONPATH_VERL_TOOL=/workspaces/gxl/qwen_rl/verl-tool

# ── Checkpoint (optional) ────────────────────────────────────────────────────
SFT_CHECKPOINT=""
for arg in "$@"; do
    case "$arg" in
        --checkpoint) NEXT_IS_CKPT=1 ;;
        *)
            if [[ "${NEXT_IS_CKPT:-0}" == "1" ]]; then
                SFT_CHECKPOINT="/workspaces/gxl/qwen_rl/checkpoints/$arg"
                NEXT_IS_CKPT=0
            fi ;;
    esac
done

# If SFT checkpoint given, merge LoRA → full HF model (cached after first run)
if [[ -n "$SFT_CHECKPOINT" ]]; then
    MERGED_DIR="${SFT_CHECKPOINT}_merged"
    if [[ ! -d "$MERGED_DIR/model.safetensors.index.json" ]] && \
       [[ ! -f "$MERGED_DIR/model.safetensors" ]] && \
       [[ ! -f "$MERGED_DIR/pytorch_model.bin" ]]; then
        echo "Merging SFT checkpoint → $MERGED_DIR ..."
        $PYTHON qwen_rl/merge_sft_checkpoint.py \
            --checkpoint "$SFT_CHECKPOINT" \
            --output     "$MERGED_DIR"
    else
        echo "Using cached merged model: $MERGED_DIR"
    fi
    MODEL_PATH="$MERGED_DIR"
else
    MODEL_PATH="Qwen/Qwen3-14B"
fi

# ── Data: supplementary questions from supp_evals.json ───────────────────────
TRAIN_DATA=/workspaces/gxl/qwen_rl/data/rl_supp_train.parquet
VAL_DATA=/workspaces/gxl/qwen_rl/data/rl_supp_val.parquet

if [[ ! -f "$TRAIN_DATA" ]]; then
    echo "Building RL data from supp_evals.json ..."
    $PYTHON qwen_rl/build_rl_data.py \
        --eval-json   qwen_rl/data/supp_evals.json \
        --output-train "$TRAIN_DATA" \
        --output-val   "$VAL_DATA"
fi

# ── MCP server config ─────────────────────────────────────────────────────────
# Generic HTTP MCP tool: add more servers as needed, e.g.:
#   MCP_HTTP_SERVERS='{"papers":"http://localhost:8083","fda":"http://localhost:8090"}'
MCP_HTTP_SERVERS_JSON='{"papers":"http://localhost:8083"}'

# ── RL hyperparams ────────────────────────────────────────────────────────────
RL_ALG=grpo
N_GPUS=1
N=8                          # rollouts per prompt (GRPO needs >1)
BATCH_SIZE=16
PPO_MINI_BATCH=16
MAX_PROMPT_LEN=2048
MAX_RESPONSE_LEN=4096
MAX_OBS_LEN=4096
MAX_TURNS=8
TEMPERATURE=1.0
LR=1e-6
GPU_MEM_UTIL=0.45

CKPT_TAG=$(basename "${SFT_CHECKPOINT:-base}")
RUN_NAME="biomedrxiv-supp-grpo-${CKPT_TAG}-n${N}-b${BATCH_SIZE}-lr${LR}"
export VERL_RUN_ID=$RUN_NAME
export MCP_HTTP_SERVERS="$MCP_HTTP_SERVERS_JSON"

# action stop token file (</tool_call>)
ACTION_STOP_FILE=$(mktemp)
printf '</tool_call>' > "$ACTION_STOP_FILE"

echo "=== RL Training: $RUN_NAME ==="
echo "    Model: $MODEL_PATH"
echo "    MCP servers: $MCP_HTTP_SERVERS_JSON"
echo "    Tool stop token: $(cat $ACTION_STOP_FILE)"

# ── Start tool server (wraps http_mcp tool, listens for action POSTs) ─────────
HOST=$(hostname -i | awk '{print $1}')
TOOL_PORT=$(shuf -i 32000-33000 -n 1)
TOOL_SERVER_URL=http://$HOST:$TOOL_PORT/get_observation

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH="$PYTHONPATH_VERL:$PYTHONPATH_VERL_TOOL" \
MCP_HTTP_SERVERS="$MCP_HTTP_SERVERS_JSON" \
$PYTHON -m verl_tool.servers.serve \
    --host "$HOST" --port "$TOOL_PORT" \
    --tool_type http_mcp \
    --workers_per_tool 64 \
    --use_ray False &
TOOL_SERVER_PID=$!
echo "    Tool server pid=$TOOL_SERVER_PID at $TOOL_SERVER_URL"
sleep 5   # wait for server to start

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH="$PYTHONPATH_VERL:$PYTHONPATH_VERL_TOOL" \
MCP_HTTP_SERVERS="$MCP_HTTP_SERVERS_JSON" \
$PYTHON -m verl_tool.trainer.main_ppo \
    algorithm.adv_estimator=$RL_ALG \
    data.train_files=$TRAIN_DATA \
    data.val_files=$VAL_DATA \
    data.train_batch_size=$BATCH_SIZE \
    data.val_batch_size=16 \
    data.max_prompt_length=$MAX_PROMPT_LEN \
    data.max_response_length=$MAX_RESPONSE_LEN \
    data.truncation=right \
    reward_model.reward_manager=biomedrxiv \
    reward_model.launch_reward_fn_async=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.lora_rank=16 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.actor.optim.lr_warmup_steps=5 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.checkpoint.save_contents=['model','optimizer','extra','hf_model'] \
    actor_rollout_ref.agent.enable_agent=True \
    actor_rollout_ref.agent.tool_server_url=$TOOL_SERVER_URL \
    actor_rollout_ref.agent.max_prompt_length=$MAX_PROMPT_LEN \
    actor_rollout_ref.agent.max_response_length=$MAX_RESPONSE_LEN \
    actor_rollout_ref.agent.max_start_length=$MAX_PROMPT_LEN \
    actor_rollout_ref.agent.max_obs_length=$MAX_OBS_LEN \
    actor_rollout_ref.agent.max_turns=$MAX_TURNS \
    actor_rollout_ref.agent.additional_eos_token_ids=[151645] \
    actor_rollout_ref.agent.mask_observations=True \
    actor_rollout_ref.agent.action_stop_tokens=$ACTION_STOP_FILE \
    actor_rollout_ref.agent.enable_mtrl=False \
    actor_rollout_ref.agent.max_action_length=$MAX_RESPONSE_LEN \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEM_UTIL \
    actor_rollout_ref.rollout.temperature=$TEMPERATURE \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.n=$N \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.max_num_seqs=64 \
    actor_rollout_ref.rollout.mode=sync \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=biomedrxiv-rl \
    trainer.experiment_name=$RUN_NAME \
    trainer.val_before_train=True \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.rollout_data_dir=/workspaces/gxl/qwen_rl/verl_step_records/$RUN_NAME \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.total_epochs=5 \
    trainer.default_local_dir=/workspaces/gxl/qwen_rl/checkpoints

kill -9 $TOOL_SERVER_PID 2>/dev/null || true
rm -f "$ACTION_STOP_FILE"
