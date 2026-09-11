#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ==============================================================================
# Multi-node Slurm launch (LUMI / ROCm)
#
# Usage:
#   sbatch launch_with_sbatch.sh
#
# One torchrun process per node (ntasks-per-node=1) spawns 8 GPU workers
# internally. SLURM_NODEID is evaluated per srun task so each node gets the
# correct --node_rank. All env-var and PYTHONPATH logic mirrors
# launch_interactive.sh.
# ==============================================================================

#SBATCH --job-name=mb_multinode
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=03:00:00
#SBATCH --partition=dev-g #standard-g
#SBATCH --account=project_462001202 #462000131  #project_462001202
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --exclusive
#SBATCH --mem=0

set -euox pipefail

TRAINING_SCRIPT="run_recipe.py"
RECIPE="qwen3_vl_8b_synth_cap_energon_config"      #"qwen3_vl_8b_throughput_mock_config"
STEP_TYPE="vlm_step"
#export NVTE_DEBUG=1
#export NVTE_DEBUG_LEVEL=2
#export NCCL_TUNER_PLUGIN=/appl/local/containers/for-turkunlp-team/tuner-2025-07-09/librccl-tuner.so
#export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=INIT,TUNING,NET,COLL
#export NCCL_DEBUG_FILE=nccl.%h.%p.log
#if [ "$SLURM_NODEID" = "0" ] || [ "$SLURM_LOCALID" = "0" ]; then
#    export NCCL_DEBUG=INFO                           # OFF, WARN(warnings and errors), INFO(Basic info + version, initialization, size info) , TRACE(Verbose)
#    export NCCL_DEBUG_SUBSYS=INIT,TUNING,NET,COLL    # INIT, TUNING(algo selection), NET(comm layer), COLL(collectives)
#    export NCCL_DEBUG_FILE=logs/nccl/nccl.%h.%p.log  # %h hostname, %p process ID
#fi

# Batch size lives in two configs: train.* (trainer: grad-accum / schedule) and
# dataset.* (dataloader: batch size it emits). Keep the two equal or they desync.
# GBS scales with node count (about 64 per 8 GPUs), so it is 256 for 4 nodes.
CLI_OVERRIDES="model.tensor_model_parallel_size=2 \
train.micro_batch_size=1 \
dataset.micro_batch_size=1 \
dataset.num_workers=1 \
dataset.pin_memory=True \
train.global_batch_size=256 \
dataset.global_batch_size=256 \
logger.log_throughput_to_tensorboard=true \
logger.log_throughput=true \
logger.log_interval=5 \
model.transformer_impl=te \
train.train_iters=10  \
model.gradient_accumulation_fusion=False \
model.cross_entropy_fusion_impl=te "

#export NVTE_DEBUG=1
#export NVTE_DEBUG_LEVEL=2
#export NVTE_UNFUSED_ATTN=0

# Container image (set by lumi-aif-singularity-bindings)
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings
export SIF_plus="/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260513_121430/lumi-multitorch-plus-u24r70f21m50t210-20260513_121430.sif"
CONTAINER_IMAGE=$SIF_plus
CONTAINER_MOUNTS=""
VENV=$PROJ_DIR/env_megatron

# ==============================================================================
# Environment Setup
# ==============================================================================
export TORCH_DIST_INIT_TIMEOUT=600  # seconds, for the TCPStore rendezvous
export MEGATRON_CONFIG_LOCK_DIR=/tmp


export TORCH_NCCL_AVOID_RECORD_STREAMS=1
# Reduce allocator fragmentation so reserved-but-unallocated memory can satisfy new allocations.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export NCCL_SOCKET_IFNAME=hsn      # Slingshot high-speed network (recommended for multi-node)

#export MIOPEN_ROOT="/scratch/project_462001202/$USER/cache/miopen
export MIOPEN_ROOT="/tmp/$USER/miopen-$SLURM_JOB_ID-$SLURM_NODEID"
mkdir -p "$MIOPEN_ROOT"
export MIOPEN_USER_DB_PATH="$MIOPEN_ROOT"
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_ROOT"
#export MIOPEN_DISABLE_CACHE=1

export PYTORCH_TUNABLEOP_ENABLED=0
#export TORCH_BLAS_PREFER_HIPBLASLT=0

export CUDA_DEVICE_MAX_CONNECTIONS=1

#export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=COLL
#export RCCL_MSCCL_ENABLE=0
#export NCCL_ALGO=Ring

export HF_HUB_CACHE=/scratch/project_462001202/$USER/cache/hub
export HF_HUB_OFFLINE=1
# SLURM_SUBMIT_DIR is the cwd at submission time.
# Convention: submit from the repo root — sbatch scripts/training/launch_with_sbatch.sh
REPO_DIR="${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"
export PYTHONPATH="$HOME/Megatron-LM:$PYTHONPATH" #"${REPO_DIR}/3rdparty/Megatron-LM:$PYTHONPATH"
export PYTHONPATH="$VENV/lib/python3.12/site-packages:$PYTHONPATH"
export PYTHONPATH="$PYTHONPATH:/opt/venv/lib/python3.12/site-packages/flash_attn-2.8.0.post2-py3.12-linux-x86_64.egg" # For Turku-nlp conatainer

# ==============================================================================
# Job Execution
# ==============================================================================

mkdir -p logs

SCRIPT_DIR="${REPO_DIR}/scripts/training"
SCRIPT_PATH="${SCRIPT_DIR}/${TRAINING_SCRIPT}"

if [ ! -f "$SCRIPT_PATH" ]; then
    echo "ERROR: Training script not found: $SCRIPT_PATH"
    exit 1
fi

if [ -z "$CONTAINER_IMAGE" ]; then
    echo "ERROR: CONTAINER_IMAGE is not set."
    exit 1
fi

MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -n1)
GPUS_PER_NODE="${SLURM_GPUS_PER_NODE:-8}"
NUM_NODES="${SLURM_JOB_NUM_NODES:-1}"

echo "======================================"
echo "Megatron Bridge Training Job (sbatch)"
echo "======================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Nodes:         $NUM_NODES"
echo "GPUs per node: $GPUS_PER_NODE"
echo "Master addr:   $MASTER_ADDR"
echo "Script:        $TRAINING_SCRIPT"
echo "Recipe:        $RECIPE"
echo "Step:          $STEP_TYPE"
echo "Overrides:     $CLI_OVERRIDES"
[ -n "${HF_TOKEN:-}" ]      && echo "HF_TOKEN:      Set"
[ -n "${WANDB_API_KEY:-}" ] && echo "WANDB_API_KEY: Set"
echo "======================================"

# \$SLURM_NODEID is intentionally un-expanded here; bash -c evaluates it per srun task.
CMD="torchrun"
CMD="$CMD --nproc_per_node=$GPUS_PER_NODE"
CMD="$CMD --nnodes=$NUM_NODES"
CMD="$CMD --node_rank=\$SLURM_NODEID"
CMD="$CMD --master_addr=$MASTER_ADDR"
CMD="$CMD --master_port=29500"
CMD="$CMD $SCRIPT_PATH"
CMD="$CMD --recipe $RECIPE"
CMD="$CMD --step_func $STEP_TYPE"
CMD="$CMD --mode pretrain"
CMD="$CMD --seq_length 8192"
[ -n "$CLI_OVERRIDES" ] && CMD="$CMD $CLI_OVERRIDES"
echo "Executing: $CMD"
echo "======================================"

MOUNT_ARGS=""
for mount in $CONTAINER_MOUNTS; do
    MOUNT_ARGS="$MOUNT_ARGS --bind $mount"
done

srun --label \
    singularity exec \
    $MOUNT_ARGS \
    $CONTAINER_IMAGE \
    bash -c "export PYTHONPATH='$PYTHONPATH':\${PYTHONPATH:-}; $CMD"

echo "======================================"
echo "Job completed"
echo "======================================"
