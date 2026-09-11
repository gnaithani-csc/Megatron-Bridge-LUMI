#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
set -euo pipefail
#export HIPBLASLT_LOG_LEVEL=2
#export HIPBLASLT_LOG_MASK=32

TRAINING_SCRIPT="run_recipe.py"

RECIPE="qwen3_vl_8b_synth_cap_energon_config" #"qwen3_vl_8b_throughput_mock_config"

# qwen3_vl_step is not registered in run_recipe.py; vlm_step is the generic VLM
# forward step that handles the same visual_inputs interface used by Qwen3-VL.
STEP_TYPE="vlm_step"

# Batch size lives in two configs: train.* (trainer: grad-accum / schedule) and
# dataset.* (dataloader: batch size it emits). Keep the two equal or they desync.
CLI_OVERRIDES="model.tensor_model_parallel_size=2 \
train.micro_batch_size=1 \
dataset.micro_batch_size=1 \
dataset.num_workers=1 \
dataset.pin_memory=True \
train.global_batch_size=64 \
dataset.global_batch_size=64 \
logger.log_throughput_to_tensorboard=true \
logger.log_throughput=true \
logger.log_interval=5 \
model.transformer_impl=te \
model.cross_entropy_fusion_impl=te \
model.gradient_accumulation_fusion=False "

# Container image
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings
CONTAINER_IMAGE=$SIF_plus
VENV=$PROJ_DIR/env_megatron

# Container mounts (space-separated host:container pairs, optional)
# CONTAINER_MOUNTS="/data:/data /scratch:/scratch"
CONTAINER_MOUNTS=""

# ==============================================================================
# Environment Setup
# ==============================================================================

export TORCH_NCCL_AVOID_RECORD_STREAMS=1
# Reduce allocator fragmentation so reserved-but-unallocated memory can satisfy new allocations.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

#export NCCL_SOCKET_IFNAME=hsn      # Slingshot high-speed network
#export MIOPEN_DISABLE_CACHE=1      # avoid MIOpen cache collisions in containers
#set MIOPEN temp folder
MIOPEN_DIR="/scratch/project_462001202/$USER/cache/miopen_cache" #$(mktemp -d)
mkdir -p $MIOPEN_DIR/cache
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_DIR/cache
export MIOPEN_USER_DB=$MIOPEN_DIR/config
#export MIOPEN_DEBUG_CONV_DIRECT=1  # no effect: MIOpen confirmed NOT the source (MIOPEN_LOG_LEVEL=4 produced no debug output)
#export MIOPEN_LOG_LEVEL=4          # no longer needed: confirmed MIOpen is not the hang source


export PYTORCH_TUNABLEOP_ENABLED=0
#export TORCH_BLAS_PREFER_HIPBLASLT=0

export CUDA_DEVICE_MAX_CONNECTIONS=1

#export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=COLL
#export NCCL_ASYNC_ERROR_HANDLING=1

#export RCCL_MSCCL_ENABLE=0
#export NCCL_ALGO=Ring


#export HF_HOME=/scratch/project_462001202/$USER/cache
export HF_HUB_CACHE=/scratch/project_462001202/$USER/cache/hub
# Repo root is two levels above this script (scripts/training/ -> repo root)
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# megatron.bridge  — the src-layout package in this repo
export PYTHONPATH="${REPO_DIR}/src:$PYTHONPATH"
export PYTHONPATH="${REPO_DIR}/3rdparty/Megatron-LM:$PYTHONPATH"
# project venv site-packages (flash-attn etc.)
export PYTHONPATH="$VENV/lib/python3.12/site-packages:$PYTHONPATH"
#export PYTHONPATH="$PYTHONPATH:/opt/venv/lib/python3.12/site-packages/flash_attn-2.8.0.post2-py3.12-linux-x86_64.egg"
#export SINGULARITYENV_PYTHONPATH="$PYTHONPATH"

# ==============================================================================
# Job Execution
# ==============================================================================

# Redirect all output through tee from this point so the header and torchrun
# output both land in the log file.
exec > >(tee train.log) 2>&1
#exec > >(tee del.log) 2>&1

echo "======================================"
echo "Megatron Bridge Training Job (interactive)"
echo "======================================"
echo "Job ID:        ${SLURM_JOB_ID:-<no allocation>}"
echo "Nodes:         ${SLURM_JOB_NUM_NODES:-1}"
echo "GPUs per node: ${SLURM_GPUS_PER_NODE:-8}"
echo "Script:        $TRAINING_SCRIPT"
echo "Recipe:        $RECIPE"
echo "Step:          $STEP_TYPE"
echo "Overrides:     $CLI_OVERRIDES"
[ -n "${HF_TOKEN:-}" ]     && echo "HF_TOKEN:      Set"
[ -n "${WANDB_API_KEY:-}" ] && echo "WANDB_API_KEY: Set"
echo "======================================"

# BASH_SOURCE[0] resolves correctly for interactive execution (unlike sbatch)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/${TRAINING_SCRIPT}"

if [ ! -f "$SCRIPT_PATH" ]; then
    echo "ERROR: Training script not found: $SCRIPT_PATH"
    exit 1
fi

if [ -z "$CONTAINER_IMAGE" ]; then
    echo "ERROR: CONTAINER_IMAGE is not set."
    exit 1
fi

#COMPILER
#export CC=gcc-12
#export CXX=g++-12


# Build torchrun command
GPUS_PER_NODE="${SLURM_GPUS_PER_NODE:-8}"
NUM_NODES="${SLURM_JOB_NUM_NODES:-1}"

CMD="torchrun"
CMD="$CMD --nproc_per_node=$GPUS_PER_NODE"
CMD="$CMD --nnodes=$NUM_NODES"
CMD="$CMD --node_rank=0"
CMD="$CMD --master_addr=$(hostname)"
CMD="$CMD --master_port=29500"
CMD="$CMD $SCRIPT_PATH"
CMD="$CMD --recipe $RECIPE"
CMD="$CMD --step_func $STEP_TYPE"
CMD="$CMD --mode pretrain"
CMD="$CMD --seq_length 8192"
[ -n "$CLI_OVERRIDES" ] && CMD="$CMD $CLI_OVERRIDES"



echo "Executing: $CMD"
echo "======================================"

# Add container mounts
MOUNT_ARGS=""
for mount in $CONTAINER_MOUNTS; do
    MOUNT_ARGS="$MOUNT_ARGS --bind $mount"
done

eval $CMD ; exit $?

#srun --label \
#    singularity exec \
#    $MOUNT_ARGS \
#    $CONTAINER_IMAGE \
#    bash -c "$CMD"

echo "======================================"
echo "Job completed"
echo "======================================"
