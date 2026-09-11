#!/bin/bash
# Generate and prepare the synthetic Qwen3-VL caption dataset, run inside the LUMI
# container + env_megatron venv (not the generator's uv).
# Pipeline: generate -> convert -> energon prepare.
# Generator repo: elliot-project/synth-data-bench-training, pinned at commit cf96419.
# Override via env vars: GENERATOR_REPO, GEN_CONFIG, GEN (raw dir), OUT, VENV.
set -euo pipefail

GENERATOR_REPO="${GENERATOR_REPO:-/scratch/project_462001202/gnaithan/synth-data-bench-training}"
GEN_CONFIG="${GEN_CONFIG:-configs/cap_pretrain.toml}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings
CONTAINER="${CONTAINER:-$SIF_plus}"
: "${CONTAINER:?container not set. Load lumi-aif-singularity-bindings first.}"
VENV="${VENV:-$PROJ_DIR/env_megatron}"

# Run a command inside the container with the venv importable.
runc() { singularity exec "$CONTAINER" bash -c \
  "export PATH=$VENV/bin:\$PATH PYTHONPATH=$VENV/lib/python3.12/site-packages:\$PYTHONPATH; $*"; }

# GEN defaults to the output_dir in the TOML.
if [ -z "${GEN:-}" ]; then
  GEN="$(grep -E '^[[:space:]]*output_dir' "$GENERATOR_REPO/$GEN_CONFIG" | head -1 | sed -E 's/.*=[[:space:]]*"([^"]+)".*/\1/')"
fi
[ -n "${GEN:-}" ] || { echo "ERROR: set GEN (raw output dir)."; exit 1; }
OUT="${OUT:-${GEN}_converted}"

echo "generate -> $GEN"
( cd "$GENERATOR_REPO" && runc "python src/generate.py $GEN_CONFIG" )

echo "convert -> $OUT"
runc "python $REPO_DIR/scripts/data/convert_synth_data.py --input $GEN --output $OUT"

echo "energon prepare -> $OUT"
runc "energon prepare $OUT --non-interactive --split-ratio 1.0,0,0 --sample-type CrudeWebdataset"

echo "done. point the recipe at: dataset.path=$OUT"
