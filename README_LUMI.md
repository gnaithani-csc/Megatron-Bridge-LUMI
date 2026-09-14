# Megatron-Bridge-LUMI

A self-contained fork of [NVIDIA-NeMo/Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge)
(`r0.3.0`) ported to run **Qwen3-VL** pre-training throughput
benchmarks on **LUMI** (ROCm 7.0).

> **Provenance.** Continues NVIDIA-NeMo/Megatron-Bridge `r0.3.0` (commit `8301abfc`),
> with the LUMI port on branch `lumi-qwen3vl`. The full upstream history is preserved.

> The original upstream README is preserved as the main [`README.md`](README.md).

---

## 1. About this repo

It measures throughput (TFLOP/s/GPU, tokens/s, MFU) for `Qwen/Qwen3-VL-8B-Instruct` under Megatron-Bridge, using either:

- **mock data**: `qwen3_vl_8b_throughput_mock_config` (random tensors, no I/O), or
- **synthetic caption data**: `qwen3_vl_8b_synth_cap_energon_config` (Energon
  WebDataset produced by the [synth-data generator](#4-data-preparation)).

The generic entrypoint `scripts/training/run_recipe.py` loads a recipe by name,
applies CLI overrides, and runs `pretrain` or `finetune` with a chosen forward
step (`vlm_step` for Qwen3-VL).

---

## 2. Environment (LUMI)

A LUMI Singularity container provides the base stack (PyTorch/ROCm, Transformer
Engine, apex, flash-attn). Create a venv on top of the container and install the
packages this port adds:

```bash
export SIF_plus=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260807_115122/lumi-multitorch-plus-u24r70f21m50t210-20260807_115122.sif
singularity shell $SIF_plus
python -m venv env_megatron --system-site-packages
source env_megatron/bin/activate
pip install --no-build-isolation \
    megatron-energon==7.3.2 \
    transformers==5.5.4 \
    hydra-core==1.3.2 \
    omegaconf==2.3.0
```

The container comes from the LUMI AI Factory Services (LAIFS) collection under
`/appl/local/laifs/containers`. See the [LAIFS container recipes](https://github.com/lumi-ai-factory/laifs-container-recipes/releases)
for the naming scheme.

`megatron-core 0.16.0` (`core_v0.16.0`) is provided as the `3rdparty/Megatron-LM`
submodule. Pull it with:

```bash
git submodule update --init 3rdparty/Megatron-LM
```

The launch scripts put it on `PYTHONPATH`.

---

## 3. Running a benchmark

Two launch scripts are provided: `scripts/training/launch_interactive.sh` for a
single node and `scripts/training/launch_with_sbatch.sh` for multi-node Slurm.
Both wrap:

```bash
torchrun ... scripts/training/run_recipe.py \
    --recipe <RECIPE> --step_func vlm_step --mode pretrain \
    --seq_length <N> <key=value overrides...>
```

`--step_func vlm_step` is the generic vision-language forward step (there is no
separate `qwen3_vl_step`). Overrides are dot-path `key=value` pairs applied to
the config after the recipe is built.

### Single node (interactive, 8 GPUs)

Edit `RECIPE` and `CLI_OVERRIDES` at the top of the script if needed, then:

```bash
cd <repo root>
bash scripts/training/launch_interactive.sh
```

Defaults: `qwen3_vl_8b_synth_cap_energon_config`, TP=2, GBS=64, `seq_length=8192`,
`transformer_impl=te`. Output is written to `train.log`.


### Multi-node

Edit the `#SBATCH` header (`--nodes`, `--account`, `--partition`) and
`RECIPE` and `CLI_OVERRIDES`, then submit **from the repo root**:

```bash
sbatch scripts/training/launch_with_sbatch.sh
```

Defaults: 4 nodes, TP=2, GBS=256, `seq_length=8192`, one `torchrun` per node
inside `singularity exec $SIF_plus`, logs under `logs/train_%j.{out,err}`.

### Recipes

| Recipe | Data | Notes |
|---|---|---|
| `qwen3_vl_8b_throughput_mock_config` | mock | random tensors, `train_iters=30`, no checkpoint |
| `qwen3_vl_8b_synth_cap_energon_config` | Energon synth captions | default, points at `cap_pretrain_converted` |
| `qwen3_vl_8b_pretrain_config` | n/a | stage-1 alignment (ViT frozen) |
| `qwen3_vl_8b_finetune_config` / `*_30b_a3b_*` / `*_235b_a22b_*` | n/a | larger and PEFT variants |

Recipes live in `src/megatron/bridge/recipes/qwen_vl/qwen3_vl.py`.

---

## 4. Data preparation

Only the energon recipe needs data. The mock recipe runs on random tensors with no
data preparation.

The synthetic dataset comes from a separate generator repo, [elliot-project/synth-data-bench-training](https://github.com/elliot-project/synth-data-bench-training).
Clone it first:

```bash
git clone https://github.com/elliot-project/synth-data-bench-training
cd synth-data-bench-training && git checkout cf96419   # commit used in this benchmark
```

This repo's `scripts/data/convert_synth_data.py` re-packs those shards into the
format the `QwenVLTaskEncoder` consumes.

### Format the encoder consumes

`QwenVLTaskEncoder` and `cook_chatml_sample` expect, per sample:

- `sample_XXXXXXXX.jpgs`: `pickle.dumps([np.ndarray, ...])` (list of images)
- `sample_XXXXXXXX.json`: a **bare** conversation list
  `[{"from":"human","value":...},{"from":"gpt","value":...}]`

The generator instead emits `sample_XXXXXXXX.jpg` (raw JPEG) and
`sample_XXXXXXXX.json` wrapped as `{"id":..., "conversations":[...]}`. The
conversion step bridges the two.

### Recommended pipeline

`bash scripts/data/prepare_synth_data.sh` runs the full pipeline (generate, convert,
`energon prepare`). Paths are set by env vars documented in the script header. The
manual steps are:

```bash
# 1. Generate raw shards in the generator repo (edit the TOML first).
python src/generate.py configs/cap_pretrain.toml

# 2. Convert to the format the encoder consumes (from this repo).
python scripts/data/convert_synth_data.py --input <gen_dir> --output <gen_dir>_converted

# 3. Build Energon metadata on the converted directory.
energon prepare <gen_dir>_converted --non-interactive --split-ratio 1.0,0,0 --sample-type CrudeWebdataset
```

Point the recipe at the converted directory with `dataset.path=<gen_dir>_converted`.

## 5. Important modifications

The recipes already set the values that run correctly on LUMI, so no changes are
required to run a benchmark. For reference, the settings that matter are:

- **`gradient_accumulation_fusion=False`** is required on LUMI. Leaving it on breaks
  training.
- **`transformer_impl`** can be `"local"` (default) or `"te"`. Both train correctly.
  `model.transformer_impl=te` selects Transformer Engine.
- **`cross_entropy_fusion_impl="native"`** is the portable default. `"te"` also works
  on this stack.

These are config defaults set inside `qwen3_vl_8b_throughput_mock_config` and
`qwen3_vl_8b_synth_cap_energon_config`. Override any of them with a `key=value`
argument, for example `model.transformer_impl=te`.
