#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "${ROOT_DIR}"

# Edit the argument values below to try different generation settings.
# The finalized DaTMe selection rules remain fixed in the generation modules.
# Add --deterministic when exact repeatability is required.

# "${PYTHON_BIN}" -m generation.sd_generate \
#   --compression_ratio 0.3 \
#   --seed 12 \
#   --num_inference_steps 50 \
#   --guidance_scale 7.5 \
#   --height 512 \
#   --width 512 \
#   --device cuda \
#   --num_spatial_partitions 4 \
#   --merge_start_step 2 \
#   --prune_start_step 1 \
#   --spatial_scale 2 \
#   --output_dir outputs/sd \
#   --prompt "A bowl of oatmeal is ready to eat with some cinnamon and sugar"

"${PYTHON_BIN}" -m generation.sdxl_generate \
  --compression_ratio 0.3 \
  --seed 12 \
  --num_inference_steps 50 \
  --guidance_scale 5.0 \
  --height 1024 \
  --width 1024 \
  --device cuda \
  --merge_layers "0,1,2,3,64,65,66,67,68,69" \
  --num_spatial_partitions 4 \
  --merge_start_step 2 \
  --prune_start_step 1 \
  --spatial_scale 2 \
  --output_dir outputs/sdxl \
  --prompt "A giraffe and its kid in a grassy pin"

# "${PYTHON_BIN}" -m generation.pixart_generate \
#   --compression_ratio 0.3 \
#   --seed 12 \
#   --num_inference_steps 20 \
#   --guidance_scale 4.5 \
#   --height 1024 \
#   --width 1024 \
#   --device cuda \
#   --scheduler dpm \
#   --model_id "PixArt-alpha/PixArt-XL-2-1024-MS" \
#   --merge_layer_start 9 \
#   --merge_layer_end 19 \
#   --num_spatial_partitions 4 \
#   --merge_start_step 2 \
#   --prune_start_step 1 \
#   --spatial_scale 4 \
#   --output_dir outputs/pixart \
#   --prompt "An old man is wearing an odd hat"

# "${PYTHON_BIN}" -m generation.flux_generate \
#   --compression_ratio 0.3 \
#   --seed 12 \
#   --num_inference_steps 35 \
#   --guidance_scale 3.5 \
#   --height 1024 \
#   --width 1024 \
#   --device cuda \
#   --double_blocks "9:19" \
#   --single_blocks "7:22" \
#   --num_spatial_partitions 4 \
#   --merge_start_step 2 \
#   --spatial_scale 4 \
#   --output_dir outputs/flux \
#   --prompt "An old toilet with a hello kitty cover top"
