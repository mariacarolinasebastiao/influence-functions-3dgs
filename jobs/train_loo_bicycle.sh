#!/bin/bash
: "${GS_WORK_ROOT:=/home/coder/data}"
: "${GS_PROJ_ROOT:=$GS_WORK_ROOT}"
source ${GS_PROJ_ROOT}/nerfstudio/venv/bin/activate
export QT_QPA_PLATFORM=offscreen
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1

DATA=${GS_WORK_ROOT}/2026_sebastiao/projects/room_splatfacto/loo_test28_data
OUT=${GS_WORK_ROOT}/2026_sebastiao/projects/room_splatfacto/loo_test28

mapfile -t MIGS < <(nvidia-smi -L | grep -oE 'MIG-[0-9a-f-]+')

run () {
    CUDA_VISIBLE_DEVICES=$1 ns-train splatfacto --output-dir "$OUT/room_loo_remove$2" \
      --data "$DATA/room_data_remove$2" --vis tensorboard --steps-per-save 30000 \
      nerfstudio-data --downscale-factor 4 --eval-mode filename
}

# 10 indices across 16 MIG slices (GPU0: 4 slices, GPU1: 4 slices, GPU2: 4 slices, GPU3: 4 slices)
# Indices: 261 262 14 17 15 272 274 269 271 132

( run "${MIGS[0]}"  261; run "${MIGS[0]}"  262 ) &   # GPU0 slice0
( run "${MIGS[1]}"   14                            ) &   # GPU0 slice1
( run "${MIGS[2]}"   17                            ) &   # GPU0 slice2
( run "${MIGS[3]}"   15                            ) &   # GPU0 slice3

( run "${MIGS[4]}"  272                            ) &   # GPU1 slice0
( run "${MIGS[5]}"  274                            ) &   # GPU1 slice1
( run "${MIGS[6]}"  269                            ) &   # GPU1 slice2
( run "${MIGS[7]}"  271                            ) &   # GPU1 slice3

( run "${MIGS[8]}"  132                            ) &   # GPU2 slice0
# GPU2 slices 1-3 and GPU3 slices 0-3 are idle (only 10 jobs total)

wait
echo "all done"