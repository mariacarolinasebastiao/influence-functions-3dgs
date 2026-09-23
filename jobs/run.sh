#!/bin/bash
: "${GS_WORK_ROOT:=/home/coder/data}"
: "${GS_PROJ_ROOT:=$GS_WORK_ROOT}"


set -e  # stop on first error

LOAD_DIR="${GS_PROJ_ROOT}/nerfstudio/nerfstudio/outputs/garden/garden_processed/splatfacto/2026-03-07_184552/nerfstudio_models"
SOFT_TARGETS="${GS_PROJ_ROOT}/2026_sebastiao/projects/soft_targets_v2.pt"
DATA_DIR="${GS_PROJ_ROOT}/2026_sebastiao/projects/nerfstudio/garden_processed"
OUTPUT_BASE="${GS_PROJ_ROOT}/2026_sebastiao/projects/proximity_reg_01"



#INDICES=(52 56 81 108 109 130 131 133 148 157 160) 
#18 23 52 56 81 108 109 130 131 133 148 157 160)
INDICES=(108 109 110 130 131 132 133 148 157 159 160 17 23 50 52 56 78 80 81)

#removi o 18 pq já fiz, colocar novamente depois!

for IDX in "${INDICES[@]}"; do
    echo "========================================"
    echo "Training PBRF: removing train index $IDX"
    echo "Started at: $(date)"
    echo "========================================"

    ns-train splatfacto-pbrf \
        --output-dir "${OUTPUT_BASE}/proximity_remove${IDX}" \
        --load-dir "${LOAD_DIR}" \
        --load-step 29999 \
        --load-scheduler True \
        --max-num-iterations 44999 \
        --steps-per-save 44999 \
        --save-only-latest-checkpoint True \
        --logging.local-writer.enable False \
        --vis tensorboard \
        --pipeline.model.pbrf_lambda 0.01 \
        --pipeline.model.removed_train_idx "${IDX}" \
        --pipeline.model.soft_targets_path "${GS_PROJ_ROOT}/2026_sebastiao/projects/soft_targets_v2.pt" \
        --pipeline.model.pbrf_epsilon=0.006 \
        nerfstudio-data \
        --data "${GS_PROJ_ROOT}/2026_sebastiao/projects/nerfstudio/garden_processed/" \
        --downscale-factor 4 \
        --eval-mode filename

    echo "Finished idx $IDX at: $(date)"
    echo ""
done

echo "All done at: $(date)"

#        --data "${GS_PROJ_ROOT}/2026_sebastiao/projects/loo/garden_data_remove${IDX}" \

# --pipeline.model.pbrf_lambda 0.001 \
# --pipeline.model.removed_train_idx ${IDX} \
# --pipeline.model.soft_targets_path "${SOFT_TARGETS}" \
# --pipeline.model.pbrf_epsilon 0.006 \