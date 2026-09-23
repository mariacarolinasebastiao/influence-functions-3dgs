#!/usr/bin/env bash
: "${GS_WORK_ROOT:=/home/coder/data}"
: "${GS_PROJ_ROOT:=$GS_WORK_ROOT}"
set -uo pipefail

BICY_DATA=${GS_WORK_ROOT}/2026_sebastiao/projects/bicycle_splatfacto/loo_test7_data
BICY_OUT=${GS_WORK_ROOT}/2026_sebastiao/projects/bicycle_splatfacto/loo_test7
ROOM_DATA=${GS_WORK_ROOT}/2026_sebastiao/projects/room_splatfacto/loo_test28_data
ROOM_OUT=${GS_WORK_ROOT}/2026_sebastiao/projects/room_splatfacto/loo_test28
DOWNSCALE=4

MIG=(
  MIG-04cdc680-184c-5e77-b2de-4535ef10e69f   # GPU0 dev0
  MIG-6e0bb724-97bd-5570-b1be-fc4fdfebae84   # GPU0 dev1
  MIG-5d8d4f9f-63c2-5302-ad10-fb3bdda2bc1f   # GPU1 dev0
  MIG-2a48cb77-31d0-559d-b550-10f5bad3f341   # GPU1 dev1
  MIG-2a133531-e2c3-55bb-8d93-ad6d7abdaa69   # GPU2 dev0
  MIG-2527fa0b-fe27-51ba-be77-0f3e90644a52   # GPU2 dev1
  MIG-cb01dd73-09eb-55bf-9f02-4e72a49821a1   # GPU3 dev0
  MIG-c184e883-f4e7-51fb-8d17-768ed3ec297c   # GPU3 dev1
)

train_one() {
  local dev=$1 data=$2 out=$3 log=$4
  if [ ! -f "$data/transforms.json" ]; then echo "!! SKIP: no dataset at $data"; return 1; fi
  mkdir -p "$out"
  echo "[$(date +%H:%M:%S)] START $(basename $out)"
  CUDA_VISIBLE_DEVICES="$dev" TORCHDYNAMO_DISABLE=1 TORCH_COMPILE_DISABLE=1 \
    ns-train splatfacto \
      --output-dir "$out" --data "$data" \
      --vis tensorboard --steps-per-save 30000 \
      nerfstudio-data --downscale-factor "$DOWNSCALE" --eval-mode filename \
      > "$log" 2>&1
  echo "[$(date +%H:%M:%S)] DONE  $(basename $out)"
}

mkdir -p "$BICY_OUT" "$ROOM_OUT"; pids=()

# Slices 0-1: bicycle first, then queue a room job
{ train_one "${MIG[0]}" "$BICY_DATA/bicycle_data_remove28"  "$BICY_OUT/bicycle_loo_remove28"  "$BICY_OUT/remove28.log"  && \
  train_one "${MIG[0]}" "$ROOM_DATA/room_data_remove124"    "$ROOM_OUT/room_loo_remove124"    "$ROOM_OUT/remove124.log"; } & pids+=($!)

{ train_one "${MIG[1]}" "$BICY_DATA/bicycle_data_remove70"  "$BICY_OUT/bicycle_loo_remove70"  "$BICY_OUT/remove70.log"  && \
  train_one "${MIG[1]}" "$ROOM_DATA/room_data_remove127"    "$ROOM_OUT/room_loo_remove127"    "$ROOM_OUT/remove127.log"; } & pids+=($!)

# Slices 2-4: remaining bicycle only
train_one "${MIG[2]}" "$BICY_DATA/bicycle_data_remove73"  "$BICY_OUT/bicycle_loo_remove73"  "$BICY_OUT/remove73.log"  & pids+=($!)
train_one "${MIG[3]}" "$BICY_DATA/bicycle_data_remove24"  "$BICY_OUT/bicycle_loo_remove24"  "$BICY_OUT/remove24.log"  & pids+=($!)
train_one "${MIG[4]}" "$BICY_DATA/bicycle_data_remove113" "$BICY_OUT/bicycle_loo_remove113" "$BICY_OUT/remove113.log" & pids+=($!)

# Slices 5-7: room only
train_one "${MIG[5]}" "$ROOM_DATA/room_data_remove263"    "$ROOM_OUT/room_loo_remove263"    "$ROOM_OUT/remove263.log" & pids+=($!)
train_one "${MIG[6]}" "$ROOM_DATA/room_data_remove126"    "$ROOM_OUT/room_loo_remove126"    "$ROOM_OUT/remove126.log" & pids+=($!)
train_one "${MIG[7]}" "$ROOM_DATA/room_data_remove265"    "$ROOM_OUT/room_loo_remove265"    "$ROOM_OUT/remove265.log" & pids+=($!)

echo "8 jobs launched (5 bicycle + 3 room in parallel; 2 more room queued on slices 0-1)..."
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
[ "$fail" -eq 0 ] && echo "ALL OK" || echo "SOME FAILED — check logs in $BICY_OUT and $ROOM_OUT"