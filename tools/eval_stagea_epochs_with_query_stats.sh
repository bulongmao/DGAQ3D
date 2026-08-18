#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

CFG=${CFG:-projects/configs/petrv2_depth/petrv2_depth_3dpe_dfl_vovnet_wogridmask_p4_1600x640_pdg_trainval_keyframe_densegt_far3d_stagea_top1_ddn_lossd005.py}
WORK=${WORK:-work_dirs/stagea_top1_ddn_warmup_lossd005}
GPU_IDS=${GPU_IDS:-0,1,2,3,4,5,6,7}
GPUS=${GPUS:-8}
PORT_BASE=${PORT_BASE:-29520}
EPOCHS=(24 25)

if [[ ! -f "$CFG" ]]; then
    echo "Config not found: $CFG" >&2
    exit 1
fi

mkdir -p "$WORK/query_stats"

for index in "${!EPOCHS[@]}"; do
    epoch=${EPOCHS[$index]}
    checkpoint="$WORK/epoch_${epoch}.pth"
    log_file="$WORK/e${epoch}_thr005_q144_stats.log"
    stats_file="$WORK/query_stats/e${epoch}_thr005_q144.json"
    port=$((PORT_BASE + index))

    if [[ ! -f "$checkpoint" ]]; then
        echo "Checkpoint not found: $checkpoint" >&2
        exit 1
    fi

    echo "============================================================"
    echo "Evaluating epoch $epoch"
    echo "Checkpoint: $checkpoint"
    echo "Log:        $log_file"
    echo "Stats:      $stats_file"
    echo "============================================================"

    CUDA_VISIBLE_DEVICES="$GPU_IDS" PORT="$port" \
    bash tools/dist_test.sh "$CFG" "$checkpoint" "$GPUS" \
        --eval bbox \
        --stagea-query-stats \
        --stagea-query-stats-out "$stats_file" \
        --cfg-options \
        model.pts_bbox_head.far3d_stagea_cfg.score_thr=0.05 \
        model.pts_bbox_head.far3d_stagea_cfg.sample_max_per_cam=24 \
        model.pts_bbox_head.far3d_stagea_cfg.topk_per_cam=24 \
        model.pts_bbox_head.far3d_stagea_cfg.max_adaptive_queries=144 \
        model.pts_bbox_head.far3d_stagea_cfg.depth_topk=1 \
        model.pts_bbox_head.far3d_stagea_cfg.depth_aggregation=window \
        model.pts_bbox_head.far3d_stagea_cfg.depth_window=3 \
        2>&1 | tee "$log_file"
done

echo
echo "======================= Summary ============================"
for epoch in "${EPOCHS[@]}"; do
    log_file="$WORK/e${epoch}_thr005_q144_stats.log"
    echo
    echo "Epoch $epoch"
    grep -oE 'mAP:[[:space:]]*[0-9.]+' "$log_file" | tail -n 1 || true
    grep -E \
        '^(mATE|mASE|mAOE|mAVE|mAAE|NDS):|^  (samples|mean|p95|max|cap|cap_hits|cap_hit_rate):' \
        "$log_file" || true
done
