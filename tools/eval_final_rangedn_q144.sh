#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

GPU_IDS=${GPU_IDS:-0,1,2,3,4,5,6,7}
GPUS=${GPUS:-8}
PORT_BASE=${PORT_BASE:-29620}
TARGETS=${TARGETS:-keyframe,sweep}

KEYFRAME_CFG=${KEYFRAME_CFG:-projects/configs/petrv2_depth/petrv2_depth_3dpe_dfl_vovnet_wogridmask_p4_1600x640_pdg_trainval_keyframe_densegt_far3d_stagea_top1_ddn_rangedn_lossd005.py}
KEYFRAME_WORK=${KEYFRAME_WORK:-work_dirs/stagea_top1_ddn_rangedn_lossd005}
KEYFRAME_EPOCH=${KEYFRAME_EPOCH:-25}

SWEEP_CFG=${SWEEP_CFG:-projects/configs/petrv2_depth/petrv2_depth_3dpe_dfl_vovnet_wogridmask_p4_1600x640_pdg_trainval_sweep1_densegt_far3d_stagea_top1_ddn_rangedn_lossd005.py}
SWEEP_WORK=${SWEEP_WORK:-work_dirs/stagea_top1_ddn_rangedn_sweep1_lossd005}
SWEEP_EPOCH=${SWEEP_EPOCH:-26}

declare -a SUMMARY_LOGS=()

target_enabled() {
    local target=$1
    [[ ",${TARGETS}," == *",${target},"* ]]
}

run_eval() {
    local label=$1
    local cfg=$2
    local work=$3
    local epoch=$4
    local port=$5

    local checkpoint="${work}/epoch_${epoch}.pth"
    local output_dir="${work}/q144_eval"
    local log_file="${output_dir}/e${epoch}_thr005_top24_q144.log"
    local stats_file="${output_dir}/e${epoch}_thr005_top24_q144_stats.json"

    if [[ ! -f "$cfg" ]]; then
        echo "Config not found: $cfg" >&2
        exit 1
    fi
    if [[ ! -f "$checkpoint" ]]; then
        echo "Checkpoint not found: $checkpoint" >&2
        exit 1
    fi

    mkdir -p "$output_dir"

    echo "============================================================"
    echo "Target:      $label"
    echo "Config:      $cfg"
    echo "Checkpoint:  $checkpoint"
    echo "GPUs:        $GPU_IDS"
    echo "Log:         $log_file"
    echo "Query stats: $stats_file"
    echo "============================================================"

    CUDA_VISIBLE_DEVICES="$GPU_IDS" PORT="$port" \
    bash tools/dist_test.sh "$cfg" "$checkpoint" "$GPUS" \
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

    SUMMARY_LOGS+=("${label}|${epoch}|${log_file}|${stats_file}")
}

if target_enabled keyframe; then
    run_eval \
        keyframe \
        "$KEYFRAME_CFG" \
        "$KEYFRAME_WORK" \
        "$KEYFRAME_EPOCH" \
        "$PORT_BASE"
fi

if target_enabled sweep; then
    run_eval \
        one-sweep \
        "$SWEEP_CFG" \
        "$SWEEP_WORK" \
        "$SWEEP_EPOCH" \
        "$((PORT_BASE + 1))"
fi

if [[ ${#SUMMARY_LOGS[@]} -eq 0 ]]; then
    echo "No target selected. Use TARGETS=keyframe, TARGETS=sweep, or TARGETS=keyframe,sweep." >&2
    exit 1
fi

echo
echo "==================== q144 evaluation summary ===================="
for item in "${SUMMARY_LOGS[@]}"; do
    IFS='|' read -r label epoch log_file stats_file <<< "$item"
    echo
    echo "[$label epoch $epoch]"
    grep -oE 'mAP:[[:space:]]*[0-9.]+' "$log_file" | tail -n 1 || true
    grep -E \
        '^(mATE|mASE|mAOE|mAVE|mAAE|NDS):|^  (samples|mean|p95|max|cap|cap_hits|cap_hit_rate):' \
        "$log_file" || true
    echo "stats_json: $stats_file"
done
