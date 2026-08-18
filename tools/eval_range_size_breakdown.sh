#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

GPU_IDS=${GPU_IDS:-0,1,2,3,4,5,6,7}
GPUS=${GPUS:-8}
PORT_BASE=${PORT_BASE:-29720}
RUN_INFERENCE=${RUN_INFERENCE:-1}
BOOTSTRAP=${BOOTSTRAP:-1000}
SEED=${SEED:-0}

BASELINE_CONFIG=${BASELINE_CONFIG:-projects/configs/petrv2_depth/petrv2_depth_3dpe_dfl_vovnet_wogridmask_p4_1600x640_pdg_trainval_keyframe.py}
BASELINE_CHECKPOINT=${BASELINE_CHECKPOINT:-}
OURS_CONFIG=${OURS_CONFIG:-projects/configs/petrv2_depth/petrv2_depth_3dpe_dfl_vovnet_wogridmask_p4_1600x640_pdg_trainval_keyframe_densegt_far3d_stagea_top1_ddn_rangedn_lossd005.py}
OURS_CHECKPOINT=${OURS_CHECKPOINT:-work_dirs/stagea_top1_ddn_rangedn_lossd005/epoch_25.pth}
ANALYSIS_CONFIG=${ANALYSIS_CONFIG:-$OURS_CONFIG}

OUT_DIR=${OUT_DIR:-work_dirs/range_size_breakdown}
BASELINE_PKL=${BASELINE_PKL:-$OUT_DIR/baseline_results.pkl}
OURS_PKL=${OURS_PKL:-$OUT_DIR/ours_results.pkl}
DATA_ROOT=${DATA_ROOT:-data/nuscenes/}
ANN_FILE=${ANN_FILE:-data/nuscenes/petr/mmdet3d_nuscenes_30f_infos_val.pkl}
PAPER_FIGURE_DIR=${PAPER_FIGURE_DIR:-}

mkdir -p "$OUT_DIR"

require_file() {
    local path=$1
    local label=$2
    if [[ -z "$path" || ! -f "$path" ]]; then
        echo "$label not found: ${path:-<unset>}" >&2
        exit 1
    fi
}

run_inference() {
    local config=$1
    local checkpoint=$2
    local output=$3
    local port=$4
    shift 4
    require_file "$config" "Config"
    require_file "$checkpoint" "Checkpoint"
    local command=(
        bash tools/dist_test.sh "$config" "$checkpoint" "$GPUS"
        --out "$output"
        --eval bbox
    )
    CUDA_VISIBLE_DEVICES="$GPU_IDS" PORT="$port" "${command[@]}" "$@"
}

if [[ "$RUN_INFERENCE" == "1" ]]; then
    run_inference "$BASELINE_CONFIG" "$BASELINE_CHECKPOINT" "$BASELINE_PKL" "$PORT_BASE"

    OURS_OPTIONS=(
        --cfg-options
        model.pts_bbox_head.far3d_stagea_cfg.score_thr=0.05
        model.pts_bbox_head.far3d_stagea_cfg.sample_max_per_cam=24
        model.pts_bbox_head.far3d_stagea_cfg.topk_per_cam=24
        model.pts_bbox_head.far3d_stagea_cfg.max_adaptive_queries=144
        model.pts_bbox_head.far3d_stagea_cfg.depth_topk=1
        model.pts_bbox_head.far3d_stagea_cfg.depth_aggregation=window
        model.pts_bbox_head.far3d_stagea_cfg.depth_window=3
    )
    run_inference "$OURS_CONFIG" "$OURS_CHECKPOINT" "$OURS_PKL" "$((PORT_BASE + 1))" "${OURS_OPTIONS[@]}"
else
    require_file "$BASELINE_PKL" "Baseline result pkl"
    require_file "$OURS_PKL" "Ours result pkl"
fi

ANALYSIS_ARGS=(
    "$ANALYSIS_CONFIG"
    "$BASELINE_PKL"
    "$OURS_PKL"
    --baseline-name "3DPPE"
    --ours-name "DGAQ3D"
    --out-dir "$OUT_DIR"
    --data-root "$DATA_ROOT"
    --ann-file "$ANN_FILE"
    --bootstrap "$BOOTSTRAP"
    --bootstrap-unit scene
    --seed "$SEED"
)
if [[ -n "$PAPER_FIGURE_DIR" ]]; then
    ANALYSIS_ARGS+=(--paper-figure-dir "$PAPER_FIGURE_DIR")
fi

python tools/analyze_range_size_breakdown.py "${ANALYSIS_ARGS[@]}" 2>&1 | tee "$OUT_DIR/analysis.log"

echo
echo "Outputs:"
echo "  $OUT_DIR/range_size_metrics.csv"
echo "  $OUT_DIR/range_size_per_class.csv"
echo "  $OUT_DIR/far3d_style_range_curves.pdf"
echo "  $OUT_DIR/far3d_style_range_curves.png"
echo "  $OUT_DIR/range_size_bucket_map.pdf"
echo "  $OUT_DIR/summary.json"
