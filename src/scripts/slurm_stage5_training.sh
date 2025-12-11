#!/bin/bash
#
# SLURM Coordinator Script for Stage 5: Model Training
#
# This script delegates work to individual algorithm scripts (one per algorithm)
# by submitting separate SLURM jobs for each algorithm.
#
# Usage:
#   sbatch src/scripts/slurm_stage5_training.sh
#
# Environment variables (passed to all substages):
#   FVC_NUM_FRAMES: Number of frames per video (default: 8)
#   FVC_N_SPLITS: Number of k-fold splits (default: 5)
#   FVC_STAGE5_OUTPUT_DIR: Output directory (default: data/training_results)
#   FVC_USE_TRACKING: Set to false to disable tracking (default: true)
#   FVC_DELETE_EXISTING: Set to 1/true/yes to delete existing models (default: 0)
#   FVC_STAGE3_OUTPUT_DIR: Stage 3 output directory (default: data/scaled_videos)
#   FVC_STAGE2_OUTPUT_DIR: Stage 2 output directory (default: data/features_stage2)
#   FVC_STAGE4_OUTPUT_DIR: Stage 4 output directory (default: data/features_stage4)

#SBATCH --job-name=fvc_stage5_coord
#SBATCH --account=eecs442f25_class
#SBATCH --partition=standard
#SBATCH --time=00:10:00
#SBATCH --mem=1G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/stage5_coord-%j.out
#SBATCH --error=logs/stage5_coord-%j.err
#SBATCH --mail-user=santoshd@umich.edu,urvim@umich.edu,suzanef@umich.edu
#SBATCH --mail-type=FAIL,TIME_LIMIT,NODE_FAIL

set -euo pipefail
set -o errtrace
umask 077

# ============================================================================
# Environment Setup
# ============================================================================

unset MallocStackLogging || true
unset MallocStackLoggingNoCompact || true

# ============================================================================
# Configuration
# ============================================================================

ORIG_DIR="${SLURM_SUBMIT_DIR:-$PWD}"

# All 21 algorithms ordered by complexity (fastest to slowest)
# 5a-5c: Baseline models (O(n*m) to O(T*H*W*C))
# 5d-5e: Pretrained CNNs (O(T*H*W*C))
# 5f-5j: XGBoost models (feature extraction + O(n*log(n)*m))
# 5k-5l: Frame-temporal models (ViT per frame + temporal)
# 5m-5n: Video transformers (full video attention)
# 5o-5q: Spatiotemporal 3D models (3D convolutions)
# 5r-5t: Advanced SlowFast variants (multiple pathways)
# 5u: Two-stream (dual streams + optical flow)
ALGORITHM_SCRIPTS=(
    "src/scripts/slurm_stage5a.sh"  # logistic_regression
    "src/scripts/slurm_stage5b.sh"  # svm
    "src/scripts/slurm_stage5c.sh"  # naive_cnn
    "src/scripts/slurm_stage5d.sh"  # pretrained_inception
    "src/scripts/slurm_stage5e.sh"  # variable_ar_cnn
    "src/scripts/slurm_stage5f.sh"  # xgboost_pretrained_inception
    "src/scripts/slurm_stage5g.sh"  # xgboost_i3d
    "src/scripts/slurm_stage5h.sh"  # xgboost_r2plus1d
    "src/scripts/slurm_stage5i.sh"  # xgboost_vit_gru
    "src/scripts/slurm_stage5j.sh"  # xgboost_vit_transformer
    "src/scripts/slurm_stage5k.sh"  # vit_gru
    "src/scripts/slurm_stage5l.sh"  # vit_transformer
    "src/scripts/slurm_stage5m.sh"  # timesformer
    "src/scripts/slurm_stage5n.sh"  # vivit
    "src/scripts/slurm_stage5o.sh"  # i3d
    "src/scripts/slurm_stage5p.sh"  # r2plus1d
    "src/scripts/slurm_stage5q.sh"  # x3d
    "src/scripts/slurm_stage5r.sh"  # slowfast
    "src/scripts/slurm_stage5s.sh"  # slowfast_attention
    "src/scripts/slurm_stage5t.sh"  # slowfast_multiscale
    "src/scripts/slurm_stage5u.sh"  # two_stream
)

# ============================================================================
# Logging Functions
# ============================================================================

log() {
    echo "$@" >&1
    echo "$@" >&2
    sync 2>/dev/null || true
}

# ============================================================================
# Main Execution
# ============================================================================

log "=========================================="
log "STAGE 5 COORDINATOR: Model Training"
log "=========================================="
log "Host:        $(hostname)"
log "Date:        $(date -Is)"
log "SLURM_JOBID: ${SLURM_JOB_ID:-none}"
log "Working directory: $ORIG_DIR"
log "=========================================="
log ""

# Verify algorithm scripts exist
log "Verifying algorithm scripts..."
MISSING_SCRIPTS=()
for script in "${ALGORITHM_SCRIPTS[@]}"; do
    script_path="$ORIG_DIR/$script"
    if [ ! -f "$script_path" ]; then
        MISSING_SCRIPTS+=("$script")
        log "✗ Missing: $script"
    else
        log "✓ Found: $script"
    fi
done

if [ ${#MISSING_SCRIPTS[@]} -gt 0 ]; then
    log "✗ ERROR: Missing algorithm scripts: ${MISSING_SCRIPTS[*]}"
    exit 1
fi

log ""
log "=========================================="
log "Submitting algorithm jobs..."
log "=========================================="

# Change to project root
cd "$ORIG_DIR" || exit 1

# Verify sbatch command is available
if ! command -v sbatch &> /dev/null; then
    log "✗ ERROR: sbatch command not found. This script must be run on a SLURM cluster."
    exit 1
fi

# Collect environment variables to pass to substages
ENV_VARS=()
if [ -n "${FVC_NUM_FRAMES:-}" ]; then
    ENV_VARS+=("FVC_NUM_FRAMES=$FVC_NUM_FRAMES")
fi
if [ -n "${FVC_N_SPLITS:-}" ]; then
    ENV_VARS+=("FVC_N_SPLITS=$FVC_N_SPLITS")
fi
if [ -n "${FVC_STAGE5_OUTPUT_DIR:-}" ]; then
    ENV_VARS+=("FVC_STAGE5_OUTPUT_DIR=$FVC_STAGE5_OUTPUT_DIR")
fi
if [ -n "${FVC_USE_TRACKING:-}" ]; then
    ENV_VARS+=("FVC_USE_TRACKING=$FVC_USE_TRACKING")
fi
if [ -n "${FVC_DELETE_EXISTING:-}" ]; then
    ENV_VARS+=("FVC_DELETE_EXISTING=$FVC_DELETE_EXISTING")
fi
if [ -n "${FVC_STAGE3_OUTPUT_DIR:-}" ]; then
    ENV_VARS+=("FVC_STAGE3_OUTPUT_DIR=$FVC_STAGE3_OUTPUT_DIR")
fi
if [ -n "${FVC_STAGE2_OUTPUT_DIR:-}" ]; then
    ENV_VARS+=("FVC_STAGE2_OUTPUT_DIR=$FVC_STAGE2_OUTPUT_DIR")
fi
if [ -n "${FVC_STAGE4_OUTPUT_DIR:-}" ]; then
    ENV_VARS+=("FVC_STAGE4_OUTPUT_DIR=$FVC_STAGE4_OUTPUT_DIR")
fi

# Submit all algorithm jobs
SUBMITTED_JOBS=()
FAILED_SUBMISSIONS=()

for script in "${ALGORITHM_SCRIPTS[@]}"; do
    # Extract model identifier (e.g., "5a" from "slurm_stage5a.sh")
    algorithm_name=$(basename "$script" .sh | sed 's/slurm_stage5//')
    log "Submitting Stage 5$algorithm_name..."
    
    # Build sbatch command with environment variables
    # Combine all env vars into a single --export argument
    if [ ${#ENV_VARS[@]} -gt 0 ]; then
        EXPORT_VARS="ALL"
        for env_var in "${ENV_VARS[@]}"; do
            EXPORT_VARS="$EXPORT_VARS,$env_var"
        done
        SBATCH_CMD=("sbatch" "--export=$EXPORT_VARS" "$script")
    else
        SBATCH_CMD=("sbatch" "--export=ALL" "$script")
    fi
    
    if SBATCH_OUTPUT=$("${SBATCH_CMD[@]}" 2>&1); then
        JOB_ID=$(echo "$SBATCH_OUTPUT" | grep -oE 'Submitted batch job [0-9]+' | grep -oE '[0-9]+' || echo "")
        if [ -n "$JOB_ID" ]; then
            SUBMITTED_JOBS+=("$JOB_ID")
            log "  ✓ Submitted $algorithm_name as job $JOB_ID"
        else
            log "  ✗ Failed to submit $algorithm_name (could not parse job ID)"
            log "    Output: $SBATCH_OUTPUT"
            FAILED_SUBMISSIONS+=("$algorithm_name")
        fi
    else
        log "  ✗ Failed to submit $algorithm_name"
        FAILED_SUBMISSIONS+=("$algorithm_name")
    fi
done

log ""
log "=========================================="
log "Submission Summary"
log "=========================================="
log "Total algorithms: ${#ALGORITHM_SCRIPTS[@]}"
log "Successfully submitted: ${#SUBMITTED_JOBS[@]}"
log "Failed submissions: ${#FAILED_SUBMISSIONS[@]}"

if [ ${#FAILED_SUBMISSIONS[@]} -gt 0 ]; then
    log ""
    log "✗ ERROR: Failed to submit the following algorithms:"
    for failed in "${FAILED_SUBMISSIONS[@]}"; do
        log "  - $failed"
    done
    exit 1
fi

if [ ${#SUBMITTED_JOBS[@]} -gt 0 ]; then
    log ""
    log "✓ Successfully submitted all algorithm jobs:"
    for i in "${!SUBMITTED_JOBS[@]}"; do
        algorithm_name=$(basename "${ALGORITHM_SCRIPTS[$i]}" .sh | sed 's/slurm_stage5//')
        log "  - Stage 5$algorithm_name: Job ID ${SUBMITTED_JOBS[$i]}"
    done
    log ""
    log "Monitor jobs with:"
    log "  squeue -u \$USER"
    log ""
    log "Check job status with:"
    for i in "${!SUBMITTED_JOBS[@]}"; do
        log "  squeue -j ${SUBMITTED_JOBS[$i]}"
    done
fi

log ""
log "=========================================="
log "STAGE 5 COORDINATOR COMPLETE"
log "=========================================="
log "All algorithm jobs have been submitted."
log "Stage 5 will complete when all algorithms finish training."
log "Note: Ensemble training should be run separately after all individual models complete."
log "=========================================="
