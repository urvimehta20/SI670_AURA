#!/bin/bash
#
# SLURM Batch Script for Stage 1: Video Augmentation
#
# Generates multiple augmented versions of each video using spatial and temporal
# transformations.
#
# Usage:
#   sbatch src/scripts/slurm_stage1_augmentation.sh
#   sbatch --time=4:00:00 src/scripts/slurm_stage1_augmentation.sh
#   sbatch --mem=64G src/scripts/slurm_stage1_augmentation.sh
#

#SBATCH --job-name=fvc_stage1_aug
#SBATCH --account=eecs442f25_class
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=8:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/stage1_aug-%j.out
#SBATCH --error=logs/stage1_aug-%j.err
#SBATCH --mail-user=santoshd@umich.edu
#SBATCH --mail-type=FAIL,TIME_LIMIT,NODE_FAIL

set -euo pipefail
set -o errtrace
umask 077

# ============================================================================
# Environment Setup
# ============================================================================

# Unset macOS malloc warnings
unset MallocStackLogging || true
unset MallocStackLoggingNoCompact || true

# Suppress Python warnings
export PYTHONWARNINGS="ignore::UserWarning,ignore::DeprecationWarning,ignore::FutureWarning"

# Set extreme conservative memory settings
# Default fixed_size to 112x112 for maximum memory efficiency
# User can override by setting FVC_FIXED_SIZE in their environment
if [ -z "${FVC_FIXED_SIZE:-}" ]; then
    export FVC_FIXED_SIZE=112
    echo "Using extreme conservative resolution: FVC_FIXED_SIZE=112 (112x112)" >&2
else
    echo "Using custom resolution: FVC_FIXED_SIZE=${FVC_FIXED_SIZE}" >&2
fi

# ============================================================================
# Configuration and Setup
# ============================================================================

module purge
module load python3.11-anaconda/2024.02

# Directory setup
mkdir -p logs .pip-cache
export PIP_CACHE_DIR="$PWD/.pip-cache"
export WORK_DIR="${SLURM_TMPDIR:-$PWD}"
export ORIG_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
export VENV_DIR="$ORIG_DIR/venv"

# ============================================================================
# Logging Functions
# ============================================================================

log() {
    echo "$@" >&1
    echo "$@" >&2
    sync 2>/dev/null || true
}

# ============================================================================
# Virtual Environment Setup
# ============================================================================

log "Activating virtual environment: $VENV_DIR"
if [ ! -d "$VENV_DIR" ]; then
    log "✗ ERROR: Virtual environment not found: $VENV_DIR"
    exit 1
fi

source "$VENV_DIR/bin/activate"
export VIRTUAL_ENV_DISABLE_PROMPT=1

# Verify Python version
PYTHON_VERSION=$(python -c "import sys; print('{}.{}'.format(sys.version_info.major, sys.version_info.minor))" 2>/dev/null || echo "unknown")
if [ "$PYTHON_VERSION" != "3.11" ] && [ "$PYTHON_VERSION" != "unknown" ]; then
    log "⚠ WARNING: Python version is $PYTHON_VERSION, expected 3.11"
fi

# ============================================================================
# Environment Variables
# ============================================================================

export PYTORCH_ALLOC_CONF="expandable_segments:true,max_split_size_mb:128"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

# ============================================================================
# System Information
# ============================================================================

log "=========================================="
log "STAGE 1: VIDEO AUGMENTATION JOB"
log "=========================================="
log "Host:        $(hostname)"
log "Date:        $(date -Is)"
log "SLURM_JOBID: ${SLURM_JOB_ID:-none}"
log "Working directory: $(pwd)"
log "Python:      $(which python 2>/dev/null || echo 'not found')"
log "Python version: $(python --version 2>&1 || echo 'unknown')"
log "=========================================="

# ============================================================================
# Verify Prerequisites
# ============================================================================

log "Verifying prerequisites..."

# Check critical Python packages
PREREQ_PACKAGES=("polars" "numpy" "opencv-python" "av" "PIL")

MISSING_PACKAGES=()
for pkg in "${PREREQ_PACKAGES[@]}"; do
    case "$pkg" in
        "opencv-python")
            if ! python -c "import cv2" 2>/dev/null; then
                MISSING_PACKAGES+=("$pkg")
            else
                log "✓ $pkg (cv2) found"
            fi
            ;;
        "PIL")
            if ! python -c "import PIL" 2>/dev/null; then
                MISSING_PACKAGES+=("$pkg")
            else
                log "✓ $pkg found"
            fi
            ;;
        *)
            if ! python -c "import $pkg" 2>/dev/null; then
                MISSING_PACKAGES+=("$pkg")
            else
                log "✓ $pkg found"
            fi
            ;;
    esac
done

if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
    log "✗ ERROR: Missing required packages: ${MISSING_PACKAGES[*]}"
    log "  Install with: pip install -r requirements.txt"
    exit 1
fi

# Verify data files - check for FVC_dup.csv first, then video_index_input.csv
DATA_CSV=""
if [ -f "$ORIG_DIR/data/FVC_dup.csv" ]; then
    DATA_CSV="$ORIG_DIR/data/FVC_dup.csv"
    log "✓ Using FVC_dup.csv: $DATA_CSV"
elif [ -f "$ORIG_DIR/data/video_index_input.csv" ]; then
    DATA_CSV="$ORIG_DIR/data/video_index_input.csv"
    log "✓ Using video_index_input.csv: $DATA_CSV"
else
    log "✗ ERROR: No data CSV found"
    log "  Expected: $ORIG_DIR/data/FVC_dup.csv or $ORIG_DIR/data/video_index_input.csv"
    log "  Run setup script first: python src/setup_fvc_dataset.py"
    exit 1
fi

log "✅ All prerequisites verified"

# ============================================================================
# Stage 1 Execution
# ============================================================================

log "=========================================="
log "Starting Stage 1: Video Augmentation"
log "=========================================="

# Get number of augmentations from environment or use default
# Default to 10 augmentations per video for FVC_dup.csv
NUM_AUGMENTATIONS="${FVC_NUM_AUGMENTATIONS:-10}"
log "Number of augmentations per video: $NUM_AUGMENTATIONS"

# Get output directory from environment or use default
OUTPUT_DIR="${FVC_STAGE1_OUTPUT_DIR:-data/augmented_videos}"
log "Output directory: $OUTPUT_DIR"

# Get delete-existing flag from environment (default: false, preserves existing)
# Set FVC_DELETE_EXISTING=1 to delete existing augmentations before regenerating
DELETE_EXISTING="${FVC_DELETE_EXISTING:-0}"
if [ "$DELETE_EXISTING" = "1" ] || [ "$DELETE_EXISTING" = "true" ] || [ "$DELETE_EXISTING" = "yes" ]; then
    DELETE_EXISTING_FLAG="--delete-existing"
    log "Delete existing augmentations: YES (will delete and regenerate)"
else
    DELETE_EXISTING_FLAG=""
    log "Delete existing augmentations: NO (will preserve existing)"
fi

STAGE1_START=$(date +%s)
LOG_FILE="$ORIG_DIR/logs/stage1_augmentation_${SLURM_JOB_ID:-$$}.log"
mkdir -p "$(dirname "$LOG_FILE")"

# Change to project root for script execution
cd "$ORIG_DIR" || exit 1

# Ensure we're using the correct Python from venv
PYTHON_CMD=$(which python || echo "python")

log "Running Stage 1 augmentation script..."
log "Log file: $LOG_FILE"

# Build command with optional delete-existing flag
STAGE1_CMD=(
    "$PYTHON_CMD" "$ORIG_DIR/src/scripts/run_stage1_augmentation.py"
    --project-root "$ORIG_DIR"
    --num-augmentations "$NUM_AUGMENTATIONS"
    --output-dir "$OUTPUT_DIR"
)

# Add delete-existing flag if enabled
if [ -n "$DELETE_EXISTING_FLAG" ]; then
    STAGE1_CMD+=("$DELETE_EXISTING_FLAG")
fi

if "${STAGE1_CMD[@]}" 2>&1 | tee "$LOG_FILE"; then
    
    STAGE1_END=$(date +%s)
    STAGE1_DURATION=$((STAGE1_END - STAGE1_START))
    log "✓ Stage 1 completed successfully in ${STAGE1_DURATION}s (${STAGE1_DURATION} / 60 minutes)"
    log "Results saved to: $ORIG_DIR/$OUTPUT_DIR"
    log "Next step: Run Stage 2 with: sbatch src/scripts/slurm_stage2_features.sh"
else
    STAGE1_END=$(date +%s)
    STAGE1_DURATION=$((STAGE1_END - STAGE1_START))
    log "✗ ERROR: Stage 1 failed after ${STAGE1_DURATION}s"
    log "Check log file: $LOG_FILE"
    exit 1
fi

log ""
log "============================================================"
log "STAGE 1 EXECUTION SUMMARY"
log "============================================================"
log "Execution time: ${STAGE1_DURATION}s ($(($STAGE1_DURATION / 60)) minutes)"
log "Output directory: $ORIG_DIR/$OUTPUT_DIR"
log "Log file: $LOG_FILE"
log "============================================================"

