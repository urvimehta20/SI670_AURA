#!/bin/bash
#
# SLURM Batch Script for Stage 1c: Video Augmentation (Range 50-75%)
#
# Generates multiple augmented versions of each video using spatial and temporal
# transformations. Processes third quarter of videos (50-75%).
#
# Usage:
#   # Default: FVC_NUM_AUGMENTATIONS=10, FVC_DELETE_EXISTING=1
#   sbatch src/scripts/slurm_stage1c_augmentation.sh
#   
#   # Or override defaults:
#   FVC_NUM_AUGMENTATIONS=10 FVC_DELETE_EXISTING=1 sbatch src/scripts/slurm_stage1c_augmentation.sh
#
# Environment variables:
#   FVC_NUM_AUGMENTATIONS: Number of augmentations per video (default: 10)
#   FVC_STAGE1_OUTPUT_DIR: Output directory (default: data/augmented_videos)
#   FVC_DELETE_EXISTING: Set to 1/true/yes to delete existing augmentations (default: 0, preserves existing)
#   FVC_TOTAL_VIDEOS: Total number of videos (required for range calculation)
#

#SBATCH --job-name=fvc_stage1c_aug
#SBATCH --account=si670f25_class
#SBATCH --partition=standard
#SBATCH --time=4:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/stage1c_aug-%j.out
#SBATCH --error=logs/stage1c_aug-%j.err
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
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# ============================================================================
# System Information
# ============================================================================

log "=========================================="
log "STAGE 1c: VIDEO AUGMENTATION JOB (Range 50-75%)"
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
# Calculate Video Range (50-75%)
# ============================================================================

# Get total number of videos from CSV (BEFORE filtering)
# This counts all videos in the metadata, not just existing ones
TOTAL_VIDEOS=$(python -c "
import polars as pl
import sys
try:
    df = pl.read_csv('$DATA_CSV')
    # Count original videos only (before augmentation)
    # If there's an 'is_original' column, filter to True, otherwise count all
    if 'is_original' in df.columns:
        original_df = df.filter(pl.col('is_original') == True)
        print(original_df.height)
    else:
        # No is_original column means this is the input CSV with only original videos
        print(df.height)
except Exception as e:
    print('0', file=sys.stderr)
    sys.exit(1)
" 2>/dev/null || echo "0")

if [ "$TOTAL_VIDEOS" = "0" ] || [ -z "$TOTAL_VIDEOS" ]; then
    log "✗ ERROR: Could not determine total number of videos from $DATA_CSV"
    exit 1
fi

log "Total videos in dataset: $TOTAL_VIDEOS"

# Calculate range for 1c (50-75%)
START_IDX=$((TOTAL_VIDEOS / 2))
END_IDX=$((TOTAL_VIDEOS * 3 / 4))

log "Stage 1c: Processing videos [$START_IDX, $END_IDX) (third quarter)"

# ============================================================================
# Stage 1c Execution
# ============================================================================

log "=========================================="
log "Starting Stage 1c: Video Augmentation (50-75%)"
log "=========================================="

# Get number of augmentations from environment or use default (10 for parallel jobs)
NUM_AUGMENTATIONS="${FVC_NUM_AUGMENTATIONS:-10}"
log "Number of augmentations per video: $NUM_AUGMENTATIONS"

# Get output directory from environment or use default
OUTPUT_DIR="${FVC_STAGE1_OUTPUT_DIR:-data/augmented_videos}"
log "Output directory: $OUTPUT_DIR"

# Get delete-existing flag from environment (default: 1 for parallel jobs to avoid conflicts)
DELETE_EXISTING="${FVC_DELETE_EXISTING:-1}"
if [ "$DELETE_EXISTING" = "1" ] || [ "$DELETE_EXISTING" = "true" ] || [ "$DELETE_EXISTING" = "yes" ]; then
    DELETE_EXISTING_FLAG="--delete-existing"
    log "Delete existing augmentations: YES (will delete and regenerate)"
else
    DELETE_EXISTING_FLAG=""
    log "Delete existing augmentations: NO (will preserve existing)"
fi

STAGE1_START=$(date +%s)
LOG_FILE="$ORIG_DIR/logs/stage1c_augmentation_${SLURM_JOB_ID:-$$}.log"
mkdir -p "$(dirname "$LOG_FILE")"

# Change to project root for script execution
cd "$ORIG_DIR" || exit 1

# Ensure we're using the correct Python from venv
PYTHON_CMD=$(which python || echo "python")

log "Running Stage 1c augmentation script..."
log "Log file: $LOG_FILE"

# Build command with optional delete-existing flag
STAGE1_CMD=(
    "$PYTHON_CMD" "$ORIG_DIR/src/scripts/run_stage1_augmentation.py"
    --project-root "$ORIG_DIR"
    --num-augmentations "$NUM_AUGMENTATIONS"
    --output-dir "$OUTPUT_DIR"
    --start-idx "$START_IDX"
    --end-idx "$END_IDX"
)

# Add delete-existing flag if enabled
if [ -n "$DELETE_EXISTING_FLAG" ]; then
    STAGE1_CMD+=("$DELETE_EXISTING_FLAG")
fi

if "${STAGE1_CMD[@]}" 2>&1 | tee "$LOG_FILE"; then
    
    STAGE1_END=$(date +%s)
    STAGE1_DURATION=$((STAGE1_END - STAGE1_START))
    log "✓ Stage 1c completed successfully in ${STAGE1_DURATION}s ($(($STAGE1_DURATION / 60)) minutes)"
    log "Results saved to: $ORIG_DIR/$OUTPUT_DIR"
    log "Next step: Run Stage 1b, 1c, 1d in parallel, then Stage 2"
else
    STAGE1_END=$(date +%s)
    STAGE1_DURATION=$((STAGE1_END - STAGE1_START))
    log "✗ ERROR: Stage 1c failed after ${STAGE1_DURATION}s"
    log "Check log file: $LOG_FILE"
    exit 1
fi

log ""
log "============================================================"
log "STAGE 1c EXECUTION SUMMARY"
log "============================================================"
log "Execution time: ${STAGE1_DURATION}s ($(($STAGE1_DURATION / 60)) minutes)"
log "Video range: [$START_IDX, $END_IDX) of $TOTAL_VIDEOS total videos"
log "Output directory: $ORIG_DIR/$OUTPUT_DIR"
log "Log file: $LOG_FILE"
log "============================================================"

