#!/bin/bash
#
# SLURM Batch Script for Stage 2: Handcrafted Feature Extraction
#
# Extracts handcrafted features from original videos (M features).
#
# Usage:
#   sbatch src/scripts/slurm_stage2_features.sh
#   sbatch --time=6:00:00 src/scripts/slurm_stage2_features.sh
#   sbatch --mem=80G src/scripts/slurm_stage2_features.sh
#

#SBATCH --job-name=fvc_stage2_feat
#SBATCH --account=stats_dept1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --mem=256G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1-00:00:00          # 1 day
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/stage2_feat-%j.out
#SBATCH --error=logs/stage2_feat-%j.err
#SBATCH --mail-user=santoshd@umich.edu
#SBATCH --mail-type=FAIL,TIME_LIMIT,NODE_FAIL

set -euo pipefail
set -o errtrace
umask 077

# ============================================================================
# Environment Setup
# ============================================================================

unset MallocStackLogging || true
unset MallocStackLoggingNoCompact || true
export PYTHONWARNINGS="ignore::UserWarning,ignore::DeprecationWarning,ignore::FutureWarning"

# ============================================================================
# Configuration and Setup
# ============================================================================

module purge
module load python3.11-anaconda/2024.02

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

# ============================================================================
# Environment Variables
# ============================================================================

export PYTORCH_ALLOC_CONF="expandable_segments:true,max_split_size_mb:512"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

# ============================================================================
# System Information
# ============================================================================

log "=========================================="
log "STAGE 2: FEATURE EXTRACTION JOB"
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
PREREQ_PACKAGES=("polars" "numpy" "opencv-python" "av" "scipy" "scikit-learn")

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
        "scikit-learn")
            if ! python -c "import sklearn" 2>/dev/null; then
                MISSING_PACKAGES+=("$pkg")
            else
                log "✓ $pkg (sklearn) found"
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
    exit 1
fi

# Verify Stage 1 output (try Arrow first, then Parquet, then CSV)
AUGMENTED_METADATA_DIR="${FVC_STAGE1_OUTPUT_DIR:-data/augmented_videos}"
AUGMENTED_METADATA=""
for ext in arrow parquet csv; do
    candidate="$ORIG_DIR/$AUGMENTED_METADATA_DIR/augmented_metadata.$ext"
    if [ -f "$candidate" ]; then
        AUGMENTED_METADATA="$candidate"
        log "✓ Stage 1 output found: $AUGMENTED_METADATA"
        break
    fi
done
if [ -z "$AUGMENTED_METADATA" ]; then
    log "✗ ERROR: Stage 1 output not found in $ORIG_DIR/$AUGMENTED_METADATA_DIR/"
    log "  Expected: augmented_metadata.arrow, augmented_metadata.parquet, or augmented_metadata.csv"
    log "  Run Stage 1 first: sbatch src/scripts/slurm_stage1_augmentation.sh"
    exit 1
fi

log "✅ All prerequisites verified"

# ============================================================================
# Stage 2 Execution
# ============================================================================

log "=========================================="
log "Starting Stage 2: Feature Extraction"
log "=========================================="

NUM_FRAMES="${FVC_NUM_FRAMES:-8}"  # Optimized for 256GB RAM
OUTPUT_DIR="${FVC_STAGE2_OUTPUT_DIR:-data/features_stage2}"
log "Number of frames: $NUM_FRAMES"
log "Output directory: $OUTPUT_DIR"

STAGE2_START=$(date +%s)
LOG_FILE="$ORIG_DIR/logs/stage2_features_${SLURM_JOB_ID:-$$}.log"
mkdir -p "$(dirname "$LOG_FILE")"

cd "$ORIG_DIR" || exit 1
PYTHON_CMD=$(which python || echo "python")

log "Running Stage 2 feature extraction script..."
log "Log file: $LOG_FILE"

if "$PYTHON_CMD" "$ORIG_DIR/src/scripts/run_stage2_features.py" \
    --project-root "$ORIG_DIR" \
    --augmented-metadata "$AUGMENTED_METADATA" \
    --num-frames "$NUM_FRAMES" \
    --output-dir "$OUTPUT_DIR" \
    2>&1 | tee "$LOG_FILE"; then
    
    STAGE2_END=$(date +%s)
    STAGE2_DURATION=$((STAGE2_END - STAGE2_START))
    log "✓ Stage 2 completed successfully in ${STAGE2_DURATION}s ($(($STAGE2_DURATION / 60)) minutes)"
    log "Results saved to: $ORIG_DIR/$OUTPUT_DIR"
    log "Next step: Run Stage 3 with: sbatch src/scripts/slurm_stage3_scaling.sh"
else
    STAGE2_END=$(date +%s)
    STAGE2_DURATION=$((STAGE2_END - STAGE2_START))
    log "✗ ERROR: Stage 2 failed after ${STAGE2_DURATION}s"
    log "Check log file: $LOG_FILE"
    exit 1
fi

log ""
log "============================================================"
log "STAGE 2 EXECUTION SUMMARY"
log "============================================================"
log "Execution time: ${STAGE2_DURATION}s ($(($STAGE2_DURATION / 60)) minutes)"
log "Output directory: $ORIG_DIR/$OUTPUT_DIR"
log "Log file: $LOG_FILE"
log "============================================================"

