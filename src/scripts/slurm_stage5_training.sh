#!/bin/bash
#
# SLURM Batch Script for Stage 5: Model Training
#
# Trains models using scaled videos and extracted features.
#
# Usage:
#   sbatch src/scripts/slurm_stage5_training.sh
#   sbatch --time=12:00:00 src/scripts/slurm_stage5_training.sh
#   sbatch --gpus=1 --mem=80G src/scripts/slurm_stage5_training.sh
#   FVC_MODELS="logistic_regression,svm" sbatch src/scripts/slurm_stage5_training.sh
#

#SBATCH --job-name=fvc_stage5_train
#SBATCH --account=stats_dept1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --mem=256G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1-00:00:00          # 1 day
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/stage5_train-%j.out
#SBATCH --error=logs/stage5_train-%j.err
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

# Set memory-optimized settings for 256GB RAM
if [ -z "${FVC_FIXED_SIZE:-}" ]; then
    export FVC_FIXED_SIZE=256
    echo "Using optimized resolution: FVC_FIXED_SIZE=256 (256x256) for 256GB RAM" >&2
fi

# ============================================================================
# Configuration and Setup
# ============================================================================

module purge
module load python3.11-anaconda/2024.02
module load cuda/12.1 || true

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
log "STAGE 5: MODEL TRAINING JOB"
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
PREREQ_PACKAGES=("torch" "torchvision" "polars" "numpy" "scikit-learn" "timm" "opencv-python" "av" "scipy" "joblib")

MISSING_PACKAGES=()
WARNING_PACKAGES=()

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
        "torch"|"torchvision"|"timm")
            # These packages may crash on import due to CUDA/GPU issues, but might work at runtime
            if timeout 5 python -c "import $pkg" 2>/dev/null; then
                log "✓ $pkg found"
            else
                if python -c "import pkg_resources; pkg_resources.get_distribution('$pkg')" 2>/dev/null || \
                   pip show "$pkg" >/dev/null 2>&1; then
                    log "⚠ $pkg is installed but import check failed (may work at runtime with GPU)"
                    WARNING_PACKAGES+=("$pkg")
                else
                    MISSING_PACKAGES+=("$pkg")
                fi
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

if [ ${#WARNING_PACKAGES[@]} -gt 0 ]; then
    log "⚠ WARNING: Import check failed for: ${WARNING_PACKAGES[*]}"
    log "  These packages are installed but import check failed (may be CUDA/GPU related)"
    log "  They may still work at runtime. Continuing..."
fi

# Verify previous stage outputs
SCALED_METADATA="${FVC_STAGE3_OUTPUT_DIR:-data/scaled_videos}/scaled_metadata.arrow"
SCALED_METADATA="$ORIG_DIR/$SCALED_METADATA"
if [ ! -f "$SCALED_METADATA" ]; then
    log "✗ ERROR: Stage 3 output not found: $SCALED_METADATA"
    log "  Run Stage 3 first: sbatch src/scripts/slurm_stage3_scaling.sh"
    exit 1
else
    log "✓ Stage 3 output found: $SCALED_METADATA"
fi

FEATURES_STAGE2="${FVC_STAGE2_OUTPUT_DIR:-data/features_stage2}/features_metadata.csv"
FEATURES_STAGE2="$ORIG_DIR/$FEATURES_STAGE2"
if [ ! -f "$FEATURES_STAGE2" ]; then
    log "✗ ERROR: Stage 2 output not found: $FEATURES_STAGE2"
    log "  Run Stage 2 first: sbatch src/scripts/slurm_stage2_features.sh"
    exit 1
else
    log "✓ Stage 2 output found: $FEATURES_STAGE2"
fi

FEATURES_STAGE4="${FVC_STAGE4_OUTPUT_DIR:-data/features_stage4}/features_scaled_metadata.arrow"
FEATURES_STAGE4="$ORIG_DIR/$FEATURES_STAGE4"
if [ ! -f "$FEATURES_STAGE4" ]; then
    log "✗ ERROR: Stage 4 output not found: $FEATURES_STAGE4"
    log "  Run Stage 4 first: sbatch src/scripts/slurm_stage4_scaled_features.sh"
    exit 1
else
    log "✓ Stage 4 output found: $FEATURES_STAGE4"
fi

log "✅ All prerequisites verified"

# ============================================================================
# Stage 5 Execution
# ============================================================================

log "=========================================="
log "Starting Stage 5: Model Training"
log "=========================================="

# Get model types from environment or use default
MODELS="${FVC_MODELS:-logistic_regression svm}"
MODELS_ARRAY=($MODELS)
NUM_FRAMES="${FVC_NUM_FRAMES:-8}"  # Optimized for 256GB RAM
N_SPLITS="${FVC_N_SPLITS:-5}"
OUTPUT_DIR="${FVC_STAGE5_OUTPUT_DIR:-data/training_results}"
USE_TRACKING="${FVC_USE_TRACKING:-true}"

log "Model types: ${MODELS_ARRAY[*]}"
log "K-fold splits: $N_SPLITS"
log "Number of frames: $NUM_FRAMES"
log "Output directory: $OUTPUT_DIR"
log "Experiment tracking: $USE_TRACKING"

STAGE5_START=$(date +%s)
LOG_FILE="$ORIG_DIR/logs/stage5_training_${SLURM_JOB_ID:-$$}.log"
mkdir -p "$(dirname "$LOG_FILE")"

cd "$ORIG_DIR" || exit 1
PYTHON_CMD=$(which python || echo "python")

log "Running Stage 5 training script..."
log "Log file: $LOG_FILE"

# Build command arguments
CMD_ARGS=(
    "$ORIG_DIR/src/scripts/run_stage5_training.py"
    --project-root "$ORIG_DIR"
    --scaled-metadata "${FVC_STAGE3_OUTPUT_DIR:-data/scaled_videos}/scaled_metadata.arrow"
    --features-stage2 "${FVC_STAGE2_OUTPUT_DIR:-data/features_stage2}/features_metadata.arrow"
    --features-stage4 "${FVC_STAGE4_OUTPUT_DIR:-data/features_stage4}/features_scaled_metadata.arrow"
    --model-types "${MODELS_ARRAY[@]}"
    --n-splits "$N_SPLITS"
    --num-frames "$NUM_FRAMES"
    --output-dir "$OUTPUT_DIR"
)

if [ "$USE_TRACKING" = "false" ]; then
    CMD_ARGS+=(--no-tracking)
fi

if "$PYTHON_CMD" "${CMD_ARGS[@]}" 2>&1 | tee "$LOG_FILE"; then
    
    STAGE5_END=$(date +%s)
    STAGE5_DURATION=$((STAGE5_END - STAGE5_START))
    log "✓ Stage 5 completed successfully in ${STAGE5_DURATION}s ($(($STAGE5_DURATION / 60)) minutes)"
    log "Results saved to: $ORIG_DIR/$OUTPUT_DIR"
    log "Models trained: ${MODELS_ARRAY[*]}"
else
    STAGE5_END=$(date +%s)
    STAGE5_DURATION=$((STAGE5_END - STAGE5_START))
    log "✗ ERROR: Stage 5 failed after ${STAGE5_DURATION}s"
    log "Check log file: $LOG_FILE"
    exit 1
fi

log ""
log "============================================================"
log "STAGE 5 EXECUTION SUMMARY"
log "============================================================"
log "Execution time: ${STAGE5_DURATION}s ($(($STAGE5_DURATION / 60)) minutes)"
log "Output directory: $ORIG_DIR/$OUTPUT_DIR"
log "Models trained: ${MODELS_ARRAY[*]}"
log "Log file: $LOG_FILE"
log "============================================================"

