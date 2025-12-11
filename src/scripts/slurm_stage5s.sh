#!/bin/bash
#
# SLURM Batch Script for Stage 5: slowfast_attention Training
#
# Trains slowfast_attention model using scaled videos and extracted features.
#
# Usage:
#   sbatch src/scripts/slurm_stage5_slowfast_attention.sh

#SBATCH --job-name=fvc_stage5_slowfast_attention
#SBATCH --account=si670f25_class
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=1
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/stage5/stage5s-%j.out
#SBATCH --error=logs/stage5/stage5s-%j.err
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
export PYTHONWARNINGS="ignore::UserWarning,ignore::DeprecationWarning,ignore::FutureWarning"

# Set memory-optimized settings
if [ -z "${FVC_FIXED_SIZE:-}" ]; then
    export FVC_FIXED_SIZE=256
    echo "Using optimized resolution: FVC_FIXED_SIZE=256 (256x256)" >&2
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
log "STAGE 5: slowfast_attention MODEL TRAINING JOB"
log "=========================================="
log "Host:        $(hostname)"
log "Date:        $(date -Is)"
log "SLURM_JOBID: ${SLURM_JOB_ID:-none}"
log "Working directory: $(pwd)"
log "Python:      $(which python 2>/dev/null || echo 'not found')"
log "Python version: $(python --version 2>&1 || echo 'unknown')"
log "=========================================="

# ============================================================================
# Prerequisites Check
# ============================================================================

log ""
log "Verifying prerequisites..."

# Check Python packages
if ! python -c "import torch" 2>/dev/null; then
    log "✗ ERROR: PyTorch not found"
    exit 1
fi
log "✓ PyTorch found"

if ! python -c "import timm" 2>/dev/null; then
    log "✗ ERROR: timm not found"
    exit 1
fi
log "✓ timm found"

# Check Stage outputs
SCALED_METADATA=""
FEATURES_STAGE2=""
FEATURES_STAGE4=""

for ext in arrow parquet csv; do
    if [ -f "$ORIG_DIR/data/scaled_videos/scaled_metadata.$ext" ]; then
        SCALED_METADATA="$ORIG_DIR/data/scaled_videos/scaled_metadata.$ext"
        break
    fi
done

for ext in arrow parquet csv; do
    if [ -f "$ORIG_DIR/data/features_stage2/features_metadata.$ext" ]; then
        FEATURES_STAGE2="$ORIG_DIR/data/features_stage2/features_metadata.$ext"
        break
    fi
done

for ext in arrow parquet csv; do
    if [ -f "$ORIG_DIR/data/features_stage4/features_scaled_metadata.$ext" ]; then
        FEATURES_STAGE4="$ORIG_DIR/data/features_stage4/features_scaled_metadata.$ext"
        break
    fi
done

if [ -z "$SCALED_METADATA" ]; then
    log "✗ ERROR: Stage 3 output not found"
    log "  Expected: data/scaled_videos/scaled_metadata.arrow, .parquet, or .csv"
    exit 1
fi
log "✓ Stage 3 output found: $SCALED_METADATA"

# Stage 2 and Stage 4 are optional for baseline models (logistic_regression, svm)
# They extract features directly from videos, not from pre-extracted features
if [ -z "$FEATURES_STAGE2" ]; then
    log "⚠ WARNING: Stage 2 output not found (optional for baseline models)"
    log "  Expected: data/features_stage2/features_metadata.arrow, .parquet, or .csv"
    log "  Baseline models will extract features directly from videos"
    FEATURES_STAGE2="$ORIG_DIR/data/features_stage2/features_metadata.arrow"  # Dummy path for script
else
    log "✓ Stage 2 output found: $FEATURES_STAGE2"
fi

if [ -z "$FEATURES_STAGE4" ]; then
    log "⚠ WARNING: Stage 4 output not found (optional for baseline models)"
    log "  Expected: data/features_stage4/features_scaled_metadata.arrow, .parquet, or .csv"
    log "  Baseline models will extract features directly from videos"
    FEATURES_STAGE4="$ORIG_DIR/data/features_stage4/features_scaled_metadata.arrow"  # Dummy path for script
else
    log "✓ Stage 4 output found: $FEATURES_STAGE4"
fi

log "✅ All prerequisites verified"

# ============================================================================
# Stage 5 Execution
# ============================================================================

log "=========================================="
log "Starting Stage 5: slowfast_attention Model Training"
log "=========================================="

MODEL_TYPE="slowfast_attention"
NUM_FRAMES="${FVC_NUM_FRAMES:-1000}"
N_SPLITS="${FVC_N_SPLITS:-5}"
OUTPUT_DIR="${FVC_STAGE5_OUTPUT_DIR:-data/stage5}"
USE_TRACKING="${FVC_USE_TRACKING:-true}"
DELETE_EXISTING="${FVC_DELETE_EXISTING:-0}"

log "Model type: $MODEL_TYPE (feature-based)"
log "Frames per video: $NUM_FRAMES (uniformly sampled from each scaled video)"
log "K-fold splits: $N_SPLITS"
log "Output directory: $OUTPUT_DIR"
log "Experiment tracking: $USE_TRACKING"
log "Delete existing: $DELETE_EXISTING"

STAGE5_START=$(date +%s)
LOG_FILE="$ORIG_DIR/logs/stage5/slowfast_attention_${SLURM_JOB_ID:-$$}.log"
mkdir -p "$(dirname "$LOG_FILE")"

cd "$ORIG_DIR" || exit 1
PYTHON_CMD=$(which python || echo "python")

# ============================================================================
# Import Validation: Verify all imports work before expensive training
# ============================================================================

log "=========================================="
log "Validating Stage 5 Imports"
log "=========================================="

VALIDATION_SCRIPT="$ORIG_DIR/src/scripts/validate_stage5_imports.py"
if [ ! -f "$VALIDATION_SCRIPT" ]; then
    log "⚠ WARNING: Import validation script not found: $VALIDATION_SCRIPT"
    log "  Skipping import validation (not recommended)"
else
    log "Running import validation (testing with dummy tensors)..."
    if "$PYTHON_CMD" "$VALIDATION_SCRIPT" 2>&1 | tee -a "$LOG_FILE"; then
        log "✓ Import validation passed"
    else
        VALIDATION_EXIT_CODE=${PIPESTATUS[0]}
        log "✗ ERROR: Import validation failed (exit code: $VALIDATION_EXIT_CODE)"
        log "  See validation output above for details"
        log "  Training will not proceed until imports are validated"
        log "  This catches import errors before expensive training jobs"
        exit $VALIDATION_EXIT_CODE
    fi
fi

# Validate Python script exists
PYTHON_SCRIPT="$ORIG_DIR/src/scripts/run_stage5_training.py"
if [ ! -f "$PYTHON_SCRIPT" ]; then
    log "✗ ERROR: Python script not found: $PYTHON_SCRIPT"
    exit 1
fi

# Build command arguments
DELETE_FLAG=""
if [ "$DELETE_EXISTING" = "1" ] || [ "$DELETE_EXISTING" = "true" ] || [ "$DELETE_EXISTING" = "yes" ]; then
    DELETE_FLAG="--delete-existing"
fi

TRACKING_FLAG=""
if [ "$USE_TRACKING" = "false" ]; then
    TRACKING_FLAG="--no-tracking"
fi

log "Running Stage 5 training script for $MODEL_TYPE..."
log "Log file: $LOG_FILE"

if "$PYTHON_CMD" "$PYTHON_SCRIPT" \
    --project-root "$ORIG_DIR" \
    --scaled-metadata "$SCALED_METADATA" \
    --features-stage2 "$FEATURES_STAGE2" \
    --features-stage4 "$FEATURES_STAGE4" \
    --model-types "$MODEL_TYPE" \
    --n-splits "$N_SPLITS" \
    --num-frames "$NUM_FRAMES" \
    --output-dir "$OUTPUT_DIR" \
    $DELETE_FLAG \
    $TRACKING_FLAG \
    2>&1 | tee "$LOG_FILE"; then
    
    STAGE5_END=$(date +%s)
    STAGE5_DURATION=$((STAGE5_END - STAGE5_START))
    log "✓ Stage 5 ($MODEL_TYPE) completed successfully in ${STAGE5_DURATION}s ($((${STAGE5_DURATION} / 60)) minutes)"
    log "Results saved to: $ORIG_DIR/$OUTPUT_DIR/$MODEL_TYPE"
else
    STAGE5_END=$(date +%s)
    STAGE5_DURATION=$((STAGE5_END - STAGE5_START))
    log "✗ ERROR: Stage 5 ($MODEL_TYPE) failed after ${STAGE5_DURATION}s"
    log "Check log file: $LOG_FILE"
    exit 1
fi

log ""
log "============================================================"
log "STAGE 5 (slowfast_attention) EXECUTION SUMMARY"
log "============================================================"
log "Execution time: ${STAGE5_DURATION}s ($((${STAGE5_DURATION} / 60)) minutes)"
log "Model: $MODEL_TYPE"
log "K-fold splits: $N_SPLITS"
log "Output directory: $ORIG_DIR/$OUTPUT_DIR/$MODEL_TYPE"
log "Log file: $LOG_FILE"
log "============================================================"
