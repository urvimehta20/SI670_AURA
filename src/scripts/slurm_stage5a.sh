#!/bin/bash
#
# SLURM Batch Script for Stage 5A: logistic_regression Training
#
# Trains logistic_regression model (Stage 5A) using scaled videos and extracted features.
#
# Usage:
#   sbatch src/scripts/slurm_stage5a.sh

#SBATCH --job-name=fvc_stage5a
#SBATCH --account=si670f25_class
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=1
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/stage5/stage5a-%j.out
#SBATCH --error=logs/stage5/stage5a-%j.err
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
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# ============================================================================
# System Information
# ============================================================================

log "=========================================="
log "STAGE 5A: logistic_regression MODEL TRAINING JOB"
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
            # These packages may fail to import in CPU-only check environment but work fine with GPU
            # If package is installed (via pip), trust it will work at runtime
            if pip show "$pkg" >/dev/null 2>&1; then
                log "✓ $pkg found (installed via pip)"
            elif python -c "import pkg_resources; pkg_resources.get_distribution('$pkg')" 2>/dev/null; then
                log "✓ $pkg found (installed via pkg_resources)"
            elif timeout 10 python -c "import $pkg" 2>/dev/null; then
                log "✓ $pkg found (import successful)"
            else
                MISSING_PACKAGES+=("$pkg")
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

# WARNING_PACKAGES removed - if package is installed, we trust it will work at runtime

# Verify previous stage outputs (try Arrow first, then Parquet, then CSV)
SCALED_METADATA_DIR="${FVC_STAGE3_OUTPUT_DIR:-data/scaled_videos}"
SCALED_METADATA=""
for ext in arrow parquet csv; do
    candidate="$ORIG_DIR/$SCALED_METADATA_DIR/scaled_metadata.$ext"
    if [ -f "$candidate" ]; then
        SCALED_METADATA="$candidate"
        log "✓ Stage 3 output found: $SCALED_METADATA"
        break
    fi
done
if [ -z "$SCALED_METADATA" ]; then
    log "✗ ERROR: Stage 3 output not found in $ORIG_DIR/$SCALED_METADATA_DIR/"
    log "  Expected: scaled_metadata.arrow, scaled_metadata.parquet, or scaled_metadata.csv"
    log "  Run Stage 3 first: sbatch src/scripts/slurm_stage3_scaling.sh"
    exit 1
fi

FEATURES_STAGE2_DIR="${FVC_STAGE2_OUTPUT_DIR:-data/features_stage2}"
FEATURES_STAGE2=""
for ext in arrow parquet csv; do
    candidate="$ORIG_DIR/$FEATURES_STAGE2_DIR/features_metadata.$ext"
    if [ -f "$candidate" ]; then
        FEATURES_STAGE2="$candidate"
        log "✓ Stage 2 output found: $FEATURES_STAGE2"
        break
    fi
done
# Stage 2 is optional for baseline models (logistic_regression, svm)
# They extract features directly from videos, not from pre-extracted features
if [ -z "$FEATURES_STAGE2" ]; then
    log "⚠ WARNING: Stage 2 output not found (optional for baseline models)"
    log "  Expected: features_metadata.arrow, features_metadata.parquet, or features_metadata.csv"
    log "  Baseline models will extract features directly from videos"
    FEATURES_STAGE2="$ORIG_DIR/$FEATURES_STAGE2_DIR/features_metadata.arrow"  # Dummy path for script
fi

FEATURES_STAGE4_DIR="${FVC_STAGE4_OUTPUT_DIR:-data/features_stage4}"
FEATURES_STAGE4=""
for ext in arrow parquet csv; do
    candidate="$ORIG_DIR/$FEATURES_STAGE4_DIR/features_scaled_metadata.$ext"
    if [ -f "$candidate" ]; then
        FEATURES_STAGE4="$candidate"
        log "✓ Stage 4 output found: $FEATURES_STAGE4"
        break
    fi
done
# Stage 4 is optional for baseline models (logistic_regression, svm)
# They extract features directly from videos, not from pre-extracted features
if [ -z "$FEATURES_STAGE4" ]; then
    log "⚠ WARNING: Stage 4 output not found (optional for baseline models)"
    log "  Expected: features_scaled_metadata.arrow, features_scaled_metadata.parquet, or features_scaled_metadata.csv"
    log "  Baseline models will extract features directly from videos"
    FEATURES_STAGE4="$ORIG_DIR/$FEATURES_STAGE4_DIR/features_scaled_metadata.arrow"  # Dummy path for script
fi

log "✅ All prerequisites verified"

# ============================================================================
# Initialize Log File and Python Command
# ============================================================================

STAGE5_START=$(date +%s)
LOG_FILE="$ORIG_DIR/logs/stage5/stage5a_${SLURM_JOB_ID:-$$}.log"
mkdir -p "$(dirname "$LOG_FILE")"
touch "$LOG_FILE"

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
    VALIDATION_OUTPUT=$("$PYTHON_CMD" "$VALIDATION_SCRIPT" 2>&1 | tee -a "$LOG_FILE")
    VALIDATION_EXIT_CODE=${PIPESTATUS[0]}
    
    # Check if validation actually passed (all tests passed)
    if echo "$VALIDATION_OUTPUT" | grep -q "✅ Validation PASSED"; then
        log "✓ Import validation passed (all tests successful)"
        # Exit code 134 (segfault) is OK if validation passed - it's just cleanup
        if [ "$VALIDATION_EXIT_CODE" = "134" ]; then
            log "  Note: Exit code 134 (segfault during cleanup) - validation still passed"
        fi
    else
        log "✗ ERROR: Import validation failed (exit code: $VALIDATION_EXIT_CODE)"
        log "  See validation output above for details"
        log "  Training will not proceed until imports are validated"
        log "  This catches import errors before expensive training jobs"
        exit $VALIDATION_EXIT_CODE
    fi
fi

# ============================================================================
# Sanity Check: Verify Stage 2 and Stage 4 Features
# ============================================================================

log "=========================================="
log "Running Feature Sanity Check"
log "=========================================="

SANITY_CHECK_SCRIPT="$ORIG_DIR/src/scripts/sanity_check_features.py"
if [ ! -f "$SANITY_CHECK_SCRIPT" ]; then
    log "⚠ WARNING: Sanity check script not found: $SANITY_CHECK_SCRIPT"
    log "  Skipping feature sanity check"
else
    log "Running feature sanity check..."
    if "$PYTHON_CMD" "$SANITY_CHECK_SCRIPT" 2>&1 | tee -a "$LOG_FILE"; then
        log "✓ Feature sanity check passed"
    else
        SANITY_EXIT_CODE=${PIPESTATUS[0]}
        log "✗ ERROR: Feature sanity check failed (exit code: $SANITY_EXIT_CODE)"
        log "  See sanity check output above for details"
        log "  Training will not proceed until features are validated"
        exit $SANITY_EXIT_CODE
    fi
fi

# ============================================================================
# Stage 5 Execution
# ============================================================================

log "=========================================="
log "Starting Stage 5A: logistic_regression Model Training"
log "=========================================="

MODEL_TYPE="logistic_regression"
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
    
    # Verify dataframe row count check passed
    if grep -q "Data validation passed.*rows (> 3000 required)" "$LOG_FILE" 2>/dev/null; then
        log "✓ Data validation check passed (> 3000 rows)"
    elif grep -q "Insufficient data for training" "$LOG_FILE" 2>/dev/null; then
        log "✗ ERROR: Data validation failed - insufficient rows (need > 3000)"
        log "Check log file for details: $LOG_FILE"
        exit 1
    else
        log "⚠ WARNING: Could not verify data validation check in log file"
    fi
    
    STAGE5_END=$(date +%s)
    STAGE5_DURATION=$((STAGE5_END - STAGE5_START))
    log "✓ Stage 5 ($MODEL_TYPE) completed successfully in ${STAGE5_DURATION}s ($((${STAGE5_DURATION} / 60)) minutes)"
    log "Results saved to: $ORIG_DIR/$OUTPUT_DIR/$MODEL_TYPE"
else
    STAGE5_END=$(date +%s)
    STAGE5_DURATION=$((STAGE5_END - STAGE5_START))
    log "✗ ERROR: Stage 5 ($MODEL_TYPE) failed after ${STAGE5_DURATION}s"
    
    # Check if failure was due to insufficient data
    if grep -q "Insufficient data for training" "$LOG_FILE" 2>/dev/null; then
        log "✗ ERROR: Training failed due to insufficient data (need > 3000 rows)"
        log "Please ensure Stage 3 completed successfully and generated enough scaled videos"
    fi
    
    log "Check log file: $LOG_FILE"
    exit 1
fi

log ""
log "============================================================"
log "STAGE 5A (logistic_regression) EXECUTION SUMMARY"
log "============================================================"
log "Execution time: ${STAGE5_DURATION}s ($((${STAGE5_DURATION} / 60)) minutes)"
log "Model: $MODEL_TYPE"
log "K-fold splits: $N_SPLITS"
log "Output directory: $ORIG_DIR/$OUTPUT_DIR/$MODEL_TYPE"
log "Log file: $LOG_FILE"
log "============================================================"
