#!/bin/bash
#
# SLURM Batch Script for Stage 5ALPHA: sklearn LogisticRegression Training
#
# Trains sklearn LogisticRegression with L1/L2/ElasticNet regularization (not MLP).
# Uses Stage 2/4 features with proper regularization.
#
# Usage:
#   sbatch src/scripts/slurm_stage5alpha.sh

#SBATCH --job-name=fvc_stage5alpha
#SBATCH --account=si670f25_class
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=1
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/stage5/stage5alpha-%j.out
#SBATCH --error=logs/stage5/stage5alpha-%j.err
#SBATCH --mail-user=santoshd@umich.edu,urvim@umich.edu,suzanef@umich.edu
#SBATCH --mail-type=FAIL,TIME_LIMIT,NODE_FAIL
#SBATCH --export=ALL

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

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTORCH_ALLOC_CONF="expandable_segments:true,max_split_size_mb:512"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export CUDA_LAUNCH_BLOCKING=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# ============================================================================
# System Information
# ============================================================================

log "=========================================="
log "STAGE 5ALPHA: sklearn LogisticRegression MODEL TRAINING JOB"
log "=========================================="
log "Host:        $(hostname)"
log "Date:        $(date -Is)"
log "SLURM_JOBID: ${SLURM_JOB_ID:-none}"
log "Working directory: $(pwd)"
log "Python:      $(which python3 2>/dev/null || which python 2>/dev/null || echo 'not found')"
log "Python version: $(python3 --version 2>&1 || python --version 2>&1 || echo 'unknown')"
log "=========================================="

# ============================================================================
# Verify Prerequisites
# ============================================================================

log "Verifying prerequisites..."

# Check critical Python packages
PREREQ_PACKAGES=("sklearn" "numpy" "scipy" "joblib" "matplotlib")

MISSING_PACKAGES=()

for pkg in "${PREREQ_PACKAGES[@]}"; do
    case "$pkg" in
        "sklearn")
            if ! python3 -c "import sklearn" 2>/dev/null; then
                MISSING_PACKAGES+=("$pkg")
            else
                log "✓ $pkg (sklearn) found"
            fi
            ;;
        *)
            if ! python3 -c "import $pkg" 2>/dev/null; then
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
if [ -z "$FEATURES_STAGE2" ]; then
    log "✗ ERROR: Stage 2 output not found in $ORIG_DIR/$FEATURES_STAGE2_DIR/"
    log "  Expected: features_metadata.arrow, features_metadata.parquet, or features_metadata.csv"
    log "  Run Stage 2 first: sbatch src/scripts/slurm_stage2_features.sh"
    exit 1
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
# Stage 4 is optional
if [ -z "$FEATURES_STAGE4" ]; then
    log "⚠ WARNING: Stage 4 output not found (optional)"
    log "  Expected: features_scaled_metadata.arrow, features_scaled_metadata.parquet, or features_scaled_metadata.csv"
    FEATURES_STAGE4=""
fi

log "✅ All prerequisites verified"

# ============================================================================
# Initialize Log File and Python Command
# ============================================================================

STAGE5_START=$(date +%s)
LOG_FILE="$ORIG_DIR/logs/stage5/stage5alpha_${SLURM_JOB_ID:-$$}.log"
mkdir -p "$(dirname "$LOG_FILE")"
touch "$LOG_FILE"

# Ensure log file is writable
if [ ! -w "$LOG_FILE" ]; then
    log "✗ ERROR: Log file is not writable: $LOG_FILE"
    exit 1
fi

# Write initial marker to log file to verify it's working
echo "==========================================" >> "$LOG_FILE"
echo "STAGE 5ALPHA LOG FILE INITIALIZED" >> "$LOG_FILE"
echo "Timestamp: $(date)" >> "$LOG_FILE"
echo "Log file: $LOG_FILE" >> "$LOG_FILE"
echo "==========================================" >> "$LOG_FILE"

cd "$ORIG_DIR" || exit 1
PYTHON_CMD=$(which python3 2>/dev/null || which python 2>/dev/null || echo "python3")
# Use unbuffered Python for immediate output
# Note: -u flag will be added when calling $PYTHON_CMD

# ============================================================================
# Stage 5ALPHA Execution
# ============================================================================

log "=========================================="
log "Starting Stage 5ALPHA: sklearn LogisticRegression Model Training"
log "=========================================="

N_SPLITS="${FVC_N_SPLITS:-5}"
OUTPUT_DIR="${FVC_STAGE5_OUTPUT_DIR:-data/stage5}"
DELETE_EXISTING="${FVC_DELETE_EXISTING:-0}"

log "Model type: sklearn LogisticRegression (L1/L2/ElasticNet regularization)"
log "K-fold splits: $N_SPLITS"
log "Output directory: $OUTPUT_DIR"
log "Delete existing: $DELETE_EXISTING"

# Validate Python script exists
PYTHON_SCRIPT="$ORIG_DIR/src/scripts/train_sklearn_logreg.py"
if [ ! -f "$PYTHON_SCRIPT" ]; then
    log "✗ ERROR: Python script not found: $PYTHON_SCRIPT"
    exit 1
fi

# Build command arguments
DELETE_FLAG=""
if [ "$DELETE_EXISTING" = "1" ] || [ "$DELETE_EXISTING" = "true" ] || [ "$DELETE_EXISTING" = "yes" ]; then
    DELETE_FLAG="--delete-existing"
    log "✓ Delete existing flag enabled: $DELETE_FLAG"
else
    log "⚠ Delete existing flag disabled (FVC_DELETE_EXISTING='${FVC_DELETE_EXISTING:-not set}', DELETE_EXISTING='$DELETE_EXISTING')"
fi

log "Running sklearn LogisticRegression training script..."
log "Log file: $LOG_FILE"
log "Command flags: DELETE_FLAG='$DELETE_FLAG'"
log "Python command: $PYTHON_CMD"
log "Python script: $PYTHON_SCRIPT"

# Verify Python script exists and is executable
if [ ! -f "$PYTHON_SCRIPT" ]; then
    log "✗ ERROR: Python script not found: $PYTHON_SCRIPT"
    echo "ERROR: Python script not found: $PYTHON_SCRIPT" >> "$LOG_FILE"
    exit 1
fi

if [ ! -x "$PYTHON_SCRIPT" ] && [ ! -r "$PYTHON_SCRIPT" ]; then
    log "✗ ERROR: Python script is not readable: $PYTHON_SCRIPT"
    echo "ERROR: Python script is not readable: $PYTHON_SCRIPT" >> "$LOG_FILE"
    exit 1
fi

FEATURES_STAGE4_ARG=""
if [ -n "$FEATURES_STAGE4" ]; then
    FEATURES_STAGE4_ARG="--features-stage4 $FEATURES_STAGE4"
fi

# Log the full command being executed
log "Executing: $PYTHON_CMD -u $PYTHON_SCRIPT --project-root $ORIG_DIR --scaled-metadata $SCALED_METADATA --features-stage2 $FEATURES_STAGE2 $FEATURES_STAGE4_ARG --output-dir $OUTPUT_DIR/sklearn_logreg --n-splits $N_SPLITS $DELETE_FLAG"
echo "Executing command..." >> "$LOG_FILE"
echo "Python: $PYTHON_CMD" >> "$LOG_FILE"
echo "Script: $PYTHON_SCRIPT" >> "$LOG_FILE"
echo "Arguments: --project-root $ORIG_DIR --scaled-metadata $SCALED_METADATA --features-stage2 $FEATURES_STAGE2 $FEATURES_STAGE4_ARG --output-dir $OUTPUT_DIR/sklearn_logreg --n-splits $N_SPLITS $DELETE_FLAG" >> "$LOG_FILE"
echo "==========================================" >> "$LOG_FILE"

# Use unbuffered Python and ensure output goes to both stdout and log file
if "$PYTHON_CMD" -u "$PYTHON_SCRIPT" \
    --project-root "$ORIG_DIR" \
    --scaled-metadata "$SCALED_METADATA" \
    --features-stage2 "$FEATURES_STAGE2" \
    $FEATURES_STAGE4_ARG \
    --output-dir "$OUTPUT_DIR/sklearn_logreg" \
    --n-splits "$N_SPLITS" \
    $DELETE_FLAG \
    2>&1 | tee -a "$LOG_FILE"; then
    
    # Log completion marker
    echo "==========================================" >> "$LOG_FILE"
    echo "Python script completed with exit code: $?" >> "$LOG_FILE"
    echo "==========================================" >> "$LOG_FILE"
    
    # Verify dataframe row count check passed
    # Check for data validation success (accounting for checkmark and colon)
    if grep -qE "(Data validation passed|✓ Data validation passed).*rows.*> 3000 required" "$LOG_FILE" 2>/dev/null; then
        log "✓ Data validation check passed (> 3000 rows)"
    elif grep -qE "(Data validation passed|✓ Data validation passed)" "$LOG_FILE" 2>/dev/null; then
        # Found validation message but pattern didn't match exactly - still consider it passed
        log "✓ Data validation check passed (message found in log)"
    elif grep -q "Insufficient data for training" "$LOG_FILE" 2>/dev/null; then
        log "✗ ERROR: Data validation failed - insufficient rows (need > 3000)"
        log "Check log file for details: $LOG_FILE"
        exit 1
    else
        log "⚠ WARNING: Could not verify data validation check in log file"
        log "⚠ This may indicate the script exited early or log buffering issues"
        log "⚠ Checking log file for errors: $LOG_FILE"
        # Check if there are any errors in the log
        ERROR_LINES=$(grep -iE "(error|exception|failed|traceback)" "$LOG_FILE" 2>/dev/null | head -5)
        if [ -n "$ERROR_LINES" ]; then
            log "⚠ Found potential errors in log file:"
            echo "$ERROR_LINES" | while IFS= read -r line; do
                log "  $line"
            done
        fi
    fi
    
    # Verify that output files were actually created (critical check)
    OUTPUT_DIR_FULL="$ORIG_DIR/$OUTPUT_DIR/sklearn_logreg"
    if [ -f "$OUTPUT_DIR_FULL/model.joblib" ] && [ -f "$OUTPUT_DIR_FULL/results.json" ]; then
        log "✓ Output files verified: model.joblib and results.json exist"
    elif [ -d "$OUTPUT_DIR_FULL" ]; then
        log "⚠ WARNING: Output directory exists but model files are missing"
        log "⚠ This suggests the script may have exited early"
        log "⚠ Output directory contents:"
        ls -lah "$OUTPUT_DIR_FULL" 2>/dev/null | head -10 | while IFS= read -r line; do
            log "  $line"
        done
        # Check if training completion message is in log
        if ! grep -qE "(Training complete|STAGE 5ALPHA TRAINING COMPLETED)" "$LOG_FILE" 2>/dev/null; then
            log "✗ ERROR: Training completion message not found in log - script likely exited early"
            exit 1
        fi
    else
        log "✗ ERROR: Output directory does not exist: $OUTPUT_DIR_FULL"
        log "✗ This indicates the script failed before creating output"
        exit 1
    fi
    
    STAGE5_END=$(date +%s)
    STAGE5_DURATION=$((STAGE5_END - STAGE5_START))
    log "✓ Stage 5ALPHA (sklearn LogisticRegression) completed successfully in ${STAGE5_DURATION}s ($((${STAGE5_DURATION} / 60)) minutes)"
    log "Results saved to: $ORIG_DIR/$OUTPUT_DIR/sklearn_logreg"
else
    STAGE5_END=$(date +%s)
    STAGE5_DURATION=$((STAGE5_END - STAGE5_START))
    log "✗ ERROR: Stage 5ALPHA (sklearn LogisticRegression) failed after ${STAGE5_DURATION}s"
    
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
log "STAGE 5ALPHA (sklearn LogisticRegression) EXECUTION SUMMARY"
log "============================================================"
log "Execution time: ${STAGE5_DURATION}s ($((${STAGE5_DURATION} / 60)) minutes)"
log "Model: sklearn LogisticRegression (L1/L2/ElasticNet)"
log "K-fold splits: $N_SPLITS"
log "Output directory: $ORIG_DIR/$OUTPUT_DIR/sklearn_logreg"
log "Log file: $LOG_FILE"
log "============================================================"
