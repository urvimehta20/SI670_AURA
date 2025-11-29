#!/bin/bash

#SBATCH --job-name=fvc_binary_classifier
#SBATCH --account=eecs442f25_class
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=8:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-user=santoshd@umich.edu
#SBATCH --mail-type=FAIL,TIME_LIMIT,NODE_FAIL

set -euo pipefail
set -o errtrace
umask 077

# Suppress Python warnings
export PYTHONWARNINGS="ignore::UserWarning,ignore::DeprecationWarning,ignore::FutureWarning"

# ============================================================================
# Configuration and Setup
# ============================================================================

module purge
module load python3.11-anaconda/2024.02
module load cuda/12.1 || true

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

# Check venv Python version BEFORE activating (venv is locked to the Python that created it)
if [ -f "$VENV_DIR/bin/python" ]; then
    VENV_PYTHON_VERSION=$("$VENV_DIR/bin/python" --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
    MODULE_PYTHON_VERSION=$(python3.11 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2 || echo "3.11")
    
    if [ "$VENV_PYTHON_VERSION" != "3.11" ] && [ "$VENV_PYTHON_VERSION" != "$MODULE_PYTHON_VERSION" ]; then
        log "✗ ERROR: venv was created with Python $VENV_PYTHON_VERSION, but module provides Python $MODULE_PYTHON_VERSION"
        log "  The venv is 'locked' to the Python version that created it."
        log "  Recreate the venv with:"
        log "    rm -rf venv"
        log "    module load python3.11-anaconda/2024.02"
        log "    python3.11 -m venv venv"
        log "    source venv/bin/activate"
        log "    pip install -r requirements.txt"
        exit 1
    fi
fi

source "$VENV_DIR/bin/activate"
export VIRTUAL_ENV_DISABLE_PROMPT=1

# Verify venv Python matches module Python after activation
ACTIVE_PYTHON=$(python --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
if [ "$ACTIVE_PYTHON" != "3.11" ]; then
    log "⚠ WARNING: Active Python version is $ACTIVE_PYTHON, expected 3.11"
    log "  This may cause compatibility issues."
fi

mkdir -p "$WORK_DIR/runs" "$WORK_DIR/logs"
ln -snf "$WORK_DIR/runs" runs 2>/dev/null || true

# ============================================================================
# Environment Variables for Performance
# ============================================================================

# Disable macOS-style malloc stack logging flags if they somehow leak into
# this environment; on some systems they can cause noisy warnings.
unset MallocStackLogging || true
unset MallocStackLoggingNoCompact || true

export PYTORCH_ALLOC_CONF="expandable_segments:true,max_split_size_mb:128"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

# ============================================================================
# System Information
# ============================================================================

set +u
log "=========================================="
log "JOB STARTUP INFORMATION"
log "=========================================="
log "Host:        $(hostname)"
log "Date:        $(date -Is)"
log "SLURM_JOBID: ${SLURM_JOB_ID:-none}"
log "Working directory: $(pwd)"
log "Python:      $(which python 2>/dev/null || echo 'not found')"
log "Python version: $(python --version 2>&1 || echo 'unknown')"
log "=========================================="
set -u

# Verify Python version (should be 3.11 from module, but check anyway)
PYTHON_VERSION=$(python -c "import sys; print('{}.{}'.format(sys.version_info.major, sys.version_info.minor))" 2>/dev/null || echo "unknown")
if [ "$PYTHON_VERSION" != "3.11" ] && [ "$PYTHON_VERSION" != "unknown" ]; then
    log "⚠ WARNING: Python version is $PYTHON_VERSION, expected 3.11"
    log "  This may cause compatibility issues. Ensure module python3.11-anaconda/2024.02 is loaded."
fi

# ============================================================================
# Verify Environment and Prerequisites
# ============================================================================

log "Verifying Python environment and prerequisites..."

# Check critical Python packages required for MLOps pipeline
PREREQ_PACKAGES=(
    "torch"
    "torchvision"
    "polars"
    "numpy"
    "pandas"
    "scikit-learn"
    "timm"
    "opencv-python"
    "av"
    "tqdm"
    "scipy"
    "joblib"
)

MISSING_PACKAGES=()
for pkg in "${PREREQ_PACKAGES[@]}"; do
    # Handle package name variations (e.g., opencv-python -> cv2)
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
    log "  Install with: pip install -r requirements.txt"
    exit 1
fi

# Check for papermill and ipykernel (for notebook execution if needed)
if python -c "import papermill" 2>/dev/null; then
    log "✓ papermill found"
else
    log "⚠ WARNING: papermill not found (only needed for notebook execution)"
fi

if python -c "import ipykernel" 2>/dev/null; then
    log "✓ ipykernel found"
else
    log "⚠ WARNING: ipykernel not found (only needed for notebook execution)"
fi

# Verify data files exist
log "Verifying data files..."
DATA_CSV="$ORIG_DIR/data/video_index_input.csv"
if [ ! -f "$DATA_CSV" ]; then
    log "✗ ERROR: Data CSV not found: $DATA_CSV"
    log "  Run setup script first: python src/setup_fvc_dataset.py"
    exit 1
else
    log "✓ Data CSV found: $DATA_CSV"
fi

# Check if videos directory exists
if [ ! -d "$ORIG_DIR/videos" ]; then
    log "⚠ WARNING: videos/ directory not found. Videos may not be accessible."
else
    log "✓ videos/ directory found"
fi

log "✅ All prerequisites verified"

# ============================================================================
# Jupyter Kernel Setup
# ============================================================================

KNAME="fvc-binary-classifier-${SLURM_JOB_ID:-$$}"
KERNEL_ARG="python3"  # Default fallback

log "Installing Jupyter kernel: ${KNAME}..."
# ipykernel should be installed by now (checked/installed in Verify Environment section)
if python -m ipykernel install --user --name "${KNAME}" --display-name "FVC Binary Classifier (${KNAME})" 2>&1; then
    log "✓ Kernel installed successfully"
    KERNEL_ARG="${KNAME}"
    trap 'jupyter kernelspec remove -y "${KNAME}" 2>/dev/null || true' EXIT
else
    log "⚠ WARNING: Kernel installation failed, using default python3"
    log "  This is usually fine - papermill can use the default python3 kernel"
fi

# ============================================================================
# Notebook Configuration
# ============================================================================

NOTEBOOK_IN="src/fvc_binary_classifier.ipynb"
STAMP="$(date +%Y%m%d-%H%M%S)"
NOTEBOOK_OUT="$WORK_DIR/runs/fvc_binary_classifier_executed_${STAMP}.ipynb"

# ============================================================================
# Pre-flight Validation
# ============================================================================

log "Running pre-flight validation..."
VALIDATION_ERRORS=0

# Check for notebook
if [ ! -f "$ORIG_DIR/$NOTEBOOK_IN" ]; then
    log "✗ ERROR: Notebook not found: $NOTEBOOK_IN"
    VALIDATION_ERRORS=$((VALIDATION_ERRORS + 1))
else
    log "✓ Notebook found: $NOTEBOOK_IN"
fi

if [ $VALIDATION_ERRORS -gt 0 ]; then
    log "❌ PRE-FLIGHT VALIDATION FAILED: $VALIDATION_ERRORS error(s) found"
    exit 1
fi

log "✅ Pre-flight validation passed!"

# ============================================================================
# File Copying (if using SLURM_TMPDIR)
# ============================================================================

if [ -n "${SLURM_TMPDIR:-}" ] && [ "$WORK_DIR" != "$ORIG_DIR" ]; then
    log "Using SLURM_TMPDIR: $SLURM_TMPDIR"
    mkdir -p "$WORK_DIR/src"
    
    cp -v "$ORIG_DIR/$NOTEBOOK_IN" "$WORK_DIR/$NOTEBOOK_IN" 2>/dev/null || exit 1
    cd "$WORK_DIR"
fi

# ============================================================================
# Cleanup Previous Instances (Fresh Run)
# ============================================================================

log "=== Cleaning Previous Runs for Fresh Start ==="
# Kill any previous instances
pkill -f "papermill.*fvc_binary_classifier" 2>/dev/null || true
pkill -f "ipykernel.*fvc-binary-classifier" 2>/dev/null || true

# Delete runs/, logs/, models/, and intermediate_data/ for completely fresh run
# Preserve: archive/, data/video_index_input.csv, videos/
log "Deleting runs/, logs/, models/, and intermediate_data/ directories..."

if [ -d "$ORIG_DIR/runs" ]; then
    rm -rf "$ORIG_DIR/runs"
    log "✓ Deleted $ORIG_DIR/runs"
fi

if [ -d "$ORIG_DIR/logs" ]; then
    rm -rf "$ORIG_DIR/logs"
    log "✓ Deleted $ORIG_DIR/logs"
fi

if [ -d "$ORIG_DIR/models" ]; then
    rm -rf "$ORIG_DIR/models"
    log "✓ Deleted $ORIG_DIR/models"
fi

if [ -d "$ORIG_DIR/intermediate_data" ]; then
    rm -rf "$ORIG_DIR/intermediate_data"
    log "✓ Deleted $ORIG_DIR/intermediate_data"
fi

# Recreate empty directories
mkdir -p "$ORIG_DIR/runs" "$ORIG_DIR/logs" "$ORIG_DIR/models"
log "✓ Created fresh runs/, logs/, models/ directories"

# Also clean WORK_DIR if different
if [ "$WORK_DIR" != "$ORIG_DIR" ]; then
    if [ -d "$WORK_DIR/runs" ]; then
        rm -rf "$WORK_DIR/runs"
    fi
    if [ -d "$WORK_DIR/logs" ]; then
        rm -rf "$WORK_DIR/logs"
    fi
    if [ -d "$WORK_DIR/intermediate_data" ]; then
        rm -rf "$WORK_DIR/intermediate_data"
    fi
    mkdir -p "$WORK_DIR/runs" "$WORK_DIR/logs"
fi

sleep 2
log "✓ Fresh cleanup completed"

# ============================================================================
# Notebook Execution (or MLOps Pipeline)
# ============================================================================

# Check if we should use MLOps pipeline instead of notebook
# Default to MLOps pipeline for better OOM handling, K-fold CV, and checkpointing
USE_MLOPS_PIPELINE="${USE_MLOPS_PIPELINE:-true}"

if [ "$USE_MLOPS_PIPELINE" = "true" ]; then
    log "Using MLOps pipeline (run_mlops_pipeline.py)"
    log "Features: K-fold CV, aggressive GC, OOM handling, per-stage checkpointing"
    log "Pipeline order:"
    log "  1. Load data (with duplicate videos from FVC_dup.csv)"
    log "  2. Download/verify pretrained models (prerequisite)"
    log "  3. Create balanced k-fold splits (stratified, class-balanced)"
    log "  4. Generate shared augmentations (BEFORE models, cached globally)"
    log "  5. Train all models sequentially (with shared data/augmentations)"
    log ""
    log "Note: Augmentations are generated ONCE and reused across all models and runs"
    log "      (cached in intermediate_data/augmented_clips/shared/)"
    
    PIPELINE_START=$(date +%s)
    LOG_FILE="$WORK_DIR/logs/fvc_binary_classifier_run.log"
    mkdir -p "$(dirname "$LOG_FILE")"
    
    # Change to project root for script execution
    cd "$ORIG_DIR" || exit 1
    
    # Ensure we're using the correct Python from venv
    PYTHON_CMD=$(which python || echo "python")
    
    if "$PYTHON_CMD" "$ORIG_DIR/src/run_mlops_pipeline.py" 2>&1 | tee "$LOG_FILE"; then
        PIPELINE_END=$(date +%s)
        PIPELINE_DURATION=$((PIPELINE_END - PIPELINE_START))
        log "✓ MLOps pipeline completed in ${PIPELINE_DURATION}s"
        NOTEBOOK_DURATION=$PIPELINE_DURATION
    else
        PIPELINE_END=$(date +%s)
        PIPELINE_DURATION=$((PIPELINE_END - PIPELINE_START))
        log "✗ ERROR: MLOps pipeline failed after ${PIPELINE_DURATION}s!"
        log "Check log file: $LOG_FILE"
        exit 1
    fi
else
    log "Running papermill -> ${NOTEBOOK_OUT}"
    log "Notebook: ${NOTEBOOK_IN}"
    log "Kernel: ${KERNEL_ARG}"
    NOTEBOOK_START=$(date +%s)

# Run with proper logging - capture both stdout and stderr
LOG_FILE="$WORK_DIR/logs/fvc_binary_classifier_run.log"
mkdir -p "$(dirname "$LOG_FILE")"

if ! papermill "${NOTEBOOK_IN}" "${NOTEBOOK_OUT}" \
    --kernel "${KERNEL_ARG}" \
    --log-output \
    --execution-timeout 14400 \
    2>&1 | tee "$LOG_FILE"; then
    NOTEBOOK_END=$(date +%s)
    NOTEBOOK_DURATION=$((NOTEBOOK_END - NOTEBOOK_START))
    log "✗ ERROR: Papermill failed after ${NOTEBOOK_DURATION}s!"
    log "Check log file: $LOG_FILE"
    exit 1
fi

NOTEBOOK_END=$(date +%s)
NOTEBOOK_DURATION=$((NOTEBOOK_END - NOTEBOOK_START))
log "✓ Notebook execution completed in ${NOTEBOOK_DURATION}s"
fi

# ============================================================================
# Copy Results Back
# ============================================================================

if [ -n "${SLURM_TMPDIR:-}" ] && [ "$WORK_DIR" != "$ORIG_DIR" ]; then
    log "Copying results back..."
    
    # Copy logs
    if [ -f "$LOG_FILE" ]; then
        mkdir -p "$ORIG_DIR/logs"
        cp -v "$LOG_FILE" "$ORIG_DIR/logs/" 2>/dev/null || true
    fi
    
    # Copy executed notebook
    if [ -n "${SLURM_JOB_ID:-}" ] && [ -f "${NOTEBOOK_OUT}" ]; then
        RUN_OUT_DIR="$ORIG_DIR/runs/run_${SLURM_JOB_ID}"
        mkdir -p "$RUN_OUT_DIR"
        cp -v "${NOTEBOOK_OUT}" "$RUN_OUT_DIR/" 2>/dev/null || true
    fi
fi

log ""
log "============================================================"
log "EXECUTION SUMMARY"
log "============================================================"
if [ "$USE_MLOPS_PIPELINE" = "true" ]; then
    log "MLOps pipeline execution time: ${NOTEBOOK_DURATION}s"
else
    log "Notebook execution time: ${NOTEBOOK_DURATION}s"
fi
log "============================================================"

