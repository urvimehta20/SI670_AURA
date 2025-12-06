#!/bin/bash
#
# SLURM Batch Script for Stage 5: Model Training
#
# Trains models using scaled videos and extracted features.
#
# Usage:
#   sbatch src/scripts/slurm_stage5_training.sh
#   sbatch --nodes=2 src/scripts/slurm_stage5_training.sh  # Use 2 nodes
#   sbatch --nodes=4 --time=12:00:00 src/scripts/slurm_stage5_training.sh  # Use 4 nodes
#   FVC_MODELS="logistic_regression,svm" sbatch src/scripts/slurm_stage5_training.sh
#

#SBATCH --job-name=fvc_stage5_train
#SBATCH --account=eecs442f25_class
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem=80G
#SBATCH --nodes=1-4
#SBATCH --ntasks-per-node=1
#SBATCH --time=8:00:00
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

# Set memory-optimized settings for multi-node execution
if [ -z "${FVC_FIXED_SIZE:-}" ]; then
    export FVC_FIXED_SIZE=256
    echo "Using optimized resolution: FVC_FIXED_SIZE=256 (256x256) for multi-node execution" >&2
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
log "SLURM_NODELIST: ${SLURM_NODELIST:-none}"
log "SLURM_NNODES: ${SLURM_NNODES:-1}"
log "SLURM_JOB_NUM_NODES: ${SLURM_JOB_NUM_NODES:-1}"
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
if [ -z "$FEATURES_STAGE4" ]; then
    log "✗ ERROR: Stage 4 output not found in $ORIG_DIR/$FEATURES_STAGE4_DIR/"
    log "  Expected: features_scaled_metadata.arrow, features_scaled_metadata.parquet, or features_scaled_metadata.csv"
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
NUM_FRAMES="${FVC_NUM_FRAMES:-8}"  # Optimized for 80GB RAM
N_SPLITS="${FVC_N_SPLITS:-5}"
OUTPUT_DIR="${FVC_STAGE5_OUTPUT_DIR:-data/training_results}"
USE_TRACKING="${FVC_USE_TRACKING:-true}"
DELETE_EXISTING="${FVC_DELETE_EXISTING:-0}"

log "Model types: ${MODELS_ARRAY[*]}"
log "K-fold splits: $N_SPLITS"
log "Number of frames: $NUM_FRAMES"
log "Output directory: $OUTPUT_DIR"
log "Experiment tracking: $USE_TRACKING"
log "Delete existing: $DELETE_EXISTING"

# Multi-node execution: distribute models across nodes (one model per node)
NUM_NODES="${SLURM_JOB_NUM_NODES:-1}"
NUM_MODELS=${#MODELS_ARRAY[@]}

if [ "$NUM_NODES" -gt "$NUM_MODELS" ]; then
    log "⚠ WARNING: More nodes ($NUM_NODES) than models ($NUM_MODELS). Some nodes will be idle."
    NUM_NODES_TO_USE=$NUM_MODELS
else
    NUM_NODES_TO_USE=$NUM_NODES
fi

log "Number of nodes: $NUM_NODES_TO_USE"
log "Number of models: $NUM_MODELS"

if [ "$NUM_NODES_TO_USE" -lt "$NUM_MODELS" ]; then
    log "⚠ WARNING: More models ($NUM_MODELS) than nodes ($NUM_NODES_TO_USE). Some models will run sequentially."
fi

STAGE5_START=$(date +%s)
COMBINED_LOG_FILE="$ORIG_DIR/logs/stage5_training_combined_${SLURM_JOB_ID:-$$}.log"
mkdir -p "$(dirname "$COMBINED_LOG_FILE")"

cd "$ORIG_DIR" || exit 1
PYTHON_CMD=$(which python || echo "python")

# Helper function to verify model completion
verify_model_completion() {
    local model_type=$1
    local output_dir=$2
    
    local model_dir="$output_dir/$model_type"
    local completion_file="$model_dir/training_complete.pt"
    
    if [ -f "$completion_file" ]; then
        log "✓ Model $model_type training verified complete"
        return 0
    else
        log "✗ ERROR: Model $model_type training incomplete (no training_complete.pt found)"
        return 1
    fi
}

# Multi-node execution: distribute models across nodes
NODE_FAILURES=0
NODE_SUCCESSES=0

for NODE_ID in $(seq 0 $((NUM_NODES_TO_USE - 1))); do
    if [ $NODE_ID -ge $NUM_MODELS ]; then
        log "Node $NODE_ID: No model assigned (more nodes than models)"
        continue
    fi
    
    MODEL_TYPE="${MODELS_ARRAY[$NODE_ID]}"
    
    log "=========================================="
    log "Node $NODE_ID: Training model: $MODEL_TYPE"
    log "=========================================="
    
    NODE_LOG_FILE="$ORIG_DIR/logs/stage5_train-${SLURM_JOB_ID:-$$}_node${NODE_ID}_${MODEL_TYPE}.log"
    mkdir -p "$(dirname "$NODE_LOG_FILE")"
    
    # Build command arguments
    DELETE_FLAG=""
    if [ "$DELETE_EXISTING" = "1" ] || [ "$DELETE_EXISTING" = "true" ] || [ "$DELETE_EXISTING" = "yes" ]; then
        DELETE_FLAG="--delete-existing"
    fi
    
    TRACKING_FLAG=""
    if [ "$USE_TRACKING" = "false" ]; then
        TRACKING_FLAG="--no-tracking"
    fi
    
    # Use srun to launch on specific node
    if [ "$NUM_NODES" -gt 1 ]; then
        NODE_HOSTNAME=$(scontrol show hostnames ${SLURM_JOB_NODELIST} | sed -n "$((NODE_ID + 1))p")
        log "Launching on node $NODE_ID (hostname: $NODE_HOSTNAME) for model $MODEL_TYPE"
        
        if srun --nodes=1 --ntasks=1 --nodelist="$NODE_HOSTNAME" \
            "$PYTHON_CMD" "$ORIG_DIR/src/scripts/run_stage5_training.py" \
            --project-root "$ORIG_DIR" \
            --scaled-metadata "$SCALED_METADATA" \
            --features-stage2 "$FEATURES_STAGE2" \
            --features-stage4 "$FEATURES_STAGE4" \
            --model-types "$MODEL_TYPE" \
            --model-idx "$NODE_ID" \
            --n-splits "$N_SPLITS" \
            --num-frames "$NUM_FRAMES" \
            --output-dir "$OUTPUT_DIR" \
            $DELETE_FLAG \
            $TRACKING_FLAG \
            2>&1 | tee "$NODE_LOG_FILE"; then
            log "✓ Node $NODE_ID (model $MODEL_TYPE) completed successfully"
            NODE_SUCCESSES=$((NODE_SUCCESSES + 1))
        else
            log "✗ ERROR: Node $NODE_ID (model $MODEL_TYPE) failed"
            NODE_FAILURES=$((NODE_FAILURES + 1))
        fi
    else
        # Single node execution - train all models sequentially
        log "Single node execution - training all models sequentially"
        for MODEL_IDX in $(seq 0 $((${#MODELS_ARRAY[@]} - 1))); do
            MODEL="${MODELS_ARRAY[$MODEL_IDX]}"
            log "Training model $MODEL_IDX/$NUM_MODELS: $MODEL"
            
            MODEL_LOG_FILE="$ORIG_DIR/logs/stage5_train-${SLURM_JOB_ID:-$$}_node0_${MODEL}.log"
            
            if "$PYTHON_CMD" "$ORIG_DIR/src/scripts/run_stage5_training.py" \
                --project-root "$ORIG_DIR" \
                --scaled-metadata "$SCALED_METADATA" \
                --features-stage2 "$FEATURES_STAGE2" \
                --features-stage4 "$FEATURES_STAGE4" \
                --model-types "$MODEL" \
                --model-idx "$MODEL_IDX" \
                --n-splits "$N_SPLITS" \
                --num-frames "$NUM_FRAMES" \
                --output-dir "$OUTPUT_DIR" \
                $DELETE_FLAG \
                $TRACKING_FLAG \
                2>&1 | tee "$MODEL_LOG_FILE"; then
                log "✓ Model $MODEL completed successfully"
                NODE_SUCCESSES=$((NODE_SUCCESSES + 1))
            else
                log "✗ ERROR: Model $MODEL failed"
                NODE_FAILURES=$((NODE_FAILURES + 1))
            fi
            
            # Append to combined log
            echo "" >> "$COMBINED_LOG_FILE"
            echo "=== Model $MODEL Log ===" >> "$COMBINED_LOG_FILE"
            cat "$MODEL_LOG_FILE" >> "$COMBINED_LOG_FILE"
        done
        break  # Exit node loop for single node
    fi
    
    # Append node log to combined log
    echo "" >> "$COMBINED_LOG_FILE"
    echo "=== Node $NODE_ID (Model $MODEL_TYPE) Log ===" >> "$COMBINED_LOG_FILE"
    cat "$NODE_LOG_FILE" >> "$COMBINED_LOG_FILE"
done

# Verify all models completed
ALL_MODELS_COMPLETE=1
for MODEL_TYPE in "${MODELS_ARRAY[@]}"; do
    if ! verify_model_completion "$MODEL_TYPE" "$OUTPUT_DIR"; then
        ALL_MODELS_COMPLETE=0
    fi
done

if [ "$NODE_FAILURES" -gt 0 ] || [ "$ALL_MODELS_COMPLETE" -eq 0 ]; then
    log "✗ ERROR: Some models failed or incomplete"
    exit 1
fi

if [ "$NODE_SUCCESSES" -ge "$NUM_MODELS" ]; then
    
    STAGE5_END=$(date +%s)
    STAGE5_DURATION=$((STAGE5_END - STAGE5_START))
    log "✓ Stage 5 completed successfully in ${STAGE5_DURATION}s ($(($STAGE5_DURATION / 60)) minutes)"
    log "Results saved to: $ORIG_DIR/$OUTPUT_DIR"
    log "Models trained: ${MODELS_ARRAY[*]}"
    log "All $NUM_MODELS model(s) completed successfully"
else
    STAGE5_END=$(date +%s)
    STAGE5_DURATION=$((STAGE5_END - STAGE5_START))
    log "✗ ERROR: Stage 5 failed after ${STAGE5_DURATION}s"
    log "Check log files: $ORIG_DIR/logs/stage5_train-${SLURM_JOB_ID:-$$}_node*.log"
    exit 1
fi

log ""
log "============================================================"
log "STAGE 5 EXECUTION SUMMARY"
log "============================================================"
log "Execution time: ${STAGE5_DURATION}s ($(($STAGE5_DURATION / 60)) minutes)"
log "Nodes used: $NUM_NODES_TO_USE"
log "Models trained: ${MODELS_ARRAY[*]}"
log "Output directory: $ORIG_DIR/$OUTPUT_DIR"
log "Combined log file: $COMBINED_LOG_FILE"
log "Node log files: $ORIG_DIR/logs/stage5_train-${SLURM_JOB_ID:-$$}_node*.log"
log "============================================================"

