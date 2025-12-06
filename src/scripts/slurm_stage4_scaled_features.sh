#!/bin/bash
#
# SLURM Batch Script for Stage 4: Downscaled Feature Extraction
#
# Extracts additional features from scaled videos (P features).
#
# Usage:
#   sbatch src/scripts/slurm_stage4_scaled_features.sh
#   sbatch --nodes=2 src/scripts/slurm_stage4_scaled_features.sh  # Use 2 nodes
#   sbatch --nodes=4 --time=12:00:00 src/scripts/slurm_stage4_scaled_features.sh  # Use 4 nodes
#

#SBATCH --job-name=fvc_stage4_feat
#SBATCH --account=eecs442f25_class
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem=80G
#SBATCH --nodes=1-4
#SBATCH --ntasks-per-node=1
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/stage4_feat-%j.out
#SBATCH --error=logs/stage4_feat-%j.err
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
module load ffmpeg || module load ffmpeg/4.4 || true

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
log "STAGE 4: SCALED VIDEO FEATURE EXTRACTION JOB"
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

# Check ffprobe availability
if ! command -v ffprobe &> /dev/null; then
    log "⚠ WARNING: ffprobe not found. FFmpeg module may not be loaded correctly."
    log "  Try: module load ffmpeg"
    log "  Continuing anyway (may use fallback methods)..."
else
    log "✓ ffprobe found: $(which ffprobe)"
fi

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

# Verify Stage 3 output (try Arrow first, then Parquet, then CSV)
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

log "✅ All prerequisites verified"

# ============================================================================
# Stage 4 Execution
# ============================================================================

log "=========================================="
log "Starting Stage 4: Scaled Video Feature Extraction"
log "=========================================="

NUM_FRAMES="${FVC_NUM_FRAMES:-8}"  # Optimized for 80GB RAM
OUTPUT_DIR="${FVC_STAGE4_OUTPUT_DIR:-data/features_stage4}"
DELETE_EXISTING="${FVC_DELETE_EXISTING:-0}"
RESUME="${FVC_RESUME:-1}"

log "Number of frames: $NUM_FRAMES"
log "Output directory: $OUTPUT_DIR"
log "Delete existing: $DELETE_EXISTING"
log "Resume mode: $RESUME"

# Calculate total videos and distribute across nodes
NUM_NODES="${SLURM_JOB_NUM_NODES:-1}"
log "Number of nodes: $NUM_NODES"

# Get total video count from metadata
TOTAL_VIDEOS=$(python -c "
import polars as pl
import sys
try:
    metadata_path = '$SCALED_METADATA'
    if metadata_path.endswith('.arrow'):
        df = pl.read_ipc(metadata_path)
    elif metadata_path.endswith('.parquet'):
        df = pl.read_parquet(metadata_path)
    else:
        df = pl.read_csv(metadata_path)
    print(df.height)
except Exception as e:
    print('0', file=sys.stderr)
    sys.exit(1)
" 2>/dev/null || echo "0")

if [ "$TOTAL_VIDEOS" = "0" ] || [ -z "$TOTAL_VIDEOS" ]; then
    log "✗ ERROR: Could not determine total number of videos from $SCALED_METADATA"
    exit 1
fi

log "Total videos to process: $TOTAL_VIDEOS"

# Calculate videos per node
VIDEOS_PER_NODE=$((TOTAL_VIDEOS / NUM_NODES))
REMAINDER=$((TOTAL_VIDEOS % NUM_NODES))
log "Videos per node: ~$VIDEOS_PER_NODE (remainder: $REMAINDER)"

STAGE4_START=$(date +%s)
COMBINED_LOG_FILE="$ORIG_DIR/logs/stage4_scaled_features_combined_${SLURM_JOB_ID:-$$}.log"
mkdir -p "$(dirname "$COMBINED_LOG_FILE")"

cd "$ORIG_DIR" || exit 1
PYTHON_CMD=$(which python || echo "python")

# Helper function to verify completion
verify_stage_completion() {
    local expected_count=$1
    local metadata_file=$2
    
    if [ ! -f "$metadata_file" ]; then
        log "✗ ERROR: Metadata file not found: $metadata_file"
        return 1
    fi
    
    local actual_count=$(python -c "
import polars as pl
import sys
try:
    metadata_path = '$metadata_file'
    if metadata_path.endswith('.arrow'):
        df = pl.read_ipc(metadata_path)
    elif metadata_path.endswith('.parquet'):
        df = pl.read_parquet(metadata_path)
    else:
        df = pl.read_csv(metadata_path)
    print(df.height)
except Exception as e:
    print('0', file=sys.stderr)
    sys.exit(1)
" 2>/dev/null || echo "0")
    
    if [ "$actual_count" -lt "$expected_count" ]; then
        log "✗ ERROR: Stage 4 incomplete. Expected $expected_count, found $actual_count"
        return 1
    fi
    log "✓ Stage 4 completion verified: $actual_count videos processed"
    return 0
}

# Multi-node execution: distribute videos across nodes
NODE_FAILURES=0
NODE_SUCCESSES=0

for NODE_ID in $(seq 0 $((NUM_NODES - 1))); do
    # Calculate range for this node
    START_IDX=$((NODE_ID * VIDEOS_PER_NODE))
    if [ $NODE_ID -eq $((NUM_NODES - 1)) ]; then
        # Last node gets remainder
        END_IDX=$TOTAL_VIDEOS
    else
        END_IDX=$((START_IDX + VIDEOS_PER_NODE))
    fi
    
    log "=========================================="
    log "Node $NODE_ID: Processing videos [$START_IDX, $END_IDX)"
    log "=========================================="
    
    NODE_LOG_FILE="$ORIG_DIR/logs/stage4_feat-${SLURM_JOB_ID:-$$}_node${NODE_ID}.log"
    mkdir -p "$(dirname "$NODE_LOG_FILE")"
    
    # Build command arguments
    DELETE_FLAG=""
    if [ "$DELETE_EXISTING" = "1" ] || [ "$DELETE_EXISTING" = "true" ] || [ "$DELETE_EXISTING" = "yes" ]; then
        DELETE_FLAG="--delete-existing"
    fi
    
    RESUME_FLAG=""
    if [ "$RESUME" != "0" ] && [ "$RESUME" != "false" ] && [ "$RESUME" != "no" ]; then
        RESUME_FLAG="--resume"
    fi
    
    # Use srun to launch on specific node
    if [ "$NUM_NODES" -gt 1 ]; then
        NODE_HOSTNAME=$(scontrol show hostnames ${SLURM_JOB_NODELIST} | sed -n "$((NODE_ID + 1))p")
        log "Launching on node $NODE_ID (hostname: $NODE_HOSTNAME)"
        
        if srun --nodes=1 --ntasks=1 --nodelist="$NODE_HOSTNAME" \
            "$PYTHON_CMD" "$ORIG_DIR/src/scripts/run_stage4_scaled_features.py" \
            --project-root "$ORIG_DIR" \
            --scaled-metadata "$SCALED_METADATA" \
            --num-frames "$NUM_FRAMES" \
            --output-dir "$OUTPUT_DIR" \
            --start-idx "$START_IDX" \
            --end-idx "$END_IDX" \
            $DELETE_FLAG \
            $RESUME_FLAG \
            2>&1 | tee "$NODE_LOG_FILE"; then
            log "✓ Node $NODE_ID completed successfully"
            NODE_SUCCESSES=$((NODE_SUCCESSES + 1))
        else
            log "✗ ERROR: Node $NODE_ID failed"
            NODE_FAILURES=$((NODE_FAILURES + 1))
        fi
    else
        # Single node execution
        log "Single node execution"
        if "$PYTHON_CMD" "$ORIG_DIR/src/scripts/run_stage4_scaled_features.py" \
            --project-root "$ORIG_DIR" \
            --scaled-metadata "$SCALED_METADATA" \
            --num-frames "$NUM_FRAMES" \
            --output-dir "$OUTPUT_DIR" \
            --start-idx "$START_IDX" \
            --end-idx "$END_IDX" \
            $DELETE_FLAG \
            $RESUME_FLAG \
            2>&1 | tee "$NODE_LOG_FILE"; then
            log "✓ Node $NODE_ID completed successfully"
            NODE_SUCCESSES=$((NODE_SUCCESSES + 1))
        else
            log "✗ ERROR: Node $NODE_ID failed"
            NODE_FAILURES=$((NODE_FAILURES + 1))
        fi
    fi
    
    # Append node log to combined log
    echo "" >> "$COMBINED_LOG_FILE"
    echo "=== Node $NODE_ID Log ===" >> "$COMBINED_LOG_FILE"
    cat "$NODE_LOG_FILE" >> "$COMBINED_LOG_FILE"
done

# Verify completion
METADATA_FILE="$ORIG_DIR/$OUTPUT_DIR/features_scaled_metadata.arrow"
if [ ! -f "$METADATA_FILE" ]; then
    METADATA_FILE="$ORIG_DIR/$OUTPUT_DIR/features_scaled_metadata.parquet"
fi

if [ "$NODE_FAILURES" -gt 0 ]; then
    log "✗ ERROR: $NODE_FAILURES node(s) failed"
    exit 1
fi

if ! verify_stage_completion "$TOTAL_VIDEOS" "$METADATA_FILE"; then
    log "✗ ERROR: Completion verification failed"
    exit 1
fi

if [ "$NODE_SUCCESSES" -eq "$NUM_NODES" ]; then
    
    STAGE4_END=$(date +%s)
    STAGE4_DURATION=$((STAGE4_END - STAGE4_START))
    log "✓ Stage 4 completed successfully in ${STAGE4_DURATION}s ($(($STAGE4_DURATION / 60)) minutes)"
    log "Results saved to: $ORIG_DIR/$OUTPUT_DIR"
    log "All $NUM_NODES node(s) completed successfully"
    log "Next step: Run Stage 5 with: sbatch src/scripts/slurm_stage5_training.sh"
else
    STAGE4_END=$(date +%s)
    STAGE4_DURATION=$((STAGE4_END - STAGE4_START))
    log "✗ ERROR: Stage 4 failed after ${STAGE4_DURATION}s"
    log "Check log files: $ORIG_DIR/logs/stage4_feat-${SLURM_JOB_ID:-$$}_node*.log"
    exit 1
fi

log ""
log "============================================================"
log "STAGE 4 EXECUTION SUMMARY"
log "============================================================"
log "Execution time: ${STAGE4_DURATION}s ($(($STAGE4_DURATION / 60)) minutes)"
log "Nodes used: $NUM_NODES"
log "Videos processed: $TOTAL_VIDEOS"
log "Output directory: $ORIG_DIR/$OUTPUT_DIR"
log "Combined log file: $COMBINED_LOG_FILE"
log "Node log files: $ORIG_DIR/logs/stage4_feat-${SLURM_JOB_ID:-$$}_node*.log"
log "============================================================"

