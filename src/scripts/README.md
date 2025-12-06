# Stage-Wise Pipeline Scripts

This directory contains individual scripts to run each stage of the FVC pipeline independently. This allows for:
- Running stages in parallel (if dependencies allow)
- Resuming from a specific stage
- Debugging individual stages
- Better resource management

## Script Types

### Python Scripts (Local/Interactive)
- `run_stage1_augmentation.py` - Stage 1: Video Augmentation
- `run_stage2_features.py` - Stage 2: Handcrafted Feature Extraction
- `run_stage3_scaling.py` - Stage 3: Video Scaling
- `run_stage4_scaled_features.py` - Stage 4: Scaled Video Feature Extraction
- `run_stage5_training.py` - Stage 5: Model Training

### Utility/Validation Scripts
- `check_stage1_completion.py` - Verify Stage 1 completion (checks for missing/incomplete augmentations)
- `validate_augmentations.py` - Validate augmented video file integrity

### SLURM Batch Scripts (Cluster)
- `slurm_stage1_augmentation.sh` - Stage 1: Video Augmentation (SLURM)
- `slurm_stage2_features.sh` - Stage 2: Feature Extraction (SLURM)
- `slurm_stage3_scaling.sh` - Stage 3: Video Scaling (SLURM)
- `slurm_stage4_scaled_features.sh` - Stage 4: Scaled Features (SLURM)
- `slurm_stage5_training.sh` - Stage 5: Model Training (SLURM)

## Available Scripts

### Stage 1: Video Augmentation
**Script:** `run_stage1_augmentation.py`

Generates multiple augmented versions of each video using spatial and temporal transformations.

**Usage:**
```bash
# Default: 10 augmentations per video
python src/scripts/run_stage1_augmentation.py

# Custom number of augmentations
python src/scripts/run_stage1_augmentation.py --num-augmentations 5

# Custom output directory
python src/scripts/run_stage1_augmentation.py --output-dir data/custom_augmented
```

**Output:**
- `data/augmented_videos/augmented_metadata.csv` - Metadata for all videos (original + augmented)
- `data/augmented_videos/*.mp4` - Augmented video files

**Prerequisites:**
- `data/video_index_input.csv` must exist (run `python src/setup_fvc_dataset.py` first)

---

### Stage 2: Handcrafted Feature Extraction
**Script:** `run_stage2_features.py`

Extracts handcrafted features from original videos (M features).

**Usage:**
```bash
# Default: uses Stage 1 output
python src/scripts/run_stage2_features.py

# Custom number of frames
python src/scripts/run_stage2_features.py --num-frames 6

# Custom metadata path
python src/scripts/run_stage2_features.py --augmented-metadata data/custom/augmented_metadata.csv
```

**Output:**
- `data/features_stage2/features_metadata.csv` - Features metadata
- `data/features_stage2/*.npy` - Feature files (one per video)

**Prerequisites:**
- Stage 1 must be completed (or provide `--augmented-metadata` path)

---

### Stage 3: Video Scaling
**Script:** `run_stage3_scaling.py`

Scales videos to a target max dimension using letterboxing or autoencoder.

**Usage:**
```bash
# Default: 224x224 with resolution method
python src/scripts/run_stage3_scaling.py

# Custom target size
python src/scripts/run_stage3_scaling.py --target-size 112

# Custom method
python src/scripts/run_stage3_scaling.py --method resolution
```

**Output:**
- `data/scaled_videos/scaled_metadata.arrow` - Scaled videos metadata
- `data/scaled_videos/*.mp4` - Scaled video files

**Prerequisites:**
- Stage 1 must be completed (or provide `--augmented-metadata` path)

---

### Stage 4: Scaled Video Feature Extraction
**Script:** `run_stage4_scaled_features.py`

Extracts additional features from scaled videos (P features).

**Usage:**
```bash
# Default: uses Stage 3 output
python src/scripts/run_stage4_scaled_features.py

# Custom number of frames
python src/scripts/run_stage4_scaled_features.py --num-frames 6

# Custom metadata path
python src/scripts/run_stage4_scaled_features.py --scaled-metadata data/custom/scaled_metadata.arrow
```

**Output:**
- `data/features_stage4/features_scaled_metadata.arrow` - Scaled features metadata
- `data/features_stage4/*.npy` - Downscaled feature files (one per video)

**Prerequisites:**
- Stage 3 must be completed (or provide `--scaled-metadata` path)

---

### Stage 5: Model Training
**Script:** `run_stage5_training.py`

Trains models using scaled videos and extracted features.

**Usage:**
```bash
# Default: train logistic_regression and svm with 5-fold CV
python src/scripts/run_stage5_training.py

# Train specific models
python src/scripts/run_stage5_training.py --model-types logistic_regression svm naive_cnn

# Train all available models
python src/scripts/run_stage5_training.py --model-types all

# Custom k-fold splits
python src/scripts/run_stage5_training.py --n-splits 10
```

**Output:**
- `data/training_results/` - Training results, models, and metrics for each model type

**Prerequisites:**
- Stages 1, 2, 3, and 4 must be completed (or provide custom paths)

**Available Models:**
- `logistic_regression` - Logistic Regression baseline
- `svm` - Linear SVM baseline
- `naive_cnn` - Naive CNN baseline
- `vit_gru` - ViT + GRU model
- `vit_transformer` - ViT + Transformer model
- `slowfast` - SlowFast model
- `x3d` - X3D model

---

## Running All Stages Sequentially

### Local Execution (Python Scripts)

```bash
# Run Stage 1
python src/scripts/run_stage1_augmentation.py

# Run Stage 2
python src/scripts/run_stage2_features.py

# Run Stage 3
python src/scripts/run_stage3_scaling.py

# Run Stage 4
python src/scripts/run_stage4_scaled_features.py

# Run Stage 5
python src/scripts/run_stage5_training.py
```

### Cluster Execution (SLURM Scripts)

```bash
# Submit Stage 1
sbatch src/scripts/slurm_stage1_augmentation.sh

# After Stage 1 completes, submit Stage 2
sbatch src/scripts/slurm_stage2_features.sh

# After Stage 2 completes, submit Stage 3
sbatch src/scripts/slurm_stage3_scaling.sh

# After Stage 3 completes, submit Stage 4
sbatch src/scripts/slurm_stage4_scaled_features.sh

# After Stage 4 completes, submit Stage 5
sbatch src/scripts/slurm_stage5_training.sh
```

### Using Full Pipeline Script

```bash
# Local
python src/run_new_pipeline.py

# Or MLOps pipeline
python src/run_new_pipeline.py
```

## SLURM Script Configuration

SLURM scripts support environment variables for customization:

### Stage 1 (Augmentation)
```bash
export FVC_NUM_AUGMENTATIONS=10  # Number of augmentations per video
export FVC_STAGE1_OUTPUT_DIR="data/augmented_videos"  # Output directory
sbatch src/scripts/slurm_stage1_augmentation.sh
```

### Stage 2 (Features)
```bash
export FVC_NUM_FRAMES=8  # Number of frames
export FVC_STAGE2_OUTPUT_DIR="data/features_stage2"  # Output directory
sbatch src/scripts/slurm_stage2_features.sh
```

### Stage 3 (Scaling)
```bash
export FVC_TARGET_SIZE=224  # Target size (224x224)
export FVC_DOWNSCALE_METHOD="resolution"  # Method: resolution or autoencoder
export FVC_STAGE3_OUTPUT_DIR="data/scaled_videos"  # Output directory
sbatch src/scripts/slurm_stage3_scaling.sh
```

### Stage 4 (Scaled Features)
```bash
export FVC_NUM_FRAMES=8  # Number of frames
export FVC_STAGE4_OUTPUT_DIR="data/features_stage4"  # Output directory
sbatch src/scripts/slurm_stage4_scaled_features.sh
```

### Stage 5 (Training)
```bash
export FVC_MODELS="logistic_regression svm"  # Models to train (space-separated)
export FVC_N_SPLITS=5  # K-fold splits
export FVC_NUM_FRAMES=8  # Number of frames
export FVC_USE_TRACKING="true"  # Enable experiment tracking
export FVC_STAGE5_OUTPUT_DIR="data/training_results"  # Output directory
sbatch src/scripts/slurm_stage5_training.sh
```

### Training All Models
```bash
export FVC_MODELS="all"  # Train all available models
sbatch src/scripts/slurm_stage5_training.sh
```

### Custom Resource Allocation
```bash
# Request more time
sbatch --time=24:00:00 src/scripts/slurm_stage5_training.sh

# Request more memory
sbatch --mem=128G src/scripts/slurm_stage5_training.sh

# Request more GPUs
sbatch --gpus=2 src/scripts/slurm_stage5_training.sh
```

## Logging

All scripts generate extensive logs:
- **Console output:** Real-time progress and status
- **Log files:** Saved to `logs/stageX_*.log` with full DEBUG-level logging
- **Memory statistics:** Logged before and after each stage
- **Timing information:** Execution time for each stage
- **Error handling:** Full tracebacks on failures

## Common Options

All scripts support:
- `--project-root`: Project root directory (default: current directory)
- `--output-dir`: Custom output directory
- Extensive logging with DEBUG level
- Error handling with full tracebacks
- Memory statistics logging

## Utility Scripts

### Stage 1 Completion Check
**Script:** `check_stage1_completion.py`

Verifies that all videos from input metadata were processed and augmented correctly in Stage 1.

**Usage:**
```bash
# Auto-detect input and output metadata
python src/scripts/check_stage1_completion.py

# Specify custom paths
python src/scripts/check_stage1_completion.py \
    --input-metadata data/video_index_input.csv \
    --output-metadata data/augmented_videos/augmented_metadata.arrow

# Custom expected augmentations
python src/scripts/check_stage1_completion.py --expected-augmentations 10
```

**What it checks:**
- Missing videos: Videos in input but not in output
- Incomplete augmentations: Videos with fewer than expected augmentations
- Extra videos: Videos in output but not in input (may indicate issues)
- Statistics: Original vs augmented counts, expected vs actual totals

**Output:**
- Detailed report of missing/incomplete videos
- Statistics on augmentation completeness
- Exit code 0 if complete, 1 if incomplete

---

## Troubleshooting

### Stage 1 fails: "Metadata file not found"
- Run `python src/setup_fvc_dataset.py` first to generate `data/video_index_input.csv`

### Stage 2/3/4/5 fails: "Metadata file not found"
- Ensure previous stages have completed successfully
- Check that metadata files exist in expected locations
- Use `--augmented-metadata`, `--scaled-metadata`, etc. to specify custom paths

### Out of Memory (OOM) errors
- Reduce `--num-augmentations` in Stage 1
- Reduce `--num-frames` in Stages 2 and 4
- Reduce `--target-size` in Stage 3
- Use smaller batch sizes (configured in model factory)

### Scripts are slow
- This is expected for large datasets
- Check log files for progress
- Consider running on a cluster with more resources

## Next Steps

After completing all stages:
- Review training results in `data/training_results/`
- Compare model performance across different architectures
- Use best models for inference

