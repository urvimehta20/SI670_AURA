# AURA - Authenticity Understanding of Real vs Artificial Shortform Videos

![AURA- Authenticity Understanding for Real vs Artificial Shortform Videos Santosh Desai Suzane Fernandes Urvi Mehta](https://github.com/user-attachments/assets/d8759923-f308-4cd8-a7d0-db3935535b86)


A deep learning project for binary classification of videos (real vs. fake) using 3D Convolutional Neural Networks with transfer learning.

## Overview

AURA (Authenticity Understanding of Real vs Artificial) is a comprehensive system for detecting fake videos in shortform content. The project implements multiple model architectures for binary video classification using the Fake Video Corpus dataset. The system uses pretrained 3D ResNet backbones (Kinetics-400 weights) with custom classification heads, along with handcrafted feature extraction and ensemble methods.

## Key Features

- **Pretrained 3D CNN**: Uses `torchvision.models.video.r3d_18` with Kinetics-400 weights
- **Fixed-Size Preprocessing**: 224x224 with letterboxing for memory efficiency
- **Comprehensive Augmentations**: Pre-generated spatial and temporal augmentations
- **K-Fold Cross-Validation**: Stratified 5-fold CV for robust evaluation
- **MLOps Infrastructure**: Experiment tracking, checkpointing, and data versioning
- **Memory Optimized**: Aggressive GC, OOM handling, progressive batch size fallback
- **Class Imbalance Handling**: Balanced batch sampling and inverse-frequency weights

## Project Structure

```
aura/
├── archive/                    # Original dataset archives
├── data/                       # Processed metadata
├── docs/                       # Documentation
├── lib/                        # Core library modules
├── src/                        # Scripts and notebooks
├── videos/                     # Extracted video files
├── runs/                       # Experiment outputs
├── models/                     # Saved models
└── logs/                       # Log files
```

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Setup Dataset

```bash
python3 src/setup_fvc_dataset.py
```

### 3. Run Training

**Option A: Using 5-Stage Pipeline (Recommended)**
```bash
python3 src/run_new_pipeline.py
```

**Option B: Using Individual Stage Scripts**
```bash
# Run stages individually
python3 src/scripts/run_stage1_augmentation.py
python3 src/scripts/run_stage2_features.py
python3 src/scripts/run_stage3_scaling.py
python3 src/scripts/run_stage4_scaled_features.py
python3 src/scripts/run_stage5_training.py
```

See [src/scripts/README.md](src/scripts/README.md) for detailed usage.

**Option D: Using SLURM (Cluster)**
```bash
sbatch src/scripts/run_fvc_training.sh
```

### 4. View Results Dashboard

After training completes, launch the interactive results dashboard:

```bash
streamlit run src/dashboard_results.py
```

The enhanced dashboard provides:
- **Model performance comparisons** with confidence intervals
- **K-fold cross-validation analysis** with violin plots
- **ROC curves and Precision-Recall curves**
- **Training curves** (loss/accuracy over epochs)
- **Statistical significance testing** (pairwise t-tests)
- **Comprehensive dataset statistics**
- **Publication-ready visualizations**

See [Dashboard Usage Guide](docs/DASHBOARD_USAGE.md) for details.

### 5. Generate Paper Figures

Generate publication-ready figures for IEEE paper submission:

```bash
python src/generate_paper_figures.py
```

This creates high-quality figures in PNG, PDF, and SVG formats with:
- 300 DPI resolution
- Academic font styling (Times New Roman)
- Proper labels and legends
- Statistical significance analysis
- LaTeX-ready tables

See [Paper Figure Generation Guide](docs/PAPER_FIGURE_GENERATION.md) for details.

## Documentation

### Core Documentation
- [**End-to-End Workflow**](docs/WORKFLOW.md) - Complete pipeline from archive to notebook generation
- [**Project Overview**](docs/PROJECT_OVERVIEW.md) - Comprehensive project documentation, architecture, and implementation details
- [**ML Methodology**](docs/METHODOLOGY.md) - Critical ML methodology fixes, best practices, and validation checks
- [**Optimizations**](docs/OPTIMIZATIONS.md) - Memory optimizations, MLOps improvements, and pipeline enhancements
- [**Production Guardrails**](docs/PRODUCTION_GUARDRAILS.md) - Comprehensive guardrail system for production reliability

### Feature Guides
- [**Autoencoder Scaling**](docs/AUTOENCODER_SCALING.md) - Guide for using Hugging Face autoencoders for video scaling
- [**Dashboard Usage**](docs/DASHBOARD_USAGE.md) - Interactive Streamlit results dashboard guide
- [**Paper Figure Generation**](docs/PAPER_FIGURE_GENERATION.md) - Generate publication-ready figures for IEEE papers

### Reference
- [**Architecture Diagrams**](presentation/ARCHITECTURE_DIAGRAM.md) - ASCII diagrams of pipeline architecture and data flow
- [**Changelog**](docs/CHANGELOG.md) - Project version history and notable changes
- [**Git Setup**](docs/GIT_SETUP.md) - Git repository information and setup instructions

### Component Documentation
- [**Scripts README**](src/scripts/README.md) - Individual stage scripts and SLURM batch scripts
- [**Notebooks README**](src/notebooks/README.md) - Jupyter notebook collection and usage
- [**Test README**](test/README.md) - Unit test suite documentation

## Dependencies

Key dependencies (see `requirements.txt` for full list):
- `torch`, `torchvision`: Deep learning framework
- `polars`: Fast DataFrame operations
- `opencv-python`, `av`, `torchcodec`: Video decoding
- `papermill`, `ipykernel`: Notebook execution

## Model Architecture

- **Backbone**: 3D ResNet-18 (r3d_18) pretrained on Kinetics-400
- **Head**: Inception3DBlock → AdaptiveAvgPool3d → Dropout(0.5) → Linear(512, 1)
- **Input**: (N, C=3, T=16, H=224, W=224)
- **Output**: Binary logits (N, 1)

## Training Configuration

- **Frames per video**: 16 (uniformly sampled)
- **Input size**: 224x224 (fixed with letterboxing)
- **Batch size**: 32 (with progressive fallback: 16 → 8 → 4 → 2 → 1)
- **Optimizer**: Adam (lr=1e-4, weight_decay=1e-4)
- **Mixed Precision**: Enabled (AMP)
- **Cross-Validation**: 5-fold stratified

## Results

Results are saved in `runs/` directory with:
- Experiment configurations (`config.json`)
- Metrics logs (`metrics.jsonl`)
- Model checkpoints (`checkpoints/`)
- System metadata (`metadata.json`)

## Dataset Credits

This project uses the **Fake Video Corpus (FVC)** dataset provided by [MKLab-ITI](https://github.com/MKLab-ITI/fake-video-corpus). We gratefully acknowledge:

- **Dataset Source**: [fake-video-corpus](https://github.com/MKLab-ITI/fake-video-corpus)
- **Dataset Authors**: Olga Papadopoulou, Markos Zampoglou, Symeon Papadopoulos, and Ioannis Kompatsiaris
- **Citation**: 
  ```bibtex
  @article{papadopoulou2018corpus,
    author = "Papadopoulou, Olga and Zampoglou, Markos and Papadopoulos, Symeon and Kompatsiaris, Ioannis",
    title = "A corpus of debunked and verified user-generated videos",
    journal = "Online Information Review",
    doi = "10.1108/OIR-03-2018-0101",
    year={2018},
    publisher={Emerald Publishing Limited}
  }
  ```
- **License**: Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
- **Funding**: Supported by the InVID project, funded by the European Commission under contract number 687786

For questions about the dataset, please contact Olga Papadopoulou at olgapapa@iti.gr.

## License

This project is part of SI670 Applied Machine Learning course.

## Author

Horopter (santoshdesai12@hotmail.com)

