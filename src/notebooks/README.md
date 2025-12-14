# Model Presentation Notebooks

This directory contains Jupyter notebooks for presenting and demonstrating the complete deepfake detection pipeline and all trained models.

## Master Pipeline Notebook

**Start here**: [`00_MASTER_PIPELINE_JOURNEY.ipynb`](00_MASTER_PIPELINE_JOURNEY.ipynb)

This comprehensive notebook demonstrates the **complete end-to-end journey** from raw ZIP files to production-ready models, including:

- **Data Extraction**: From password-protected ZIP archives to organized datasets
- **Data Exploration**: Understanding dataset characteristics, class distribution, video statistics
- **Stage 1: Augmentation**: Spatial and temporal augmentation strategies, pre-generation rationale
- **Stage 2: Feature Extraction**: Handcrafted features (noise residual, DCT, blur/sharpness, codec cues)
- **Stage 3: Video Scaling**: Letterbox resizing, upscaling/downscaling strategies
- **Stage 4: Scaled Features**: Feature extraction from scaled videos
- **Stage 5: Model Training**: 23 different architectures with hyperparameter tuning
- **MLOps Infrastructure**: MLflow experiment tracking, DuckDB analytics, Airflow orchestration
- **Results & Insights**: Performance analysis, key findings, next steps

**Technologies Demonstrated**:
- PyTorch, torchvision, timm (Deep Learning)
- Polars, PyArrow, DuckDB (Data Processing)
- MLflow (Experiment Tracking)
- Apache Airflow (Orchestration)
- PyAV, OpenCV (Video Processing)

This notebook is designed for **ML Engineers, Data Scientists, and Researchers** at a **production-grade level**.

## Overview

Each notebook is designed for **presentation purposes** - they load and display trained models, show examples with sample videos, and present results. **They do NOT train models** - training instructions are provided in commented markdown sections.

## Notebook Structure

Each notebook follows this structure:

1. **Model Overview**: Description of the model architecture and approach
2. **Training Instructions**: Commented code showing how to train the model
3. **Setup**: Import libraries and configure paths
4. **Check for Saved Models**: Verify that trained models exist
5. **Load Model**: Load the trained model from disk
6. **Display Sample Videos**: Show example videos with thumbnails
7. **Model Performance Summary**: Display metrics and visualizations
8. **Model Architecture Summary**: Detailed architecture description

## Available Notebooks

### Standalone Training Scripts
- **5alpha_sklearn_logreg.ipynb**: sklearn LogisticRegression (standalone implementation)
- **5beta_gradient_boosting.ipynb**: Gradient Boosting models (XGBoost, LightGBM, CatBoost)

### Baseline Models (Feature-Based)
- **5a_logistic_regression.ipynb**: Logistic Regression classifier
- **5b_svm.ipynb**: Support Vector Machine classifier

### PyTorch CNN Models
- **5c_naive_cnn.ipynb**: Simple 3D CNN
- **5d_pretrained_inception.ipynb**: Pretrained R3D-18 with Inception blocks
- **5e_variable_ar_cnn.ipynb**: Variable aspect ratio CNN

### XGBoost Models (Feature Extraction + Classification)
- **5f_xgboost_pretrained_inception.ipynb**: XGBoost with Pretrained Inception features
- **5g_xgboost_i3d.ipynb**: XGBoost with I3D features
- **5h_xgboost_r2plus1d.ipynb**: XGBoost with R(2+1)D features
- **5i_xgboost_vit_gru.ipynb**: XGBoost with ViT-GRU features
- **5j_xgboost_vit_transformer.ipynb**: XGBoost with ViT-Transformer features

### Vision Transformer Models
- **5k_vit_gru.ipynb**: ViT backbone + GRU temporal head
- **5l_vit_transformer.ipynb**: ViT backbone + Transformer encoder temporal head

### Video Transformer Models
- **5m_timesformer.ipynb**: TimeSformer with divided space-time attention
- **5n_vivit.ipynb**: ViViT with tubelet embedding

### 3D CNN Models
- **5o_i3d.ipynb**: I3D (Inflated 3D ConvNet)
- **5p_r2plus1d.ipynb**: R(2+1)D (Factorized 3D Convolutions)
- **5q_x3d.ipynb**: X3D (Efficient video model)

### SlowFast Models
- **5r_slowfast.ipynb**: SlowFast dual-pathway architecture
- **5s_slowfast_attention.ipynb**: SlowFast with attention mechanisms
- **5t_slowfast_multiscale.ipynb**: Multi-scale SlowFast

### Two-Stream Models
- **5u_two_stream.ipynb**: Two-stream network (RGB + optical flow)

## Usage

### Prerequisites

1. Trained models must exist:
   - **5alpha**: `data/stage5/sklearn_logreg/model.joblib` (single model, not in fold subdirectories)
   - **5beta**: `data/stage5/{model_name}/model.json` (XGBoost), `model.joblib` (LightGBM), or `model.cbm` (CatBoost)
   - **5a-5u**: `data/stage5/{model_type}/fold_*/model.pt` (PyTorch) or `model.joblib` (XGBoost/sklearn)
2. Scaled video metadata at `data/stage3/scaled_metadata.parquet`
3. Feature metadata (for baseline models) at `data/stage2/features_metadata.parquet`

### Running a Notebook

```bash
# Start Jupyter
jupyter notebook

# Or use JupyterLab
jupyter lab

# Navigate to src/notebooks/ and open the desired notebook
```

### Example: Viewing Model 5r (SlowFast)

1. Open `5r_slowfast.ipynb`
2. Run all cells (Cell → Run All)
3. The notebook will:
   - Check for saved SlowFast models
   - Load the model if available
   - Display sample videos
   - Show performance metrics
   - Present architecture details

## Model Locations

Models are saved in the following structure:

**5alpha (sklearn_logreg):**
```
data/stage5/sklearn_logreg/
├── model.joblib
├── scaler.joblib
├── metrics.json
└── roc_pr_curves.png
```

**5beta (gradient boosting):**
```
data/stage5/
├── xgboost/
│   ├── model.json
│   ├── metrics.json
│   └── ...
├── lightgbm/
│   ├── model.joblib
│   ├── metrics.json
│   └── ...
└── catboost/
    ├── model.cbm
    ├── metrics.json
    └── ...
```

**5a-5u (pipeline models):**
```
data/stage5/
├── {model_type}/
│   ├── fold_1/
│   │   ├── model.pt (or model.joblib)
│   │   ├── metrics.json
│   │   └── ...
│   ├── fold_2/
│   └── ...
```

## Training Models

To train models, use the provided SLURM scripts or Python API:

```bash
# Using SLURM scripts
sbatch src/scripts/slurm_stage5a.sh  # Train logistic regression
sbatch src/scripts/slurm_stage5r.sh  # Train SlowFast
# ... etc

# Using Python API (see training instructions in each notebook)
from lib.training.pipeline import stage5_train_models
results = stage5_train_models(...)
```

## Notes

- **These notebooks are for presentation only** - they load existing models, not train new ones
- Training instructions are provided in commented markdown sections
- Sample videos are displayed as thumbnails (first frame)
- To play full videos, use: `display(Video('path/to/video.mp4', embed=True))`
- Metrics are loaded from `metrics.json` files in each fold directory

## Troubleshooting

### "No trained models found"
- Ensure models have been trained first using the SLURM scripts
- Check that `data/stage5/{model_type}/fold_*/model.pt` (or `model.joblib`) exists

### "Could not load metadata files"
- Ensure Stage 3 scaled metadata exists: `data/stage3/scaled_metadata.parquet`
- For baseline models, ensure Stage 2 features exist: `data/stage2/features_metadata.parquet`

### Import errors
- Ensure project root is correctly set in the notebook
- Run the setup cell first to add project root to Python path
