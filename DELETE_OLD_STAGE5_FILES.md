# Files to Delete - Old Stage 5 Implementations

The following files contain old video-based implementations that should be deleted
now that we have the new feature-based pipeline:

## Model Implementation Files (to be replaced):
- `lib/training/_cnn.py` - Old CNN implementation
- `lib/training/_transformer_gru.py` - Old ViT-GRU implementation  
- `lib/training/_transformer.py` - Old ViT-Transformer implementation
- `lib/training/i3d.py` - Old I3D implementation
- `lib/training/r2plus1d.py` - Old R2Plus1D implementation
- `lib/training/slowfast.py` - Old SlowFast implementation
- `lib/training/slowfast_advanced.py` - Old advanced SlowFast
- `lib/training/timesformer.py` - Old TimeSformer implementation
- `lib/training/two_stream.py` - Old Two-Stream implementation
- `lib/training/vivit.py` - Old ViViT implementation
- `lib/training/x3d.py` - Old X3D implementation
- `lib/training/_xgboost_pretrained.py` - Old XGBoost implementation

## Keep (but update):
- `lib/training/_linear.py` - Update to use new pipeline
- `lib/training/_svm.py` - Update to use new pipeline
- `lib/training/model_factory.py` - Update to use new models
- `lib/training/pipeline.py` - Replace with new feature-based pipeline
- `lib/training/grid_search.py` - Update hyperparameter grids
- `lib/training/trainer.py` - May need updates
- `lib/training/ensemble.py` - May need updates

## New Files (keep):
- `lib/training/feature_pipeline.py` - NEW: Core feature pipeline
- `lib/training/feature_models.py` - NEW: Feature-based PyTorch models
- `lib/training/feature_training_pipeline.py` - NEW: Training pipeline
- `lib/training/stage5_feature_pipeline.py` - NEW: Main entry point

