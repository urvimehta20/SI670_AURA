"""
Pandera schema definitions for data validation across pipeline stages.

This module provides schema validation using Pandera to ensure data integrity
between pipeline stages.
"""

from __future__ import annotations

import logging
from typing import Optional

try:
    import pandera as pa
    from pandera.typing import DataFrame
    PANDERA_AVAILABLE = True
except ImportError:
    PANDERA_AVAILABLE = False
    # Create dummy classes for type hints
    class DataFrame:
        pass

logger = logging.getLogger(__name__)


if PANDERA_AVAILABLE:
    # Stage 1 Output Schema: Augmented Metadata
    Stage1AugmentedMetadataSchema = pa.DataFrameSchema({
        "video_path": pa.Column(str, nullable=False, description="Path to video file"),
        "label": pa.Column(str, nullable=False, description="Video label"),
        "original_video": pa.Column(str, nullable=True, description="Original video path"),
        "augmentation_idx": pa.Column(int, nullable=True, description="Augmentation index (-1 for original videos)"),
        "is_original": pa.Column(bool, nullable=True, description="Whether this is the original video"),
    }, strict=True, coerce=True)
    
    # Stage 2 Output Schema: Handcrafted Features Metadata
    Stage2FeaturesMetadataSchema = pa.DataFrameSchema({
        "video_path": pa.Column(str, nullable=False, description="Path to video file"),
        "label": pa.Column(str, nullable=False, description="Video label"),
        "features_path": pa.Column(str, nullable=False, description="Path to extracted features file"),
    }, strict=True, coerce=True)
    
    # Stage 3 Output Schema: Scaled Video Metadata
    Stage3ScaledMetadataSchema = pa.DataFrameSchema({
        "video_path": pa.Column(str, nullable=False, description="Path to scaled video file"),
        "label": pa.Column(str, nullable=False, description="Video label"),
        "original_video_path": pa.Column(str, nullable=True, description="Original video path before scaling"),
        "original_width": pa.Column(int, nullable=True, description="Original video width"),
        "original_height": pa.Column(int, nullable=True, description="Original video height"),
        "scaled_width": pa.Column(int, nullable=True, description="Scaled video width"),
        "scaled_height": pa.Column(int, nullable=True, description="Scaled video height"),
        "is_upscaled": pa.Column(bool, nullable=True, description="Whether video was upscaled"),
        "is_downscaled": pa.Column(bool, nullable=True, description="Whether video was downscaled"),
    }, strict=True, coerce=True)
    
    # Stage 4 Output Schema: Scaled Features Metadata
    Stage4ScaledFeaturesMetadataSchema = pa.DataFrameSchema({
        "video_path": pa.Column(str, nullable=False, description="Path to scaled video file"),
        "label": pa.Column(str, nullable=False, description="Video label"),
        "features_path": pa.Column(str, nullable=False, description="Path to extracted features file"),
    }, strict=True, coerce=True)
    
    # Stage 5 Input Schema: Combined Metadata for Training
    Stage5TrainingMetadataSchema = pa.DataFrameSchema({
        "video_path": pa.Column(str, nullable=False, description="Path to video file"),
        "label": pa.Column(str, nullable=False, description="Video label"),
    }, strict=True, coerce=True)
else:
    # Dummy schemas when Pandera is not available
    Stage1AugmentedMetadataSchema = None
    Stage2FeaturesMetadataSchema = None
    Stage3ScaledMetadataSchema = None
    Stage4ScaledFeaturesMetadataSchema = None
    Stage5TrainingMetadataSchema = None


def validate_stage1_output(
    df: pl.DataFrame, schema_name: str = "Stage1AugmentedMetadataSchema"
) -> bool:
    """
    Validate Stage 1 output using Pandera schema.
    
    Args:
        df: Polars DataFrame to validate
        schema_name: Name of schema to use
    
    Returns:
        True if validation passes, False otherwise
    """
    if not PANDERA_AVAILABLE:
        logger.warning("Pandera not available, skipping schema validation")
        return True
    
    try:
        # Convert Polars DataFrame to Pandas for Pandera validation
        pandas_df = df.to_pandas()
        
        # Get schema
        schema = globals().get(schema_name)
        if schema is None:
            logger.warning(f"Schema {schema_name} not found, skipping validation")
            return True
        
        # Validate
        schema.validate(pandas_df, lazy=True)
        logger.info(
            "Stage 1 output validation passed: %d rows", df.height
        )
        return True
    except Exception as e:
        logger.error("Stage 1 output validation failed: %s", e)
        return False


def validate_stage_output(df, stage: int, schema_name: Optional[str] = None) -> bool:
    """
    Validate stage output using appropriate schema.
    
    Args:
        df: Polars DataFrame to validate
        stage: Stage number (1-5)
        schema_name: Optional schema name override
    
    Returns:
        True if validation passes, False otherwise
    """
    if not PANDERA_AVAILABLE:
        logger.warning("Pandera not available, skipping schema validation")
        return True
    
    schema_map = {
        1: "Stage1AugmentedMetadataSchema",
        2: "Stage2FeaturesMetadataSchema",
        3: "Stage3ScaledMetadataSchema",
        4: "Stage4ScaledFeaturesMetadataSchema",
        5: "Stage5TrainingMetadataSchema",
    }
    
    if schema_name is None:
        schema_name = schema_map.get(stage)
        if schema_name is None:
            logger.warning(f"No schema defined for stage {stage}")
            return True
    
    return validate_stage1_output(df, schema_name)


__all__ = [
    "validate_stage1_output",
    "validate_stage_output",
    "Stage1AugmentedMetadataSchema",
    "Stage2FeaturesMetadataSchema",
    "Stage3ScaledMetadataSchema",
    "Stage4ScaledFeaturesMetadataSchema",
    "Stage5TrainingMetadataSchema",
    "PANDERA_AVAILABLE",
]

