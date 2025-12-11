#!/usr/bin/env python3
"""
Reconstruct unified metadata files from individual feature parquet files.

This script scans a directory for individual feature parquet files and combines
them into a unified metadata file (Arrow or Parquet format).
"""

import sys
import logging
from pathlib import Path
import polars as pl
import numpy as np
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)


def reconstruct_metadata_from_parquet_files(
    features_dir: Path,
    output_filename: str = "features_metadata.parquet",
    pattern: str = "*_features.parquet"
) -> Optional[Path]:
    """
    Reconstruct unified metadata from individual parquet feature files.
    
    Args:
        features_dir: Directory containing individual feature parquet files
        output_filename: Name of output metadata file
        pattern: Glob pattern to match feature files (default: "*_features.parquet")
    
    Returns:
        Path to reconstructed metadata file, or None if failed
    """
    if not features_dir.exists():
        logger.error(f"Directory does not exist: {features_dir}")
        return None
    
    # Find all feature parquet files
    feature_files = sorted(features_dir.glob(pattern))
    
    if not feature_files:
        logger.error(f"No feature files found matching pattern '{pattern}' in {features_dir}")
        return None
    
    logger.info(f"Found {len(feature_files)} feature files")
    
    # Load and combine all feature files
    all_rows = []
    failed_count = 0
    
    for i, feature_file in enumerate(feature_files):
        if (i + 1) % 100 == 0:
            logger.info(f"Processing {i + 1}/{len(feature_files)} files...")
        
        try:
            # Read the parquet file
            df = pl.read_parquet(feature_file)
            
            # Extract video_id from filename (remove _features.parquet suffix)
            video_id = feature_file.stem.replace("_features", "").replace("_scaled_features", "")
            
            # Create a row with video_path
            row_dict = {"video_path": video_id}
            
            # Handle different parquet file structures
            if df.height == 0:
                # Empty dataframe - skip
                logger.debug(f"Skipping empty file: {feature_file}")
                continue
            elif df.height == 1:
                # Single row - extract all values (already aggregated)
                row = df.row(0, named=True)
                for col, val in row.items():
                    if col != "video_path":  # Don't overwrite video_path
                        # Handle different value types
                        if isinstance(val, (list, np.ndarray)):
                            # If it's an array, take mean (shouldn't happen but handle it)
                            row_dict[col] = float(np.mean(val)) if len(val) > 0 else 0.0
                        else:
                            row_dict[col] = val
            else:
                # Multiple rows - this is frame-level features that need aggregation
                # Aggregate all numeric columns by mean
                for col in df.columns:
                    if col != "video_path":
                        if df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]:
                            # Aggregate numeric columns by mean
                            row_dict[col] = float(df[col].mean())
                        elif df[col].dtype in [pl.List]:
                            # If column contains arrays, aggregate those too
                            # This handles the (1, 8) shape issue - aggregate across the 8 frames
                            try:
                                # Try to extract and aggregate nested arrays
                                all_values = []
                                for val in df[col]:
                                    if isinstance(val, list):
                                        all_values.extend(val)
                                    elif isinstance(val, np.ndarray):
                                        all_values.extend(val.flatten().tolist())
                                    else:
                                        all_values.append(val)
                                row_dict[col] = float(np.mean(all_values)) if all_values else 0.0
                            except Exception:
                                # Fallback: take first value
                                row_dict[col] = df[col][0] if df.height > 0 else 0.0
                        else:
                            # Take first value for non-numeric
                            row_dict[col] = df[col][0] if df.height > 0 else 0.0
            
            all_rows.append(row_dict)
            
        except Exception as e:
            logger.warning(f"Failed to read {feature_file}: {e}")
            failed_count += 1
            continue
    
    if not all_rows:
        logger.error("No valid feature rows found")
        return None
    
    logger.info(f"Successfully loaded {len(all_rows)} feature rows ({failed_count} failed)")
    
    # Create DataFrame
    try:
        combined_df = pl.DataFrame(all_rows)
        logger.info(f"Combined DataFrame shape: {combined_df.shape}")
        
        # Save to output file
        output_path = features_dir / output_filename
        
        # Try to save as parquet first (more reliable)
        if output_filename.endswith('.parquet'):
            combined_df.write_parquet(output_path)
            logger.info(f"✓ Saved unified metadata to: {output_path}")
        elif output_filename.endswith('.arrow'):
            combined_df.write_ipc(output_path)
            logger.info(f"✓ Saved unified metadata to: {output_path}")
        else:
            # Default to parquet
            output_path = features_dir / "features_metadata.parquet"
            combined_df.write_parquet(output_path)
            logger.info(f"✓ Saved unified metadata to: {output_path}")
        
        return output_path
        
    except Exception as e:
        logger.error(f"Failed to create combined DataFrame: {e}", exc_info=True)
        return None


def main():
    """Main function to reconstruct metadata files."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Reconstruct unified metadata from individual feature parquet files"
    )
    parser.add_argument(
        "--features-dir",
        type=str,
        required=True,
        help="Directory containing individual feature parquet files"
    )
    parser.add_argument(
        "--output-filename",
        type=str,
        default="features_metadata.parquet",
        help="Output metadata filename (default: features_metadata.parquet)"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*_features.parquet",
        help="Glob pattern to match feature files (default: *_features.parquet)"
    )
    
    args = parser.parse_args()
    
    features_dir = Path(args.features_dir).resolve()
    
    logger.info("=" * 80)
    logger.info("RECONSTRUCTING METADATA FROM PARQUET FILES")
    logger.info("=" * 80)
    logger.info(f"Features directory: {features_dir}")
    logger.info(f"Output filename: {args.output_filename}")
    logger.info(f"Pattern: {args.pattern}")
    logger.info("=" * 80)
    
    result = reconstruct_metadata_from_parquet_files(
        features_dir,
        args.output_filename,
        args.pattern
    )
    
    if result:
        logger.info("=" * 80)
        logger.info("✓ RECONSTRUCTION COMPLETE")
        logger.info(f"Metadata file: {result}")
        logger.info("=" * 80)
        return 0
    else:
        logger.error("=" * 80)
        logger.error("✗ RECONSTRUCTION FAILED")
        logger.error("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(main())

