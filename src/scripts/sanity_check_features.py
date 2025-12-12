#!/usr/bin/env python3
"""
Sanity check for Stage 2 and Stage 4 features before Stage 5 training.

Validates that:
- Stage 2 feature metadata exists and has valid data
- Stage 4 feature metadata exists and has valid data (if needed)
- Feature files are accessible
- Minimum data requirements are met
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path
from typing import Optional

# Add project root to path before importing lib
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def get_project_root() -> Path:
    """Get project root directory."""
    # Try to find project root by looking for lib/ directory
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "lib").exists() and (current / "lib" / "__init__.py").exists():
            return current
        current = current.parent
    # Fallback to parent of src/scripts/
    return Path(__file__).resolve().parent.parent.parent


def check_features_metadata(metadata_path: str, stage_name: str, min_rows: int = 100) -> bool:
    """
    Check if feature metadata exists and has valid data.
    
    Args:
        metadata_path: Path to feature metadata file
        stage_name: Name of stage (e.g., "Stage 2", "Stage 4")
        min_rows: Minimum number of rows required
    
    Returns:
        True if valid, False otherwise
    """
    try:
        from lib.utils.paths import load_metadata_flexible
        
        logger.info(f"Checking {stage_name} features metadata: {metadata_path}")
        
        if not metadata_path or not Path(metadata_path).exists():
            logger.warning(f"  ⚠ {stage_name} metadata file not found: {metadata_path}")
            return False
        
        df = load_metadata_flexible(metadata_path)
        
        if df is None:
            logger.warning(f"  ⚠ {stage_name} metadata could not be loaded: {metadata_path}")
            return False
        
        if df.height == 0:
            logger.warning(f"  ⚠ {stage_name} metadata is empty (0 rows): {metadata_path}")
            return False
        
        if df.height < min_rows:
            logger.warning(
                f"  ⚠ {stage_name} metadata has only {df.height} rows "
                f"(minimum {min_rows} recommended): {metadata_path}"
            )
            # Don't fail, just warn
        
        logger.info(f"  ✓ {stage_name} metadata valid: {df.height} feature rows")
        logger.info(f"    Columns: {len(df.columns)}")
        logger.info(f"    Sample columns: {list(df.columns[:5])}")
        
        return True
        
    except Exception as e:
        logger.error(f"  ✗ Error checking {stage_name} metadata: {e}", exc_info=True)
        return False


def main() -> int:
    """Run sanity check for Stage 2 and Stage 4 features."""
    logger.info("=" * 80)
    logger.info("Feature Sanity Check")
    logger.info("=" * 80)
    
    # Use the project root that was already added to sys.path
    project_root = Path(__file__).resolve().parent.parent.parent
    logger.info(f"Project root: {project_root}")
    
    # Default paths (can be overridden by environment variables)
    features_stage2_path = Path(project_root) / "data" / "features_stage2" / "metadata.parquet"
    features_stage4_path = Path(project_root) / "data" / "features_stage4" / "metadata.parquet"
    
    # Check if paths exist as files, if not try alternative formats
    if not features_stage2_path.exists():
        # Try .arrow format
        alt_path = features_stage2_path.with_suffix('.arrow')
        if alt_path.exists():
            features_stage2_path = alt_path
        else:
            # Try .csv format
            alt_path = features_stage2_path.with_suffix('.csv')
            if alt_path.exists():
                features_stage2_path = alt_path
    
    if not features_stage4_path.exists():
        # Try .arrow format
        alt_path = features_stage4_path.with_suffix('.arrow')
        if alt_path.exists():
            features_stage4_path = alt_path
        else:
            # Try .csv format
            alt_path = features_stage4_path.with_suffix('.csv')
            if alt_path.exists():
                features_stage4_path = alt_path
    
    logger.info("")
    logger.info("Checking Stage 2 features (required for baseline models)...")
    stage2_valid = check_features_metadata(str(features_stage2_path), "Stage 2", min_rows=100)
    
    logger.info("")
    logger.info("Checking Stage 4 features (required for stage2_stage4 models)...")
    stage4_valid = check_features_metadata(str(features_stage4_path), "Stage 4", min_rows=100)
    
    logger.info("")
    logger.info("=" * 80)
    
    # Determine overall status
    if stage2_valid and stage4_valid:
        logger.info("✅ Feature sanity check PASSED")
        logger.info("  - Stage 2 features: Valid")
        logger.info("  - Stage 4 features: Valid")
        return 0
    elif stage2_valid:
        logger.info("⚠️  Feature sanity check PASSED (with warnings)")
        logger.info("  - Stage 2 features: Valid")
        logger.info("  - Stage 4 features: Missing or invalid (may be OK for stage2-only models)")
        return 0  # Don't fail if Stage 4 is missing - some models don't need it
    else:
        logger.error("❌ Feature sanity check FAILED")
        logger.error("  - Stage 2 features: Missing or invalid (REQUIRED)")
        logger.error("  - Stage 4 features: Missing or invalid")
        logger.error("")
        logger.error("Please ensure Stage 2 feature extraction completed successfully.")
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.error("\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)
