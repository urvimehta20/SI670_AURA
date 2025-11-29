"""CLI entry point for FVC data preparation"""
from .config import FVCConfig
from .build_index import build_video_index


def run_default_prep():
    """Run default data preparation pipeline"""
    cfg = FVCConfig()
    
    print("=" * 60)
    print("FVC Dataset Preparation")
    print("=" * 60)
    print(f"Root directory: {cfg.root_dir}")
    print(f"Metadata directory: {cfg.metadata_dir}")
    print(f"Data directory: {cfg.data_dir}")
    print()
    
    build_video_index(
        cfg,
        drop_duplicates=False,  # Keep all duplicate videos (grouped by dup_group for split awareness)
        compute_stats=True  # compute comprehensive video stats using ffprobe
    )


if __name__ == "__main__":
    run_default_prep()

