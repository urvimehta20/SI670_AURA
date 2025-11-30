#!/usr/bin/env python3
"""
Validation script for Stage 1 augmentations.

Checks:
1. SLURM output files (.out, .err) for errors
2. Log files for failures
3. Physical presence of all expected augmentation files
4. File integrity (corruption checks)

Usage:
    python src/scripts/validate_augmentations.py
    python src/scripts/validate_augmentations.py --project-root /path/to/project
    python src/scripts/validate_augmentations.py --output-dir data/augmented_videos
"""

from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Set
import polars as pl
import av

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from lib.data import load_metadata
from lib.utils.paths import resolve_video_path




def validate_video_file(file_path: Path) -> Tuple[bool, str]:
    """
    Validate that a video file exists and is not corrupted.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not file_path.exists():
        return False, "File does not exist"
    
    if file_path.stat().st_size == 0:
        return False, "File is empty (0 bytes)"
    
    # Try to open and read video header
    try:
        container = av.open(str(file_path))
        if len(container.streams.video) == 0:
            return False, "No video stream found"
        
        # Try to read first frame to verify file is not corrupted
        try:
            for frame in container.decode(video=0):
                # If we can decode at least one frame, file is likely valid
                break
        except Exception as e:
            return False, f"Cannot decode video: {e}"
        
        container.close()
        return True, "OK"
    except Exception as e:
        return False, f"Cannot open video file: {e}"


def check_augmentation_files(
    metadata_path: Path,
    project_root: Path,
    output_dir: Path
) -> Tuple[Dict[str, Dict], List[str]]:
    """
    Check that all expected augmentation files exist and are valid.
    
    Returns:
        Tuple of (file_status_dict, missing_files_list)
        file_status_dict: {file_path: {'exists': bool, 'valid': bool, 'error': str}}
    """
    if not metadata_path.exists():
        return {}, [f"Metadata file not found: {metadata_path}"]
    
    # Load metadata
    try:
        df = load_metadata(str(metadata_path))
    except Exception as e:
        return {}, [f"Cannot load metadata: {e}"]
    
    file_status = {}
    missing_files = []
    
    # Filter to only augmented videos (is_original == False)
    if 'is_original' in df.columns:
        augmented_df = df.filter(pl.col('is_original') == False)
    else:
        # If no is_original column, assume all are augmented (shouldn't happen but handle gracefully)
        augmented_df = df
    
    print(f"Checking {augmented_df.height} augmented video files...")
    
    for row in augmented_df.iter_rows(named=True):
        video_path_rel = row.get('video_path', '')
        if not video_path_rel:
            continue
        
        # Resolve full path
        video_path = project_root / video_path_rel
        
        # Check if file exists
        exists = video_path.exists()
        
        if not exists:
            missing_files.append(video_path_rel)
            file_status[str(video_path)] = {
                'exists': False,
                'valid': False,
                'error': 'File does not exist',
                'relative_path': video_path_rel
            }
        else:
            # Validate file integrity
            is_valid, error_msg = validate_video_file(video_path)
            file_status[str(video_path)] = {
                'exists': True,
                'valid': is_valid,
                'error': error_msg if not is_valid else None,
                'relative_path': video_path_rel,
                'size_bytes': video_path.stat().st_size if exists else 0
            }
    
    return file_status, missing_files


def check_all_files_from_original_metadata(
    project_root: Path,
    output_dir: Path,
    num_augmentations: int = 10
) -> Dict[str, any]:
    """
    Check all expected files directly from original metadata, without depending on augmented_metadata.csv.
    
    Returns:
        Dictionary with missing files and statistics
    """
    # Load original metadata
    input_metadata_path = None
    for csv_name in ["FVC_dup.csv", "video_index_input.csv"]:
        candidate_path = project_root / "data" / csv_name
        if candidate_path.exists():
            input_metadata_path = candidate_path
            break
    
    if not input_metadata_path:
        return {'error': 'Cannot find input metadata file'}
    
    try:
        original_df = load_metadata(str(input_metadata_path))
        # Filter to only original videos if is_original column exists
        if 'is_original' in original_df.columns:
            original_df = original_df.filter(pl.col('is_original') == True)
    except Exception as e:
        return {'error': f'Cannot load original metadata: {e}'}
    
    missing_original_files = []
    missing_augmentation_files = []
    corrupted_files = []
    videos_with_missing_files = []
    videos_complete = 0
    
    print(f"Checking {original_df.height} videos for {num_augmentations} augmentations each...")
    
    for row in original_df.iter_rows(named=True):
        video_rel = row.get('video_path', '')
        label = row.get('label', '')
        
        if not video_rel:
            continue
        
        try:
            # Resolve original video path to extract video_id
            video_path = resolve_video_path(video_rel, project_root)
            if not Path(video_path).exists():
                videos_with_missing_files.append({
                    'video': video_rel,
                    'reason': 'Original video file not found',
                    'missing_files': [video_rel]
                })
                continue
            
            # Extract video_id from path (same logic as pipeline.py)
            video_path_obj = Path(video_path)
            video_path_parts = video_path_obj.parts
            if len(video_path_parts) >= 2:
                video_id = video_path_parts[-2]  # Parent of "video.mp4"
            else:
                # Fallback: use hash
                import hashlib
                video_id = hashlib.md5(str(video_path).encode()).hexdigest()[:12]
            
            video_id = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in video_id)
            
            # Check original copy
            original_output = output_dir / f"{video_id}_original.mp4"
            video_missing_files = []
            
            if not original_output.exists():
                missing_original_files.append({
                    'video': video_rel,
                    'video_id': video_id,
                    'expected_file': str(original_output.relative_to(project_root))
                })
                video_missing_files.append(str(original_output.relative_to(project_root)))
            else:
                # Validate original file
                is_valid, error_msg = validate_video_file(original_output)
                if not is_valid:
                    corrupted_files.append({
                        'file': str(original_output.relative_to(project_root)),
                        'error': error_msg
                    })
            
            # Check all augmentations
            for aug_idx in range(num_augmentations):
                aug_file = output_dir / f"{video_id}_aug{aug_idx}.mp4"
                if not aug_file.exists():
                    missing_augmentation_files.append({
                        'video': video_rel,
                        'video_id': video_id,
                        'augmentation_idx': aug_idx,
                        'expected_file': str(aug_file.relative_to(project_root))
                    })
                    video_missing_files.append(str(aug_file.relative_to(project_root)))
                else:
                    # Validate augmentation file
                    is_valid, error_msg = validate_video_file(aug_file)
                    if not is_valid:
                        corrupted_files.append({
                            'file': str(aug_file.relative_to(project_root)),
                            'error': error_msg
                        })
            
            if video_missing_files:
                videos_with_missing_files.append({
                    'video': video_rel,
                    'video_id': video_id,
                    'label': label,
                    'missing_count': len(video_missing_files),
                    'missing_files': video_missing_files
                })
            else:
                videos_complete += 1
                
        except Exception as e:
            videos_with_missing_files.append({
                'video': video_rel,
                'reason': f'Error checking video: {e}',
                'missing_files': []
            })
    
    return {
        'total_videos': original_df.height,
        'videos_complete': videos_complete,
        'videos_with_missing_files': len(videos_with_missing_files),
        'missing_original_files': len(missing_original_files),
        'missing_augmentation_files': len(missing_augmentation_files),
        'corrupted_files': len(corrupted_files),
        'missing_original_details': missing_original_files,
        'missing_augmentation_details': missing_augmentation_files,
        'corrupted_files_details': corrupted_files,
        'videos_with_missing_details': videos_with_missing_files
    }


def reconstruct_metadata_csv(
    project_root: Path,
    output_dir: Path,
    num_augmentations: int = 10
) -> bool:
    """
    Reconstruct augmented_metadata.csv from existing augmentation files.
    
    Returns:
        True if successful, False otherwise
    """
    metadata_path = output_dir / "augmented_metadata.csv"
    
    print(f"Reconstructing {metadata_path} from existing files...")
    
    # Load original metadata
    input_metadata_path = None
    for csv_name in ["FVC_dup.csv", "video_index_input.csv"]:
        candidate_path = project_root / "data" / csv_name
        if candidate_path.exists():
            input_metadata_path = candidate_path
            break
    
    if not input_metadata_path:
        print(f"   ✗ Error: Cannot find input metadata file")
        return False
    
    try:
        original_df = load_metadata(str(input_metadata_path))
        # Filter to only original videos if is_original column exists
        if 'is_original' in original_df.columns:
            original_df = original_df.filter(pl.col('is_original') == True)
    except Exception as e:
        print(f"   ✗ Error: Cannot load original metadata: {e}")
        return False
    
    # Create a mapping from video_id to original video path and label
    video_id_to_info = {}
    for row in original_df.iter_rows(named=True):
        video_rel = row.get('video_path', '')
        label = row.get('label', '')
        try:
            video_path = resolve_video_path(video_rel, project_root)
            if Path(video_path).exists():
                video_path_obj = Path(video_path)
                video_path_parts = video_path_obj.parts
                if len(video_path_parts) >= 2:
                    video_id = video_path_parts[-2]  # Parent of "video.mp4"
                else:
                    # Fallback: use hash
                    import hashlib
                    video_id = hashlib.md5(str(video_path).encode()).hexdigest()[:12]
                
                video_id = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in video_id)
                video_id_to_info[video_id] = {
                    'original_video': video_rel,
                    'label': label
                }
        except Exception:
            continue
    
    # Scan for all augmentation and original files
    entries = []
    
    # Find all *_aug*.mp4 files
    for aug_file in output_dir.glob("*_aug*.mp4"):
        aug_filename = aug_file.stem  # Remove .mp4 extension
        if "_aug" in aug_filename:
            video_id = aug_filename.split("_aug")[0]
            aug_idx_str = aug_filename.split("_aug")[1]
            try:
                aug_idx = int(aug_idx_str)
            except ValueError:
                continue
            
            if video_id in video_id_to_info:
                info = video_id_to_info[video_id]
                aug_path_rel = str(aug_file.relative_to(project_root))
                entries.append({
                    'video_path': aug_path_rel,
                    'label': info['label'],
                    'original_video': info['original_video'],
                    'augmentation_idx': aug_idx,
                    'is_original': False
                })
    
    # Find all *_original.mp4 files
    for orig_file in output_dir.glob("*_original.mp4"):
        orig_filename = orig_file.stem  # Remove .mp4 extension
        if orig_filename.endswith("_original"):
            video_id = orig_filename[:-9]  # Remove "_original" suffix
            
            if video_id in video_id_to_info:
                info = video_id_to_info[video_id]
                orig_path_rel = str(orig_file.relative_to(project_root))
                entries.append({
                    'video_path': orig_path_rel,
                    'label': info['label'],
                    'original_video': info['original_video'],
                    'augmentation_idx': -1,  # -1 for original videos
                    'is_original': True
                })
    
    # Write metadata CSV
    if entries:
        import csv
        with open(metadata_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["video_path", "label", "original_video", "augmentation_idx", "is_original"])
            for entry in entries:
                writer.writerow([
                    entry['video_path'],
                    entry['label'],
                    entry['original_video'],
                    entry['augmentation_idx'],
                    entry['is_original']
                ])
        print(f"   ✓ Reconstructed metadata CSV with {len(entries)} entries")
        print(f"   ✓ Saved to: {metadata_path}")
        return True
    else:
        print(f"   ✗ No augmentation files found to reconstruct metadata from")
        return False


def check_expected_vs_actual(
    metadata_path: Path,
    project_root: Path,
    output_dir: Path,
    num_augmentations: int = 10
) -> Dict[str, any]:
    """
    Compare expected augmentations (from original metadata) vs actual augmentations.
    
    Returns:
        Dictionary with comparison statistics
    """
    # Load original metadata to know how many videos should have augmentations
    input_metadata_path = None
    for csv_name in ["FVC_dup.csv", "video_index_input.csv"]:
        candidate_path = project_root / "data" / csv_name
        if candidate_path.exists():
            input_metadata_path = candidate_path
            break
    
    if not input_metadata_path:
        return {'error': 'Cannot find input metadata file'}
    
    try:
        original_df = load_metadata(str(input_metadata_path))
        original_df = original_df.filter(pl.col('is_original') == True) if 'is_original' in original_df.columns else original_df
    except Exception as e:
        return {'error': f'Cannot load original metadata: {e}'}
    
    # Load augmented metadata
    try:
        augmented_df = load_metadata(str(metadata_path))
        if 'is_original' in augmented_df.columns:
            augmented_df = augmented_df.filter(pl.col('is_original') == False)
    except Exception as e:
        return {'error': f'Cannot load augmented metadata: {e}'}
    
    # Count augmentations per original video
    if 'original_video' in augmented_df.columns:
        aug_counts = augmented_df.group_by('original_video').agg(pl.count().alias('count'))
        aug_counts_dict = {row['original_video']: row['count'] for row in aug_counts.iter_rows(named=True)}
    else:
        aug_counts_dict = {}
    
    # Check which videos have correct number of augmentations
    videos_with_correct_count = 0
    videos_with_incorrect_count = []
    videos_with_no_augmentations = []
    
    for row in original_df.iter_rows(named=True):
        video_path = row.get('video_path', '')
        expected_count = num_augmentations
        actual_count = aug_counts_dict.get(video_path, 0)
        
        if actual_count == 0:
            videos_with_no_augmentations.append(video_path)
        elif actual_count == expected_count:
            videos_with_correct_count += 1
        else:
            videos_with_incorrect_count.append({
                'video': video_path,
                'expected': expected_count,
                'actual': actual_count
            })
    
    return {
        'total_original_videos': original_df.height,
        'total_augmented_videos': augmented_df.height,
        'expected_total_augmentations': original_df.height * num_augmentations,
        'actual_total_augmentations': augmented_df.height,
        'videos_with_correct_count': videos_with_correct_count,
        'videos_with_incorrect_count': len(videos_with_incorrect_count),
        'videos_with_no_augmentations': len(videos_with_no_augmentations),
        'incorrect_count_details': videos_with_incorrect_count[:20],  # Limit to first 20
        'no_augmentations_list': videos_with_no_augmentations[:20]  # Limit to first 20
    }


def main():
    parser = argparse.ArgumentParser(
        description="Validate Stage 1 augmentations",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--project-root',
        type=str,
        default=str(project_root),
        help='Project root directory'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/augmented_videos',
        help='Output directory for augmented videos'
    )
    parser.add_argument(
        '--num-augmentations',
        type=int,
        default=10,
        help='Expected number of augmentations per video'
    )
    parser.add_argument(
        '--reconstruct-metadata',
        action='store_true',
        help='Reconstruct augmented_metadata.csv from existing files'
    )
    
    args = parser.parse_args()
    
    project_root_path = Path(args.project_root)
    output_dir_path = project_root_path / args.output_dir
    metadata_path = output_dir_path / "augmented_metadata.csv"
    
    print("=" * 80)
    print("STAGE 1 AUGMENTATION VALIDATION")
    print("=" * 80)
    print(f"Project root: {project_root_path}")
    print(f"Output directory: {output_dir_path}")
    print(f"Expected augmentations per video: {args.num_augmentations}")
    print("=" * 80)
    print()
    
    # Reconstruct metadata if requested or if it doesn't exist
    metadata_path = output_dir_path / "augmented_metadata.csv"
    if args.reconstruct_metadata or not metadata_path.exists():
        if args.reconstruct_metadata:
            print("Reconstructing metadata CSV (--reconstruct-metadata flag set)...")
        else:
            print("Metadata CSV not found, reconstructing from existing files...")
        print("-" * 80)
        success = reconstruct_metadata_csv(
            project_root_path,
            output_dir_path,
            args.num_augmentations
        )
        if success:
            print()
        else:
            print("   ⚠ Continuing validation without metadata CSV...")
            print()
    
    all_errors = []
    all_warnings = []
    
    # 1. Check all files directly from original metadata (independent of augmented_metadata.csv)
    print("1. Checking all expected files from original metadata...")
    print("-" * 80)
    file_check = check_all_files_from_original_metadata(
        project_root_path,
        output_dir_path,
        args.num_augmentations
    )
    
    if 'error' in file_check:
        print(f"   ✗ Error: {file_check['error']}")
        all_errors.append(file_check['error'])
    else:
        print(f"   Total videos to check: {file_check['total_videos']}")
        print(f"   Videos with all files present: {file_check['videos_complete']} ({file_check['videos_complete']/file_check['total_videos']*100:.1f}%)")
        print(f"   Videos with missing files: {file_check['videos_with_missing_files']}")
        print(f"   Missing original files: {file_check['missing_original_files']}")
        print(f"   Missing augmentation files: {file_check['missing_augmentation_files']}")
        print(f"   Corrupted files: {file_check['corrupted_files']}")
        
        if file_check['videos_with_missing_files'] > 0:
            print(f"\n   ✗ Videos with missing files (showing first 20):")
            for video_info in file_check['videos_with_missing_details'][:20]:
                print(f"      Video: {video_info['video']}")
                print(f"        Video ID: {video_info.get('video_id', 'unknown')}")
                print(f"        Missing {video_info.get('missing_count', len(video_info.get('missing_files', [])))} file(s):")
                for missing_file in video_info.get('missing_files', [])[:5]:  # Show first 5 missing files per video
                    print(f"          - {missing_file}")
                if len(video_info.get('missing_files', [])) > 5:
                    print(f"          ... and {len(video_info.get('missing_files', [])) - 5} more")
            
            if file_check['videos_with_missing_files'] > 20:
                print(f"      ... and {file_check['videos_with_missing_files'] - 20} more videos with missing files")
            
            # Add to errors
            all_errors.append(f"{file_check['videos_with_missing_files']} videos have missing files")
            all_errors.append(f"{file_check['missing_original_files']} original files missing")
            all_errors.append(f"{file_check['missing_augmentation_files']} augmentation files missing")
        
        if file_check['corrupted_files'] > 0:
            print(f"\n   ✗ Corrupted files (showing first 10):")
            for corrupted in file_check['corrupted_files_details'][:10]:
                print(f"      {corrupted['file']}: {corrupted['error']}")
            all_errors.append(f"{file_check['corrupted_files']} corrupted files found")
    print()
    
    # 2. Check expected vs actual augmentations (from metadata CSV if available)
    print("2. Comparing expected vs actual augmentations (from metadata CSV if available)...")
    print("-" * 80)
    if metadata_path.exists():
        comparison = check_expected_vs_actual(
            metadata_path,
            project_root_path,
            output_dir_path,
            args.num_augmentations
        )
        
        if 'error' in comparison:
            print(f"   ⚠ Warning: {comparison['error']} (using file check results above instead)")
        else:
            print(f"   Total original videos: {comparison['total_original_videos']}")
            print(f"   Total augmented videos in CSV: {comparison['actual_total_augmentations']}")
            print(f"   Expected total augmentations: {comparison['expected_total_augmentations']}")
            print(f"   Videos with correct augmentation count: {comparison['videos_with_correct_count']}")
            print(f"   Videos with incorrect augmentation count: {comparison['videos_with_incorrect_count']}")
            print(f"   Videos with no augmentations: {comparison['videos_with_no_augmentations']}")
    else:
        print("   ⚠ Metadata CSV not found - using file check results above")
    print()
    
    # 3. Check physical files (from metadata CSV if available - optional secondary check)
    print("3. Checking physical augmentation files (from metadata CSV if available)...")
    print("-" * 80)
    file_status, missing_files = check_augmentation_files(
        metadata_path,
        project_root_path,
        output_dir_path
    )
    
    if not file_status:
        if missing_files:
            print(f"   ✗ {missing_files[0]}")
            all_errors.extend(missing_files)
        else:
            print("   ✗ Cannot check files (metadata not found or invalid)")
    else:
        total_files = len(file_status)
        existing_files = sum(1 for status in file_status.values() if status['exists'])
        valid_files = sum(1 for status in file_status.values() if status['exists'] and status['valid'])
        corrupted_files = [path for path, status in file_status.items() if status['exists'] and not status['valid']]
        missing_count = len(missing_files)
        
        print(f"   Total files in metadata: {total_files}")
        print(f"   Files that exist: {existing_files} ({existing_files/total_files*100:.1f}%)")
        print(f"   Files that are valid: {valid_files} ({valid_files/total_files*100:.1f}%)")
        print(f"   Missing files: {missing_count}")
        print(f"   Corrupted files: {len(corrupted_files)}")
        
        if corrupted_files:
            print(f"   ✗ First few corrupted files:")
            for file_path in corrupted_files[:5]:
                status = file_status[file_path]
                print(f"      {status['relative_path']}: {status['error']}")
                all_errors.append(f"{status['relative_path']}: {status['error']}")
        
        if missing_files:
            print(f"   ✗ First few missing files:")
            for file_path in missing_files[:5]:
                print(f"      {file_path}")
                all_errors.append(f"Missing: {file_path}")
    print()
    
    # Summary
    print("=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    
    if all_errors:
        print(f"✗ VALIDATION FAILED: Found {len(all_errors)} error(s)")
        print("\nFirst 10 errors:")
        for i, error in enumerate(all_errors[:10], 1):
            print(f"  {i}. {error}")
        if len(all_errors) > 10:
            print(f"  ... and {len(all_errors) - 10} more errors")
        return 1
    else:
        print("✓ VALIDATION PASSED: No errors found")
        if all_warnings:
            print(f"⚠ Found {len(all_warnings)} warning(s) (see details above)")
        return 0


if __name__ == "__main__":
    sys.exit(main())

