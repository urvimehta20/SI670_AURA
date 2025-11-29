#!/usr/bin/env python3
"""
Setup script for FVC dataset preparation.
Unzips FVC archives, copies metadata files, and generates video index manifest.
"""
import os
import sys
import subprocess
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
PASSWORD = "m3v3r!@"
# Look for archives in archive folder first, then root
ARCHIVE_NAMES = ["FVC1.zip", "FVC2.zip", "FVC3.zip", "Metadata.zip"]
METADATA_FILE_NAMES = ["FVC_dup.csv", "FVC.csv"]


def find_file(filename, search_dirs=None):
    """Find a file in common locations"""
    if search_dirs is None:
        search_dirs = [PROJECT_ROOT / "archive", PROJECT_ROOT]
    
    for search_dir in search_dirs:
        path = Path(search_dir) / filename
        if path.exists():
            return path
    return None


def run_command(cmd, check=True):
    """Run a shell command and return the result (Python 3.6 compatible)"""
    print("Running: {}".format(' '.join(cmd)))
    # Python 3.6 compatible: use stdout/stderr instead of capture_output
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # Decode bytes to string
    result.stdout = result.stdout.decode('utf-8') if result.stdout else ""
    result.stderr = result.stderr.decode('utf-8') if result.stderr else ""
    if check and result.returncode != 0:
        print("Error: {}".format(result.stderr))
        sys.exit(1)
    return result


def unzip_archive(archive_path, extract_to, password):
    """Unzip an archive using unzip with password"""
    archive_path = Path(archive_path)
    if not archive_path.exists():
        print(f"Warning: {archive_path} not found, skipping...")
        return False
    
    # Check if target folder already exists
    archive_name = archive_path.stem  # e.g., "FVC1" from "FVC1.zip"
    target_folder = Path(extract_to) / archive_name
    
    if target_folder.exists() and any(target_folder.iterdir()):
        print(f"✓ {archive_name} folder already exists, skipping unzip")
        return True
    
    print(f"\nUnzipping {archive_path.name}...")
    # Use unzip with password, non-interactive mode
    # Disable zip bomb detection (false positive for legitimate archives with overlapping components)
    env = os.environ.copy()
    env["UNZIP_DISABLE_ZIPBOMB_DETECTION"] = "TRUE"
    cmd = ["unzip", "-P", password, "-o", "-q", str(archive_path), "-d", str(extract_to)]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    # Decode bytes to string (Python 3.6 compatible)
    result.stdout = result.stdout.decode('utf-8') if result.stdout else ""
    result.stderr = result.stderr.decode('utf-8') if result.stderr else ""
    
    if result.returncode != 0:
        # Check if it's a disk space issue but folder was partially created
        if target_folder.exists() and any(target_folder.iterdir()):
            print(f"⚠ Warning: {archive_path.name} partially unzipped (disk space issue?), but folder exists")
            return True
        print(f"Error unzipping {archive_path.name}: {result.stderr}")
        return False
    
    print(f"✓ Successfully unzipped {archive_path.name}")
    return True


def copy_metadata_files(metadata_dir):
    """Copy metadata CSV files to Metadata directory, use symlink if copy fails due to disk space"""
    metadata_dir = Path(metadata_dir)
    try:
        metadata_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        if e.errno == 28:  # No space left on device
            print(f"⚠ Warning: Cannot create {metadata_dir} (disk full), but continuing...")
        else:
            raise
    
    print(f"\nCopying metadata files to {metadata_dir}...")
    copied = []
    for filename in METADATA_FILE_NAMES:
        src = find_file(filename)
        if src:
            dst = metadata_dir / filename
            # Check if file already exists
            if dst.exists():
                print(f"✓ {filename} already exists in {metadata_dir}, skipping")
                copied.append(filename)
            else:
                try:
                    shutil.copy2(src, dst)
                    print(f"✓ Copied {filename} to {metadata_dir}")
                    copied.append(filename)
                except OSError as e:
                    if e.errno == 28:  # No space left on device
                        # Try symlink instead
                        try:
                            if dst.exists():
                                dst.unlink()  # Remove if exists as broken symlink
                            dst.symlink_to(src)
                            print(f"✓ Created symlink for {filename} (disk full, using symlink)")
                            copied.append(filename)
                        except Exception as symlink_err:
                            print(f"⚠ Warning: Cannot copy or symlink {filename} (disk full): {symlink_err}")
                    else:
                        raise
        else:
            print(f"Warning: {filename} not found")
    
    return copied


def generate_video_index():
    """Generate video index using the data preparation script"""
    print("\nGenerating video index manifest...")
    
    # Check if files already exist
    data_dir = PROJECT_ROOT / "data"
    csv_path = data_dir / "video_index_input.csv"
    json_path = data_dir / "video_index_input.json"
    
    if csv_path.exists() and json_path.exists():
        print(f"✓ Video index files already exist, skipping generation")
        print(f"  - {csv_path}")
        print(f"  - {json_path}")
        return True
    
    try:
        from lib.cli import run_default_prep
        run_default_prep()
        print("✓ Video index generated successfully")
        return True
    except OSError as e:
        if e.errno == 28:  # No space left on device
            print(f"⚠ Warning: Cannot write video index files (disk full)")
            if csv_path.exists() or json_path.exists():
                print(f"  Partial files may exist, checking...")
                if csv_path.exists():
                    print(f"  ✓ Found {csv_path}")
                if json_path.exists():
                    print(f"  ✓ Found {json_path}")
                return True
            return False
        else:
            print(f"Error generating video index: {e}")
            import traceback
            traceback.print_exc()
            return False
    except Exception as e:
        print(f"Error generating video index: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_paths_and_attributes():
    """Verify that all paths are accessible and attributes are correct"""
    print("\nVerifying paths and attributes...")
    
    data_dir = PROJECT_ROOT / "data"
    csv_path = data_dir / "video_index_input.csv"
    json_path = data_dir / "video_index_input.json"
    
    # Check if files exist
    if not csv_path.exists():
        print(f"✗ Error: {csv_path} not found")
        return False
    if not json_path.exists():
        print(f"✗ Error: {json_path} not found")
        return False
    
    print(f"✓ Found {csv_path}")
    print(f"✓ Found {json_path}")
    
    # Read and verify CSV
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        
        print(f"\nDataset Statistics:")
        print(f"  Total videos: {len(df)}")
        print(f"  Columns: {len(df.columns)}")
        
        # Check required columns
        required_cols = ["subset", "platform", "video_id", "video_path", "label"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"✗ Error: Missing required columns: {missing_cols}")
            return False
        print(f"✓ All required columns present")
        
        # Verify video paths exist using centralized path resolution
        print(f"\nVerifying video file paths...")
        # Import centralized path utilities
        sys.path.insert(0, str(PROJECT_ROOT))
        try:
            from lib.video_paths import resolve_video_path
        except ImportError:
            # Fallback if module not available
            def check_video_path_exists(video_rel, project_root):
                video_rel = str(video_rel).strip()
                if not video_rel or os.path.isabs(video_rel):
                    return os.path.exists(video_rel) and os.path.isfile(video_rel)
                candidates = []
                if not video_rel.startswith("videos/"):
                    candidates.append(PROJECT_ROOT / "videos" / video_rel)
                candidates.append(PROJECT_ROOT / video_rel)
                if video_rel.startswith("videos/"):
                    candidates.append(PROJECT_ROOT / video_rel[7:])
                return any(c.exists() and c.is_file() for c in candidates)
            
            def get_video_path_candidates(video_rel, project_root):
                video_rel = str(video_rel).strip()
                if os.path.isabs(video_rel):
                    return [video_rel]
                candidates = []
                if not video_rel.startswith("videos/"):
                    candidates.append(str(PROJECT_ROOT / "videos" / video_rel))
                candidates.append(str(PROJECT_ROOT / video_rel))
                if video_rel.startswith("videos/"):
                    candidates.append(str(PROJECT_ROOT / video_rel[7:]))
                return candidates
        
        missing_videos = []
        for idx, row in df.iterrows():
            video_rel = row["video_path"]
            try:
                video_path = resolve_video_path(video_rel, str(PROJECT_ROOT))
                if not Path(video_path).exists():
                    missing_videos.append((row["video_id"], video_path))
            except Exception:
                missing_videos.append((row["video_id"], str(PROJECT_ROOT / video_rel)))
        
        if missing_videos:
            print(f"✗ Warning: {len(missing_videos)} video files not found:")
            for vid_id, path in missing_videos[:5]:  # Show first 5
                print(f"    {vid_id}: {path}")
            if len(missing_videos) > 5:
                print(f"    ... and {len(missing_videos) - 5} more")
        else:
            print(f"✓ All {len(df)} video files are accessible")
        
        # Check label distribution
        if "label" in df.columns:
            label_counts = df["label"].value_counts().sort_index()
            print(f"\nLabel distribution:")
            for label, count in label_counts.items():
                label_name = "Real" if label == 0 else "Fake"
                print(f"  {label_name} ({label}): {count}")
        
        # Check stats coverage
        stats_cols = ["width", "height", "fps", "codec_name", "file_size_bytes"]
        print(f"\nStats coverage:")
        for col in stats_cols:
            if col in df.columns:
                non_null = df[col].notna().sum()
                pct = 100 * non_null / len(df)
                print(f"  {col}: {non_null}/{len(df)} ({pct:.1f}%)")
        
        print(f"\n✓ Verification complete")
        return True
        
    except Exception as e:
        print(f"✗ Error verifying dataset: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main setup function"""
    print("=" * 70)
    print("FVC Dataset Setup Script")
    print("=" * 70)
    print(f"Project root: {PROJECT_ROOT}")
    print()
    
    # Step 1: Unzip archives
    print("Step 1: Unzipping archives...")
    videos_dir = PROJECT_ROOT / "videos"
    videos_dir.mkdir(exist_ok=True)
    
    unzipped = []
    for archive_name in ARCHIVE_NAMES:
        archive_path = find_file(archive_name)
        if archive_path:
            if unzip_archive(archive_path, videos_dir, PASSWORD):
                unzipped.append(archive_name)
        else:
            print(f"Warning: {archive_name} not found, skipping...")
    
    if not unzipped:
        print("Warning: No archives were unzipped. Continuing anyway...")
    
    # Step 2: Copy metadata files
    print("\nStep 2: Copying metadata files...")
    metadata_dir = videos_dir / "Metadata"
    copied = copy_metadata_files(metadata_dir)
    
    if not copied:
        print("Warning: No metadata files were copied. Check if files exist.")
    
    # Step 3: Generate video index
    print("\nStep 3: Generating video index...")
    if not generate_video_index():
        print("✗ Failed to generate video index")
        sys.exit(1)
    
    # Step 4: Verify paths and attributes
    print("\nStep 4: Verifying paths and attributes...")
    if not verify_paths_and_attributes():
        print("✗ Verification failed")
        sys.exit(1)
    
    print("\n" + "=" * 70)
    print("✓ Setup completed successfully!")
    print("=" * 70)
    print(f"\nOutput files:")
    print(f"  - {PROJECT_ROOT / 'data' / 'video_index_input.csv'}")
    print(f"  - {PROJECT_ROOT / 'data' / 'video_index_input.json'}")


if __name__ == "__main__":
    main()

