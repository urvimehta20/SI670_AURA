#!/bin/bash
# Sync ALL code files to Great Lakes cluster
# Excludes only: cache files, virtual environments, data, models, logs, and OS files

SOURCE_DIR="/Users/santoshdesai/Downloads/fvc/"
DEST_HOST="santoshd@greatlakes.arc-ts.umich.edu"
DEST_PATH="/scratch/si670f25_class_root/si670f25_class/santoshd/fvc/"

echo "=========================================="
echo "Syncing ALL code to Great Lakes cluster"
echo "=========================================="
echo "Source: $SOURCE_DIR"
echo "Destination: $DEST_HOST:$DEST_PATH"
echo ""

rsync -avh --progress --stats --delete \
  \
  --exclude '.git/' \
  \
  --exclude '.venv/' \
  --exclude 'venv/' \
  --exclude 'env/' \
  --exclude 'ENV/' \
  \
  --exclude 'archive/' \
  \
  --exclude '__pycache__/' \
  --exclude '**/__pycache__/' \
  --exclude '.pytest_cache/' \
  --exclude '.mypy_cache/' \
  --exclude '.cache/' \
  --exclude '.pip-cache/' \
  \
  --exclude 'models/' \
  --exclude 'data/' \
  --exclude 'logs/' \
  \
  --exclude '.DS_Store' \
  --exclude 'Thumbs.db' \
  \
  --exclude '*.pyc' \
  --exclude '*.pyo' \
  --exclude '*.pyd' \
  --exclude '.Python' \
  --exclude '*.so' \
  --exclude '*.egg-info/' \
  --exclude '*.egg' \
  \
  --exclude '.ipynb_checkpoints/' \
  --exclude '*.swp' \
  --exclude '*.swo' \
  --exclude '*~' \
  --exclude '*.tmp' \
  --exclude '*.temp' \
  --exclude '*.bak' \
  \
  "$SOURCE_DIR" \
  "$DEST_HOST:$DEST_PATH"

echo ""
echo "=========================================="
echo "Sync complete!"
echo "=========================================="
