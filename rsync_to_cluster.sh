#!/bin/bash
# Rsync script to sync fvc project to Great Lakes cluster
# This script excludes unnecessary files while preserving important ones

SOURCE_DIR="/Users/santoshdesai/Downloads/fvc/"
DEST_HOST="santoshd@greatlakes.arc-ts.umich.edu"
DEST_PATH="/scratch/si670f25_class_root/si670f25_class/santoshd/fvc/"

echo "Syncing fvc project to Great Lakes cluster..."
echo "Source: $SOURCE_DIR"
echo "Destination: $DEST_HOST:$DEST_PATH"
echo ""

rsync -avh --delete \
  --exclude '.git/' \
  --exclude '.venv/' \
  --exclude 'venv/' \
  --exclude 'archive/' \
  --exclude '__pycache__/' \
  --exclude '**/__pycache__/' \
  --exclude '.pytest_cache/' \
  --exclude '.mypy_cache/' \
  --exclude '.cache/' \
  --exclude '.pip-cache/' \
  --exclude 'models/' \
  --exclude 'data/' \
  --exclude 'logs/' \
  --exclude '.DS_Store' \
  --exclude '*.pyc' \
  --exclude '*.pyo' \
  --exclude '*.pyd' \
  --exclude '.Python' \
  --exclude '*.so' \
  --exclude '*.egg-info/' \
  --exclude '*.egg' \
  --exclude '.ipynb_checkpoints/' \
  --exclude '*.swp' \
  --exclude '*.swo' \
  --exclude '*~' \
  --exclude '*.tmp' \
  --exclude '*.temp' \
  --exclude '*.bak' \
  --exclude 'Thumbs.db' \
  "$SOURCE_DIR" \
  "$DEST_HOST:$DEST_PATH"

echo ""
echo "Sync complete!"
