#!/bin/bash
# Quick script to sync just the __init__.py file to the cluster

echo "Syncing lib/models/__init__.py to Great Lakes cluster..."

rsync -avh \
  lib/models/__init__.py \
  santoshd@greatlakes.arc-ts.umich.edu:/scratch/si670f25_class_root/si670f25_class/santoshd/fvc/lib/models/

echo ""
echo "File synced! Verifying on remote..."

ssh santoshd@greatlakes.arc-ts.umich.edu "head -45 /scratch/si670f25_class_root/si670f25_class/santoshd/fvc/lib/models/__init__.py | tail -20"

echo ""
echo "Done!"
