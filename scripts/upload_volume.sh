#!/bin/bash
set -e

# Upload local normalized data to Modal volume.
# Run after download_data.py to sync new images to the remote volume.
#
# Usage:
#   bash scripts/upload_volume.sh              # upload all dirs
#   bash scripts/upload_volume.sh modern_flux  # upload specific dir only

NORM_DIR="data/normalized"

if [ ! -d "$NORM_DIR" ]; then
  echo "ERROR: $NORM_DIR not found. Run download_data.py first."
  exit 1
fi

if [ -n "$1" ]; then
  dirs="$@"
else
  dirs=$(ls "$NORM_DIR")
fi

for dir in $dirs; do
  if [ -d "$NORM_DIR/$dir" ]; then
    count=$(find "$NORM_DIR/$dir" -type f | wc -l | tr -d ' ')
    echo "Uploading $dir ($count files)..."
    modal volume put cs231n-data "$NORM_DIR/$dir" "normalized/$dir" --force
  fi
done

echo "Done. Upload complete."
