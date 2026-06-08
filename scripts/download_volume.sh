#!/bin/bash
set -e
rm -rf /tmp/modal-normalized
mkdir -p /tmp/modal-normalized

dirs="cf_eval genimage_adm genimage_biggan genimage_glide genimage_midjourney genimage_vqdm genimage_wukong imagenet_real_A imagenet_real_B modern_flux modern_midjourney_modern modern_real_coco modern_real_ffhq modern_sd35"

for dir in $dirs; do
  echo "Downloading $dir..."
  modal volume get cs231n-data "normalized/$dir" "/tmp/modal-normalized/$dir" --force
done

echo "Done. Files in /tmp/modal-normalized/"
