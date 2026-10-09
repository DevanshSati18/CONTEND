#!/usr/bin/env bash
# COCO 2017 validation images (~778 MB) + annotations (~241 MB) into data/coco/
#   bash scripts/01_download_coco.sh
set -euo pipefail
mkdir -p data/coco && cd data/coco
[ -f val2017.zip ] || wget -c http://images.cocodataset.org/zips/val2017.zip
[ -f annotations_trainval2017.zip ] || wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip
[ -d val2017 ] || unzip -q val2017.zip
[ -f annotations/instances_val2017.json ] || unzip -q annotations_trainval2017.zip
echo "images: $(ls val2017 | wc -l) (expected 5000)"
