#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

python coco_annotator.py \
    --image-dir /path/to/your/image/dir \
    --model-config configs/unsamv2_small.yaml \
    --checkpoint /path/to/your/checkpoint.pt \
    --output-dir /path/to/your/output/dir \
    --granularities 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --start-index 0 \
    --end-index 1000
    "$@"