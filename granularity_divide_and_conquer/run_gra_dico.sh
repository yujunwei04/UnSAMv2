#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python gra_divide_conquer.py \
--input-dir /path/to/input/image \
--output-dir /path/to/save/output \
--confidence-threshold 0.3 \
--thetas 0.9 0.8 0.7 0.6 0.5 \
--NMS-iou 0.8 \
--patch-size 16 \
--local-size 768 \
--postprocess True \
--preprocess True \
--granularity True \
--start-id 1 \
--end-id 5 \
--opts MODEL.WEIGHTS /path/to/cutler_checkpoint
