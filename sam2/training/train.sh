CUDA_VISIBLE_DEVICES=4,5 python training/train.py \
    -c configs/sam2.1_training/0827_unsup_dinov3.yaml \
    --use-cluster 0 \
    --num-gpus 2