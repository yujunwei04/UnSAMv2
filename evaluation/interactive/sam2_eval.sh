# the sam2 model config path should be relative path under sam2 folder
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nproc_per_node=8 \
    --master_port=45341 \
    sam2_evaluate.py NoBRS \
    --datasets GrabCut,Berkeley,DAVIS,PartImageNet,PascalPart,SBD,SA1B \
    --logs-path /home/yujunwei/UnSAMv2/evaluation/interactive/outputs/evaluation_logs \
    --checkpoint /home/yujunwei/UnSAMv2/sam2/checkpoints/unsamv2_plus_ckpt.pt \
    --sam-cfg-path "configs/unsamv2_small.yaml" \
    --sam-type SAM2 \
    --graco \
    --distributed