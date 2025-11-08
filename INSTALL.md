## Installation
```bash
conda create --name UnSAMv2 python=3.10 -y
conda activate UnSAMv2
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 \
--index-url https://download.pytorch.org/whl/cu121

git clone https://github.com/yujunwei04/UnSAMv2.git
cd UnSAMv2
pip install -r requirements.txt
```

DINOv3 and Detectron2 are only used in granularity divide-and-conquer pipeline. Feel free to skip them if you just want to try UnSAMv2 inference.

- Setup DINOv3
```bash
cd granularity_divide_and_conquer
git clone https://github.com/facebookresearch/dinov3.git
# Follow DINOv3 repo to download ViT-B/16 model weights
cd ..
```
<!-- - Follow DINOv3 [link](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/) to download ViT-B/16 model weights -->

- Setup Detectron2
```bash
python -m pip install 'git+https://github.com/MaureenZOU/detectron2-xyz.git'
cd granularity_divide_and_conquer/mask2former/modeling/pixel_decoder/ops
sh make.sh
```


- Setup SAM-2
```bash
cd sam2
pip install -e ".[notebooks]"

# Ignore the ERROR: "detectron2 0.6 requires iopath<0.1.10,>=0.1.7, but you have iopath 0.1.10 which is incompatible."

# We use sam2.1_hiera_small
cd checkpoints
bash download_ckpts.sh
cd ../..
```