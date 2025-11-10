# ACID-Style: An Adaptive Condition Injection Diffusion Model for Arbitrary Style Transfer
Code for the AAAI 2026 paper "ACID-Style: An Adaptive Condition Injection Diffusion Model for Arbitrary Style Transfer".

## Environment

```
install python=3.10.14
pip3 install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 xformers==0.0.26.post1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Checkpoint for Training and Inference


download the files and put them into ./checkpoints before the training and inference

v2-1_512-ema-pruned.ckpt
download from https://huggingface.co/stabilityai/stable-diffusion-2-1-base/tree/main

open_clip_pytorch_model.bin
download from https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/tree/main

vgg_normalised.pth
download from https://github.com/naoto0804/pytorch-AdaIN/releases/tag/v0.0.0

ACID-Style checkpoint ckpt.pth
download from https://huggingface.co/Kinghuea/ACID-Style/tree/main

## Training

```
python main.py --train --base configs/v4-train_finetune.yaml --gpus GPU_ID(0,1,2,3), --name <log_name>
```
#### Resume
```
python main.py --train --base configs/v4-train_finetune.yaml --gpus GPU_ID(0,1,2,3),  --resume checkpoints_path
```

## Inference
```
python test_loss_whole_content.py
```
