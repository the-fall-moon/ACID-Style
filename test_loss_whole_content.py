import os 
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

CFG_PATH = './config/v4-train_finetune.yaml'
CKPT_PATH = "path to your checkpoint"

OUTPUT_PATH = "./result/test_result"
style_dir="./data/style/"
content_dir="./data/content/"

SAMPLER="ddim"
DDIM_STEPS = 20
H = W = 512
ETA = 1.
SEED = 23
DEVICE = 'cuda:0'

OUTPUT_PATH = OUTPUT_PATH+f"/ddim{DDIM_STEPS}"

import sys
sys.path.append('../')

import random
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning import seed_everything
from PIL import ImageDraw, ImageFont, Image
import matplotlib.pyplot as plt
from einops import rearrange
from omegaconf import OmegaConf
import albumentations
from main import instantiate_from_config
from pathlib import Path
from torchvision.utils import save_image

seed_everything(SEED)

config = OmegaConf.load(CFG_PATH)
config.model.params.resume_ckpt_path = CKPT_PATH
config.model.params.sampler = SAMPLER
config.model.params.image_size = H
config.model.params.eta = ETA
model = instantiate_from_config(config.model)
model = model.to(DEVICE)


def open_image(image_path, size=(W, H)):
    image = Image.open(image_path)
    if not image.mode == "RGB":
        image = image.convert("RGB")
    image = image.resize(size)
    image = np.array(image).astype(np.uint8)
    image = (image/255).astype(np.float32)
    # image = rearrange(image, 'h w c -> c h w')
    return torch.from_numpy(image)


def tensor_to_rgb(x):
    return torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)



output_dir = Path(OUTPUT_PATH)
output_dir.mkdir(exist_ok=True, parents=True)

assert (style_dir)
assert (content_dir)

content_dir = Path(content_dir)
content_paths = [f for f in content_dir.glob('*')]

style_dir = Path(style_dir)
style_paths = [f for f in style_dir.glob('*')]

i=0
# -----------------------start------------------------
for content_path in content_paths:
    for style_path in style_paths:
        with torch.no_grad():
            output_name = output_dir / '{:s}_stylized_{:s}.jpg'.format(content_path.stem, style_path.stem)
            if os.path.exists(output_name):
                print(output_name," has existed!")
                continue
            
            content = open_image(str(content_path))[None,:].to(DEVICE)
            style = open_image(str(style_path))[None,:].to(DEVICE)
            output = model.sample_log(content, style, batch_size=1, ddim_steps=DDIM_STEPS)
            # content/style ---> z ---> stylized
            print(f"i:{i}")
            i=i+1
            save_image(output, str(output_name))