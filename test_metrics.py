import os 
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
H = W = 512
SEED = 23
DEVICE = device = 'cuda:0'

OUTPUT_PATH = "./result/test_result/ddim20"

# python test_metrics.py
import argparse 
parser = argparse.ArgumentParser(description=' ')
parser.add_argument('--output_path', default=None,type=str)
args = parser.parse_args()

if args.output_path is not None:
    OUTPUT_PATH = args.output_path
print(OUTPUT_PATH)

style_dir="./data/style/"
content_dir="./data/content/"
vgg19_path = "./checkpoints/vgg_normalised.pth"
csd_path = "./CSD_Score/models/checkpoint.pth"

from models.losses import lpips
import argparse
import os,random
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
import torchvision
import cv2 
import time
import numpy as np
import torchvision.transforms as transforms
import argparse 
from torchvision.utils import save_image
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

def open_tensor_image(image_path, size=(H,W)):
    image = Image.open(image_path)
    if not image.mode == "RGB":
        image = image.convert("RGB")
    transf = transforms.Compose([
			transforms.ToTensor(),
			transforms.Resize(H),
			transforms.CenterCrop(size),
			])
    image = transf(image)[None,...]
    return image

def open_PIL_image(image_path, size=(H,W)):
    image = Image.open(image_path)
    transf = transforms.Compose([
			transforms.Resize(H),
			transforms.CenterCrop(size),
			])
    image = transf(image)
    return image

output_dir = Path(OUTPUT_PATH)
output_dir.mkdir(exist_ok=True, parents=True)

assert (style_dir)
assert (content_dir)

content_dir = Path(content_dir)
content_paths = [f for f in content_dir.glob('*')]

style_dir = Path(style_dir)
style_paths = [f for f in style_dir.glob('*')]

content_loss=0
style_loss=0
content_metric_ssim = 0
content_metric_lpips = 0
style_metric_gram_5 = 0
style_metric_gram_4 = 0
clip_style_score_total = 0
clip_content_score_total =0
csd_score_total = 0
i=0

encoder = lpips.VGG19(checkpoint=vgg19_path)
encoder.to(DEVICE)


ssim_func = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
lpips_func = LearnedPerceptualImagePatchSimilarity(net_type="alex").to(DEVICE)


import clip
clip_model, preprocess = clip.load("ViT-L/14")
clip_model.eval()
clip_model= clip_model.to(DEVICE)

from CSD_Score.model import CSD_CLIP,convert_state_dict

csd_model = CSD_CLIP('vit_large', "default")

checkpoint = torch.load(csd_path, map_location="cpu")
state_dict = convert_state_dict(checkpoint['model_state_dict'])
csd_model.load_state_dict(state_dict, strict=False)
normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
csd_preprocess = transforms.Compose([
				transforms.Resize(size=224, interpolation=torchvision.transforms.functional.InterpolationMode.BICUBIC),
				transforms.CenterCrop(224),
				transforms.ToTensor(),
				normalize,
			])
torch.backends.cudnn.benchmark = True
csd_model.eval()
csd_model.cuda()


# -----------------------start------------------------
for content_path in content_paths:
	for style_path in style_paths:
		with torch.no_grad():
			i=i+1 
			output_name = output_dir / '{:s}_stylized_{:s}.jpg'.format(content_path.stem, style_path.stem)

			content = open_tensor_image(str(content_path)).to(DEVICE)
			style = open_tensor_image(str(style_path)).to(DEVICE)
			output = open_tensor_image(output_name).to(DEVICE)

			# loss_content, loss_style
			loss_c, loss_s = encoder(content, style, output)
			content_loss+=loss_c
			style_loss+=loss_s
			
			# clip content score
			content_image = preprocess(open_PIL_image(str(content_path))).unsqueeze(0).to(DEVICE)
			output_image = preprocess(open_PIL_image(output_name)).unsqueeze(0).to(DEVICE)

			content_image_features = clip_model.encode_image(content_image)
			content_image_features /= content_image_features.norm(dim=-1, keepdim=True)
			output_image_features = clip_model.encode_image(output_image)
			output_image_features /= output_image_features.norm(dim=-1, keepdim=True)
			clip_content_score = 100 * (content_image_features @ output_image_features.T)
			clip_content_score_total += clip_content_score

			# csd 
			csd_style = csd_preprocess(Image.open(str(style_path))).unsqueeze(0).cuda()
			csd_output = csd_preprocess(Image.open(str(output_name))).unsqueeze(0).cuda()
			_, _, csd_style_feats = csd_model(csd_style)
			_, _, csd_output_feats = csd_model(csd_output)
			csd_score = csd_style_feats @ csd_output_feats.T
			csd_score_total+=csd_score

			print(f"i:{i},lc:{loss_c},ls:{loss_s}, unopen_clip_content:{clip_content_score},csd:{csd_score}")
			
			# print(f"i:{i},csd:{csd_score}")

output_str = OUTPUT_PATH + "\n"
content_loss/=i
style_loss/=i
output_str += f'  ||  lc:{content_loss}'
output_str += f'  ||  ls:{style_loss}'

clip_content_score_total/=i
output_str += f'  ||  unopen_clip_content_score:{clip_content_score_total}'

csd_score_total/=i
output_str += f'  ||  csd_score:{csd_score_total}'

print(output_str)
with open(output_dir/'log.txt',"a+") as f:
    f.write(output_str)