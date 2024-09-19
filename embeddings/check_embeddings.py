#%%Imports
import torch
import torch.nn as nn
import tqdm as tqdm
import os
import timm
from PIL import Image
import numpy as np
from torch.utils.tensorboard import SummaryWriter
#%% Paths
TRAIN_PATH = r"D:\plantclef2024\PlantCLEF2024singleplanttrainingdata/train"
AVG = r"D:\plantclef2024\embeds\avg_all.pt"
CLASSES = r"D:\plantclef2024\PlantCLEF2024singleplanttrainingdata/train"

#%% Load DinoV2
print(f"is cuda enabled: {torch.cuda.is_available()}")
model = timm.create_model(
    'vit_base_patch14_reg4_dinov2.lvd142m',
    pretrained=True)
device = torch.device('cuda')
model = model.eval()
print("Model loaded")
model = model.to(device)

data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)


IMG = r"D:\plantclef2024\PlantCLEF2024singleplanttrainingdata\train\1355870\00d2b35cf1a0f97483392e96206bae8c3496f733.jpg"
img = Image.open(IMG)
output = model.forward_features(transforms(img).unsqueeze(0).to(device))
output = model.forward_head(output, pre_logits=True)
img.close()

img2 = Image.open(IMG)
output2 = model.forward_features(transforms(img2).unsqueeze(0).to(device))
output2 = model.forward_head(output2, pre_logits=True)
img2.close()

print(output)
print(output2)
print(torch.equal(output, output2))
