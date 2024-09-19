#%%Imports
import torch
import patch_images_unblended as img_gen
import cv2
import os
import timm
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
#%% Paths
TRAIN_PATH = r"C:\Users\ASUS\Desktop\miniclef\images"
CLASSES = r"C:\Users\ASUS\Desktop\miniclef\images"
CHKPNT = r"C:\Users\ASUS\Desktop\miniclef\model\vit_base_patch14_reg4_dinov2_lvd142m_pc24_onlyclassifier_then_all\model_best.pth.tar"
#%% Parameters
PATCH_HEIGHT = 500
PATCH_WIDTH = 500
MIN_CLASS = 1
MAX_CLASS = 1
MIN_SIZE = 150
batch_size = 8
#%% 

classes = os.listdir(CLASSES)
no_classes = len(classes)
#%% Patch generator
args = TRAIN_PATH, PATCH_HEIGHT, PATCH_WIDTH, MIN_CLASS, MAX_CLASS, MIN_SIZE
img_generator = img_gen.patched_img(*args)

#%% Load DinoV2
print(f"Using GPU: {torch.cuda.is_available()}")
model = timm.create_model(
    'vit_base_patch14_reg4_dinov2.lvd142m',
    pretrained=False,
    num_classes=7806,
    checkpoint_path=CHKPNT
    )
device = torch.device('cuda')
model = model.eval()
print("Model loaded")
model = model.to(device)

data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)

#%% Get embeddings

patch, label, img_path = img_generator.generate_patched_img()
print(f"class label: {label[0]}")
print(f"img_path: {img_path[0]}")

selected = Image.open(img_path[0])
plt.figure()
plt.title("From folder")
plt.imshow(selected)
plt.show()

recolor = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB) #FUCKING IMPORTANT DAMN IT 
generator = Image.fromarray(recolor)
plt.figure()
plt.title("From generator")
plt.imshow(generator)
plt.show()

output = model.forward_features(transforms(generator).unsqueeze(0).to(device))
gen_embed_torch = model.forward_head(output, pre_logits=True)
generator.close()

output2 = model.forward_features(transforms(selected).unsqueeze(0).to(device))
folder_embed_torch = model.forward_head(output2, pre_logits=True)
selected.close()
cos = torch.nn.CosineSimilarity()
print(f"cosine sim: {cos(gen_embed_torch, folder_embed_torch)}") #passes >0.85