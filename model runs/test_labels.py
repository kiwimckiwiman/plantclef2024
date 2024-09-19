#%%Imports
import torch
import patch_images_unblended as img_gen
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

#%% Parameters
PATCH_HEIGHT = 500
PATCH_WIDTH = 500
MIN_CLASS = 1
MAX_CLASS = 1
MIN_SIZE = 150
batch_size = 1000

#%% 

mean_embed = torch.load(AVG)
classes = os.listdir(CLASSES)

no_classes = len(classes)
#%% Patch generator
args = TRAIN_PATH, PATCH_HEIGHT, PATCH_WIDTH, MIN_CLASS, MAX_CLASS, MIN_SIZE
img_generator = img_gen.patched_img(*args)

#%% Load DinoV2
# print(f"is cuda enabled: {torch.cuda.is_available()}")
# model = timm.create_model(
#     'vit_base_patch14_reg4_dinov2.lvd142m',
#     pretrained=True)
# device = torch.device('cuda')
# model = model.eval()
# print("Model loaded")
# model = model.to(device)

# data_config = timm.data.resolve_model_data_config(model)
# transforms = timm.data.create_transform(**data_config, is_training=False)

def data_loader(embeds, labels):
    print("Generating embeddings")
    labels = []
    for i in tqdm.tqdm(range(0, batch_size)):
        #get patch
        patch, label = img_generator.generate_patched_img()
        
        #process through dino
        # img = Image.fromarray(np.uint8(patch*255))
        # output = model.forward_features(transforms(img).unsqueeze(0).to(device))
        # output = model.forward_head(output, pre_logits=True)
        # img.close()
        
        #normalise embed with mean of mean per class
        # embed = torch.subtract(output, mean_embed.to(device))
        # embeds.append(embed)
        
        #one hot encode labels
        # one_hot = []
        # for j in classes:
        #     if j in label:
        #         one_hot.append(1)
        #     else:
        #         one_hot.append(0)
        # labels.append(one_hot)
        labels.append(label[0])
    return embeds,  labels

def onehot_to_class(onehot):
    obtained = []
    for i in range(len(classes)):
        if onehot[i] == 1:
           obtained.append(classes[i])
    return obtained
            
            
embed, labels = data_loader([], [])

# print(f"obtained class label: {labels}")
# print(f"obtained from one hot: {onehot_to_class(onehot)}")
print(len(set(labels)))