import torch
import timm
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import csv
import torch.nn as nn
import math
#%%
EMBEDS = r"D:\plantclef2024\embedsv2"
CLASSES = os.listdir(EMBEDS)
MODEL = r"C:\Users\User\plantclef\code\model\models\mlp_run_6_0.05_100_single_class.pth "
TEST = r"D:\plantclef2024\PlantCLEF2024test"
CHKPNT = r"C:\Users\User\plantclef\code\model\pretrained_models\vit_base_patch14_reg4_dinov2_lvd142m_pc24_onlyclassifier_then_all\model_best.pth.tar"
#%%
print(len(CLASSES))

#%%
#%% Load DinoV2
print(f"Using GPU: {torch.cuda.is_available()}")
model = timm.create_model(
    'vit_base_patch14_reg4_dinov2.lvd142m',
    pretrained=False,
    num_classes=len(CLASSES),
    checkpoint_path=CHKPNT
    )
device = torch.device('cuda')
model = model.eval()
print("VIT Model loaded")
model = model.to(device)

data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)
#%%
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(768, 1024)
        self.output = nn.Linear(1024, len(CLASSES))
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(1024)
    def forward(self, xb):
        xb = xb.reshape(-1, 768)
        x = self.relu(self.bn(self.input(xb)))
        x = self.output(x)
        return(x)
    
mlp = MLP()
mlp.load_state_dict(torch.load(MODEL))
mlp = mlp.to(device)
mlp.eval()
print("Model loaded")
#%%

def softmax(x, denom):
    return (math.exp(x))/denom

def variance(data):
     n = len(data)
     mean = sum(data) / n
     deviations = [(x - mean) ** 2 for x in data]
     variance = sum(deviations) / n
     return variance

def get_pred(img):
    output = model.forward_features(transforms(img).unsqueeze(0).to(device))
    output = model.forward_head(output, pre_logits=True)
    img.close()
    
    mlp_out = mlp.forward(output)
    pred = sorted(range(len(mlp_out[0])), key=lambda k: mlp_out[0][k], reverse=True)
    score = sorted(mlp_out[0], reverse = True)
    pred_labels = []
    
    denom = 0
    for i in score:
        denom += math.exp(i.item())
        
    for i in pred[:10]:
        pred_labels.append(CLASSES [i])
    
    all_softmax = []
    for i in score:
        all_softmax.append(softmax(i.item(), denom))
    
    return variance(all_softmax), all_softmax[:10], pred_labels

non_plants = [19, 0, 1, 2, 3, 4, 9, 10, 11, 12, 63]
plants = [13, 14, 21, 22, 26, 28, 29, 30, 34, 45, 25, 27, 40, 39, 62, 61]

plant_var = []
non_plant_var = []
for i in plants:
    img_sel = "CBN-can-B5-20230705.jpg_" + str(i) + ".jpg"
    path = r"D:\plantclef2024\patch_64\CBN-can-B5-20230705.jpg"
    
    img_path = os.path.join(path, img_sel)
    img = Image.open(img_path)
    plt.figure()
    plt.imshow(img)
    plt.show()
    
    var, scores, labels = get_pred(img)
    plant_var.append([var, i])
    print(f"variance: {var}")    
    print(f"top_10 scores: {scores}")
    print(f"top_10 labels: {labels}")
    print("==============")

for i in non_plants:
    img_sel = "CBN-can-B5-20230705.jpg_" + str(i) + ".jpg"
    path = r"D:\plantclef2024\patch_64\CBN-can-B5-20230705.jpg"
    
    img_path = os.path.join(path, img_sel)
    img = Image.open(img_path)
    plt.figure()
    plt.imshow(img)
    plt.show()
    
    
    var, scores, labels = get_pred(img)
    non_plant_var.append(var)
    print(f"variance: {var}")    
    print(f"top_10 scores: {scores}")
    print(f"top_10 labels: {labels}")
    print("==============")

print(plant_var)
print(non_plant_var)

# [ plant variance
# [1.1398931489424097e-05, 13],
# [0.00010511468785377513, 14],
# [8.095931403278749e-06, 21],
# [3.9726602210666254e-05, 22],
# [7.368980541873261e-05, 26],
# [9.107026892513358e-05, 28],
# [4.5021705164991185e-06, 29],
# [1.7753349305876713e-06, 30],
# [1.2103549084599627e-05, 34],
# [9.954258236238647e-05, 45],
# [9.375361901541649e-06, 25],
# [5.448103331017183e-05, 27],
# [8.046465456734092e-08, 40],
# [1.7755950202448868e-07, 39],
# [1.0030564731710325e-07, 62],
# [2.5492292233496797e-05, 61]
# ]
# [ non plant variance
# [1.5409947260255264e-07, 19],
# [1.1919513803410647e-07, 0],
# [8.110041873044597e-08, 1],
# [4.2253592370264007e-08, 2],
# [4.8483349299844204e-08, 3],
# [6.421164056787669e-08, 4],
# [5.8285121345953936e-08, 9],
# [1.0500542628120073e-07, 10],
# [9.580529774507265e-08, 11],
# [1.2405653251320443e-07, 12],
# 8.325579548273902e-08, 63]
# ]

# overall, e-8 to e-7 typically non plant majority
# e-5 to e-4 plant majority
# e-6 midpoint
# assuming pairings power against confidence of:
    # 2 = 1
    # 4 = 0.9
    # 6 = 0.5
    # 8 = 0.3
    
    # equation (y=-0.0094x^2 - 0.0287x + 1.1125)