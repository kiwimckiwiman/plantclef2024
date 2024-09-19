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
CSV = r"C:\Users\User\plantclef\code\benchmark\mlp_run_6_0.05_100_single_class_whole.csv"
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
    
    mlp_out = np.array(mlp.forward(output).cpu().detach()[0])
    
    pred = np.argsort(mlp_out)[::-1]
    score = mlp_out[pred]
    
    pred_labels = []
    denom = 0
    for i in score:
        denom += math.exp(i.item())
   
    for i in pred[:50]:
        pred_labels.append(CLASSES [i])
   
    all_softmax = []
    for i in score:
        all_softmax.append(softmax(i.item(), denom))
    
    return variance(all_softmax), all_softmax[:50], pred_labels

all_pred = []
for img_name in tqdm(os.listdir(TEST)):
    img_path = os.path.join(TEST, img_name)
    img = Image.open(img_path)    
    var, scores, labels = get_pred(img)
    pred = []
    for i in range(len(scores)):
        pred.append([labels[i], scores[i]])
        
    all_pred.append([img_name, pred, var])

#%%
fields = ['img_name', 'top_50', 'variance']
with open(CSV, 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(fields)
    csvwriter.writerows(all_pred)