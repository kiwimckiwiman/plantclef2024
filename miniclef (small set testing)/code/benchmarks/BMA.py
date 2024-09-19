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
TEST = r"D:\plantclef2024\patch_4"
CHKPNT = r"C:\Users\User\plantclef\code\model\pretrained_models\vit_base_patch14_reg4_dinov2_lvd142m_pc24_onlyclassifier_then_all\model_best.pth.tar"
CSV = r"C:\Users\User\plantclef\code\benchmark\mlp_run_6_0.05_100_patch_4_class_whole.csv"

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

def softmax(scores):
    softmax_scores = []
    denom = 0
    for i in scores:
        denom += math.exp(i.item())
    for i in scores:
        softmax_scores.append(math.exp(i.item())/denom)
    return softmax_scores

def variance(data):
     n = len(data)
     mean = sum(data) / n
     deviations = [(x - mean) ** 2 for x in data]
     variance = sum(deviations) / n
     return variance

def get_pred(img):
    #feature extractor
    output = model.forward_features(transforms(img).unsqueeze(0).to(device))
    output = model.forward_head(output, pre_logits=True)
    img.close()
    
    #mlp classifier
    mlp_out = np.array(mlp.forward(output).cpu().detach()[0])
    
    #sort in descending
    pred = np.argsort(mlp_out)[::-1]
    score = mlp_out[pred]
    
    #get top 50 classes
    pred_labels = []
    for i in pred[:50]:
        pred_labels.append(CLASSES [i])
    
    #softmax scores
    all_softmax = softmax(score)
    
    #pair class to score
    predictions = []
    for i in range(50):
        predictions.append([pred_labels[i], all_softmax[i]])
    
    #calculate variance of set
    return variance(all_softmax), predictions

def get_confidence(var):
    # get the confidence score according to graph: −0.0119318x^2 + 0.00204545x + 1.01182
    # graph calculating by seeing the average variance across multiple patches
    # overall, 10e-8 to 10e-7 typically non plant majority
    # 10e-5 to 10e-4 plant majority
    # 10e-6 midpoint
    # assuming pairings power against confidence of:
        # 2 = 1
        # 4 = 0.9
        # 6 = 0.5
        # 8 = 0.3
    x = int(str(format(var, "e")).split("-")[1]) # janky method to get only the power e.g.(1.4123221749211187e-07 becomes 07)
    return (0.00204545)*(x) - (0.0119318)*(x**2) + 1.01182

all_preds = []
for folder in os.listdir(TEST):
    BMA_denom = 0 # running sum of BMA numerator
    all_preds = []
    BMA_numer = [] # individual BMA numerator for each patch
    classes = [] # classes seen in top 50 for class-wise BMA calculations
    folder_path = os.path.join(TEST, folder)
    N = len(os.listdir(folder_path)) # priori, in this case number of patches
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        img = Image.open(img_path)
        
        var, preds = get_pred(img) # get variance (how sparse probabilities are) and predicitions ([class, score])
        likelihood = get_confidence(var) # likelihood calculated by variance (how sure the model is the patch contains plants)
        
        numer = likelihood/N # numerator of BMA = likelihood x priori
        BMA_denom += numer
        
        all_preds.append([preds, numer])
        for i in preds: # add seen classes into list for class-wise BMA
            if i[0] not in classes:
                classes.append(i[0])

    BMA_scores = []
    
    for c in classes:
        BMA = 0 #initialise
        for pred in all_preds: #foreach patch 
            for p, n in pred: # foreach prediction and numerator
                if p[0] == c: #if prediction class matches current class iter
                    BMA += p[1] * (n/BMA_denom) #calculate BMA for pred for class c
                    
        BMA_scores.append([c, BMA])   
         
    all_preds.append([folder, BMA_scores])           
    

#%%
fields = ['img_name', 'top_50']
with open(CSV, 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(fields)
    csvwriter.writerows(all_preds)