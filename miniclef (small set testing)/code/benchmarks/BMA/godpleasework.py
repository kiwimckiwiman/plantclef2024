# Imports
import torch
import timm
import os
import numpy as np
from PIL import Image
import csv
import torch.nn as nn
import math
from tqdm import tqdm
import pandas as pd
import scipy.stats as stats

#%%
# Paths
CLASS_LIST = r"D:\plantclef2024\embedsv2"
CLASSES = os.listdir(CLASS_LIST)

CLASSES_ = r"C:\Users\User\plantclef\code\model\pretrained_models\class_mapping.txt"
SPECIES = r"C:\Users\User\plantclef\code\model\pretrained_models\species_id_to_name.txt"

# Patches
PATCHES = r"D:\plantclef2024\patch_64"

# Model
MLP = r"C:\Users\User\plantclef\code\model\models\mlp_run_6_0.05_100_single_class.pth "
CHKPNT = r"C:\Users\User\plantclef\code\model\pretrained_models\vit_base_patch14_reg4_dinov2_lvd142m_pc24_onlyclassifier_then_all\model_best.pth.tar"
P_NP = r"C:\Users\User\plantclef\please\models\mobilenet_v2_epoch_100.pth"

OUTPUT = r"C:\Users\User\plantclef\code\benchmark\64"

#%%
# Functions for classifying with ViT
def load_class_mapping(class_list_file):
    with open(class_list_file) as f:
        class_index_to_class_name = {i: line.strip() for i, line in enumerate(f)}
    return class_index_to_class_name


def load_species_mapping(species_map_file):
    df = pd.read_csv(species_map_file, sep=';', quoting=1, dtype={'species_id': str})
    df = df.set_index('species_id')
    return  df['species'].to_dict()

cid_to_spid = load_class_mapping(CLASSES_)
spid_to_sp = load_species_mapping(SPECIES)

print(len(cid_to_spid))
#%%

#Models
device = torch.device('cuda')
print(f"is cuda enabled: {torch.cuda.is_available()}")
#ViT feature extractor
def init_vit_classifier():
    model = timm.create_model(
        'vit_base_patch14_reg4_dinov2.lvd142m',
        pretrained=False,
        num_classes=len(cid_to_spid),
        checkpoint_path=CHKPNT
    )
    model = model.eval()
    model = model.to(device)
    print("ViT Model loaded")
    return model
    
#MLP
def init_MLP():
    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.input = nn.Linear(768, 1024)
            self.output = nn.Linear(1024, len(cid_to_spid))
            self.relu = nn.ReLU()
            self.bn = nn.BatchNorm1d(1024)
        def forward(self, xb):
            xb = xb.reshape(-1, 768)
            x = self.relu(self.bn(self.input(xb)))
            x = self.output(x)
            return(x)
    mlp = MLP()
    mlp.load_state_dict(torch.load(MLP))
    mlp = mlp.to(device)
    mlp.eval()
    print("MLP loaded")
    return mlp

#%%

#Processing
def vit_only(patch):
    model = init_vit_classifier()
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)
    
    img = Image.open(patch)
    output = model(transforms(img).unsqueeze(0))  #unsqueeze single image into batch of 1
    n = len(output) #top-n to get variance from (hyper param)
    probabilities, indices = torch.topk(output.softmax(dim=1) * 100, k=n)
    img.close()
    
    preds = []
    for i in range(n):
        preds.append([cid_to_spid[indices[i]], probabilities[i]])
    return preds

def vit_to_mlp(patch):
    vit = init_vit_classifier()
    mlp = init_MLP()
    
    data_config = timm.data.resolve_model_data_config(vit)
    transforms = timm.data.create_transform(**data_config, is_training=False)
    
    img = Image.open(patch)
    output = vit.forward_features(transforms(img).unsqueeze(0).to(device))
    output = vit.forward_head(output, pre_logits=True)
    img.close()
    
    mlp_out = np.array(mlp.forward(output).cpu().detach()[0])
    
    n = len(mlp_out) #top-n to get variance from (hyper param)
    sorted_indexes = np.argsort(mlp_out)[::-1]
    sorted_array = mlp_out[sorted_indexes]
    softmax_arr = softmax(sorted_array)[:n]
    preds = []
    for i in range(n):
        preds.append([cid_to_spid[sorted_indexes[i]]], softmax_arr[i])
    return preds

#%%

#Maths
def softmax(scores):
    softmax_scores = []
    denom = 0
    for i in scores:
        denom += math.exp(i.item())
    for i in scores:
        softmax_scores.append((math.exp(i.item())/denom)*100)
    return softmax_scores

def confidence(x, a):
    a = a + 1
    return (math.sqrt((a-x)/a))

def variance(data):
     n = len(data)
     mean = sum(data) / n
     deviations = [(x - mean) ** 2 for x in data]
     var = sum(deviations) / n
     return var
#%%

filtered_pred = []
for image in os.listdir(PATCHES):
    print(f"Now predicting {image}")
    folder = os.path.join(PATCHES, image)
    all_preds = []
    all_log_var = []
    present_classes = []
    k = 100 #top-n predictions to consider for final output
    for patch in tqdm(os.listdir(folder)):
        img_path = os.path.join(folder, patch)
        predictions = vit_to_mlp(img_path)
        
        scores = []
        for pair in predictions:
            if pair[0] not in present_classes:
                present_classes.append(pair[0])
                scores.append(pair[1])
                
        all_log_var.append(abs(math.log(variance(scores), 10))) # append absolute log of variance
        all_preds.append(predictions[:k])
    
    max_var = pd.Series(all_log_var).describe()[-1] # get max variance
    all_log_var = [confidence(i, max_var) for i in all_log_var] # calculate confidence with curve (likelihood)
    total_var = sum(all_log_var)
    
    BMA_weighted = []
    for p in len(all_preds): #multiply probability by posterior (likelihood x priori/sum of all likelihood x priori)
        b = [(i[1] * all_log_var[p]/total_var) for i in all_preds[p]]
        BMA_weighted.append(b)
    
    BMA = [] #sum up scores based on class
    for c in present_classes:
        BMA_s = 0
        for pred in BMA_weighted:
            if(pred[0] == c):
                BMA_s += pred[1]
                
        BMA.append([c, BMA_s])
    
    BMA_score = []
    BMA_label = []
    
    for i in BMA: #split label and score for sorting
        BMA_score.append[i[1]]
        BMA_label.append[i[0]]
        
    sorted_BMA_score = np.argsort(BMA_score)[::-1] #sort scores
    BMA_indices = BMA_score[sorted_BMA_score] #keep indices
    
    sorted_BMA_label = [i for i in BMA_label[BMA_indices[i]]] #get labels based on score sorted indices
    
    z_score = stats.zscore(sorted_BMA_score) #z_score computation for threshold
    fil_lab = []
    for j in range(len(z_score)):
        if(abs(z_score[j]) >= 2): #anything above a standard deviation of 2 (hyper parameter)
            fil_lab.append(int(sorted_BMA_label[j]))
    filtered_pred.append([image.split(".")[0], fil_lab])
    
fields = ['plot_id', 'species_ids']
with open(OUTPUT, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter=';')
    csvwriter.writerow(fields)
    csvwriter.writerows(filtered_pred)