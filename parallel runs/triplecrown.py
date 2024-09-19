# Imports
import torch
import timm
import os
import numpy as np
from PIL import Image
import csv
import torch.nn as nn
import math
import pandas as pd
import scipy.stats as stats
import sys
import warnings
#%%
# Paths
CLASS_LIST = r"D:\plantclef2024\embedsv2"
CLASSES = os.listdir(CLASS_LIST)

CLASSES_ = r"C:\Users\User\plantclef\code\model\pretrained_models\class_mapping.txt"
SPECIES = r"C:\Users\User\plantclef\code\model\pretrained_models\species_id_to_name.txt"

# Patches
PATCHES64 = r"D:\plantclef2024\patch_64"
PATCHES16 =r"D:\plantclef2024\patch_16"
# Models

CHKPNT = r"C:\Users\User\plantclef\code\model\pretrained_models\vit_base_patch14_reg4_dinov2_lvd142m_pc24_onlyclassifier_then_all\model_best.pth.tar"

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
def init_MLP(MLP_PATH):
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
    mlp.load_state_dict(torch.load(MLP_PATH))
    mlp = mlp.to(device)
    mlp.eval()
    print("MLP loaded: " + MLP_PATH)
    return mlp

#%%
#Processing
def vit_only(patch, model):
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)
    
    img = Image.open(patch)
    output = model(transforms(img).unsqueeze(0).to(device))  #unsqueeze single image into batch of 1
    n = 1000 #top-n to get variance from (hyper param)
    img.close()
    probs, indices = torch.topk(output.softmax(dim=1) * 100, k=n)
    probs = probs.cpu().detach().numpy()
    indices = indices.cpu().detach().numpy()
    preds = []
    for proba, cid in zip(probs[0], indices[0]):
        species_id = cid_to_spid[cid]
        preds.append([species_id, proba])
    return preds

def vit_extractor(patch, vit):
    data_config = timm.data.resolve_model_data_config(vit)
    transforms = timm.data.create_transform(**data_config, is_training=False)
    
    img = Image.open(patch)
    output = vit.forward_features(transforms(img).unsqueeze(0).to(device))
    output = vit.forward_head(output, pre_logits=True)
    img.close()
    return output

def mlp_pred(embed, mlp):
    mlp_out = np.array(mlp.forward(embed).cpu().detach()[0])
    
    n = 1000 #top-n to get variance from (hyper param)
    sorted_indexes = np.argsort(mlp_out)[::-1]
    sorted_array = mlp_out[sorted_indexes]
    softmax_arr = softmax(sorted_array)[:n]
    preds = []
    for i in range(n):
        preds.append([CLASSES[sorted_indexes[i]], softmax_arr[i]])
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
    a = a + 0.5
    return (math.sqrt((a-x)/a))

def variance(data):
     n = len(data)
     mean = sum(data) / n
     deviations = [(x - mean) ** 2 for x in data]
     var = sum(deviations) / n
     return var
#%%

def plsplsplsplspls(process):
    MLP_1_PATH = r"C:\Users\User\plantclef\code\model\models4\mlp_run_6_0.05_100_single_class.pth"
    MLP_5_PATH = r"C:\Users\User\plantclef\code\model\models4\mlp_run_8_0.07_300.pth"
    filtered_pred = []
    
    vit = init_vit_classifier()
    mlp1 = init_MLP(MLP_1_PATH)
    mlp5 = init_MLP(MLP_5_PATH)
    patch_list = os.listdir(PATCHES64)
    def split_into_batch(lst, chunk_size):
        return list(zip(*[iter(lst)] * chunk_size))
    count = 0
    patch_batch = split_into_batch(patch_list, int(1695/5))
    for image in patch_batch[int(process)]:
        count += 1
        folder64 = os.path.join(PATCHES64, image)
        folder16 = os.path.join(PATCHES16, image)
        all_preds = []
        all_log_var = []
        present_classes = []
        k = 100 #top-n predictions to consider for final output
        print("====================================")
        print(f"{process} | Now predicting {image} both 64 & 16 | {count} of {1695/5} | ViT | 64")
        for patch in os.listdir(folder64):
            img_path = os.path.join(folder64, patch)
            
            #ViT only (3)
            vit_predictions = vit_only(img_path, vit)
            vit_scores = []
            for pair in vit_predictions:
                if pair[0] not in present_classes:
                    present_classes.append(pair[0])
                vit_scores.append(pair[1])
            all_log_var.append(abs(math.log(variance(vit_scores), 10))) # append absolute log of variance
            all_preds.append(vit_predictions[:k])
            
            #Vit Feature extractor
            embed = vit_extractor(img_path, vit)
        
        print("====================================")
        print(f"{process} | Now predicting {image} both 64 & 16 | {count} of {1695/5} | MLP1 | 64")
        for patch in os.listdir(folder64):
            img_path = os.path.join(folder64, patch)
            
            #MLP 1 0.05, 100 epochs
            mlp_1_predictions = mlp_pred(embed, mlp1)
            mlp_1_scores = []
            for pair in mlp_1_predictions:
                if pair[0] not in present_classes:
                    present_classes.append(pair[0])
                mlp_1_scores.append(pair[1])
            all_log_var.append(abs(math.log(variance(mlp_1_scores), 10))) # append absolute log of variance
            all_preds.append(mlp_1_predictions[:k]) 
        
        print("====================================")
        print(f"{process} | Now predicting {image} both 64 & 16 | {count} of {1695/5} | MLP5 | 64")
        for patch in os.listdir(folder64):
            img_path = os.path.join(folder64, patch)
            
            #MLP 5 0.07, 300 epochs
            mlp_5_predictions = mlp_pred(embed, mlp5)
            mlp_5_scores = []
            for pair in mlp_5_predictions:
                if pair[0] not in present_classes:
                    present_classes.append(pair[0])
                mlp_5_scores.append(pair[1])
            all_log_var.append(abs(math.log(variance(mlp_5_scores), 10))) # append absolute log of variance
            all_preds.append(mlp_5_predictions[:k]) 
            
        print("====================================")
        print(f"{process} | Now predicting {image} both 64 & 16 | {count} of {1695/5} | ViT | 16")
        for patch in os.listdir(folder16):
            img_path = os.path.join(folder16, patch)
            
            #ViT only (3)
            vit_predictions = vit_only(img_path, vit)
            vit_scores = []
            for pair in vit_predictions:
                if pair[0] not in present_classes:
                    present_classes.append(pair[0])
                vit_scores.append(pair[1])
            all_log_var.append(abs(math.log(variance(vit_scores), 10))) # append absolute log of variance
            all_preds.append(vit_predictions[:k])
            
            #Vit Feature extractor
            embed = vit_extractor(img_path, vit)
        
        print("====================================")
        print(f"{process} | Now predicting {image} both 64 & 16 | {count} of {1695/5} | MLP1 | 16")
        for patch in os.listdir(folder16):
            img_path = os.path.join(folder16, patch)
            
            #MLP 1 0.05, 100 epochs
            mlp_1_predictions = mlp_pred(embed, mlp1)
            mlp_1_scores = []
            for pair in mlp_1_predictions:
                if pair[0] not in present_classes:
                    present_classes.append(pair[0])
                mlp_1_scores.append(pair[1])
            all_log_var.append(abs(math.log(variance(mlp_1_scores), 10))) # append absolute log of variance
            all_preds.append(mlp_1_predictions[:k]) 
        
        print("====================================")
        print(f"{process} | Now predicting {image} both 64 & 16 | {count} of {1695/5} | MLP5 | 16")
        for patch in os.listdir(folder16):
            img_path = os.path.join(folder16, patch)
            
            #MLP 5 0.07, 300 epochs
            mlp_5_predictions = mlp_pred(embed, mlp5)
            mlp_5_scores = []
            for pair in mlp_5_predictions:
                if pair[0] not in present_classes:
                    present_classes.append(pair[0])
                mlp_5_scores.append(pair[1])
            all_log_var.append(abs(math.log(variance(mlp_5_scores), 10))) # append absolute log of variance
            all_preds.append(mlp_5_predictions[:k]) 
            
        max_var = pd.Series(all_log_var).describe()[-1] # get max variance
        all_log_var = [confidence(i, max_var) for i in all_log_var] # calculate confidence with curve (likelihood)
        total_var = sum(all_log_var)
        BMA_weighted = []
        for p in range(len(all_preds)): #multiply probability by posterior (likelihood x priori/sum of all likelihood x priori)
            posterior_prob = all_log_var[p]/total_var
            BMA_pred = []
            for pred in all_preds[p]:
                BMA_pred.append([pred[0], pred[1]*posterior_prob])
            BMA_weighted.append(BMA_pred)
    
        BMA = [] #sum up scores based on class
        print("summing up scores...")
        for c in present_classes:
            BMA_s = 0
            for pred in BMA_weighted:
                for p in pred:
                    if(p[0] == c):
                        BMA_s += p[1]
            BMA.append([c, BMA_s])
            
        BMA_score = []
        BMA_label = []
        for i in BMA: #split label and score for sorting
            BMA_score.append(i[1])
            BMA_label.append(i[0])
        
        BMA_score = np.array(BMA_score)
        BMA_label = np.array(BMA_label)
        
        BMA_indices = np.argsort(BMA_score)[::-1] #sort scores
        sorted_BMA_score = BMA_score[BMA_indices] #keep indices
    
        sorted_BMA_label = [BMA_label[i] for i in BMA_indices] #get labels based on score sorted indices
    
        z_score = stats.zscore(sorted_BMA_score[:100]) #z_score computation for threshold
        
        fil_lab = []
        for j in range(len(z_score)):
            if(abs(z_score[j]) >= 2): #anything above a standard deviation of 2 (hyper parameter)
                fil_lab.append(int(sorted_BMA_label[j]))
        filtered_pred.append([image.split(".")[0], fil_lab])
        
        
    fields = ['plot_id', 'species_ids']
    file_name = "mlp_run_8_0.07_300_single_class_64_16_BMA_z_score" + str(process) + ".csv"
    OUTPUT = r"C:\Users\User\plantclef\code\benchmark"

    with open(os.path.join(OUTPUT, file_name), 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=';')
        csvwriter.writerow(fields)
        csvwriter.writerows(filtered_pred)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python sub_script.py <identifier>")
        sys.exit(1)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    
    identifier = sys.argv[1]
    plsplsplsplspls(identifier)