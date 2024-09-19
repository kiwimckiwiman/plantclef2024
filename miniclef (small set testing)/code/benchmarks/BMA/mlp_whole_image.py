import torch
import timm
import os
import pandas as pd
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import csv
import torch.nn as nn
import scipy.stats as stats
import numpy as np
import math
#%%
TEST_IMGS =  r"D:\plantclef2024\PlantCLEF2024test"
MODEL = r"C:\Users\User\plantclef\code\model\pretrained_models\vit_base_patch14_reg4_dinov2_lvd142m_pc24_onlyclassifier_then_all\model_best.pth.tar"
CLASSES = r"C:\Users\User\plantclef\code\model\pretrained_models\class_mapping.txt"
SPECIES = r"C:\Users\User\plantclef\code\model\pretrained_models\species_id_to_name.txt"
RESULTS = r"C:\Users\User\plantclef\code\benchmark\2\mlp_whole_image_top_50_sig.csv"
FILTERED = r"C:\Users\User\plantclef\code\benchmark\2\filtered_bench_2.csv"
MLP_PATH = r"C:\Users\User\plantclef\code\model\models\mlp_run_6_0.05_100_single_class.pth "
print(len(os.listdir(TEST_IMGS)))
#%%
def load_class_mapping(class_list_file):
    with open(class_list_file) as f:
        class_index_to_class_name = {i: line.strip() for i, line in enumerate(f)}
    return class_index_to_class_name


def load_species_mapping(species_map_file):
    df = pd.read_csv(species_map_file, sep=';', quoting=1, dtype={'species_id': str})
    df = df.set_index('species_id')
    return  df['species'].to_dict()

cid_to_spid = load_class_mapping(CLASSES)
spid_to_sp = load_species_mapping(SPECIES)

print(len(cid_to_spid))
#%%
print(f"Using GPU: {torch.cuda.is_available()}")
model = timm.create_model(
    'vit_base_patch14_reg4_dinov2.lvd142m',
    pretrained=False,
    num_classes=len(cid_to_spid),
    checkpoint_path=MODEL
    )
device = torch.device('cuda')
model = model.to(device)
model = model.eval()
print("ViT loaded")
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)

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
print("MLP loaded")
#%%
def softmax(scores):
    softmax_scores = []
    denom = 0
    for i in scores:
        denom += math.exp(i.item())
    for i in scores:
        softmax_scores.append((math.exp(i.item())/denom)*100)
    return softmax_scores

#%%
top_n = 50
all_pred = []
count = 0
for img_name in tqdm(os.listdir(TEST_IMGS)):
    img = Image.open(os.path.join(TEST_IMGS, img_name))
    output = model.forward_features(transforms(img).unsqueeze(0).to(device))
    output = model.forward_head(output, pre_logits=True)
    
    pred = np.array(mlp.forward(output).cpu().detach()[0])
    
    sorted_indexes = np.argsort(pred)[::-1]
    sorted_array = pred[sorted_indexes]
    softmax_arr = softmax(sorted_array)
    predictions = []
    
    for i in range(50):
        predictions.append([cid_to_spid[sorted_indexes[i]], sorted_array[i]])
    all_pred.append([img_name, predictions])
#%%
fields = ['img_name', 'top_50']
with open(RESULTS, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(fields)
    csvwriter.writerows(all_pred)
    
#%%

# filtered = []
# for preds in all_pred:
#     score = []
#     label = []
#     for i in preds[1]:
#         label.append(i[0])
#         score.append(i[1])
        
#     z_score = stats.zscore(score)
#     fil_lab = []
#     fil_score = []
#     for j in range(len(z_score)):
#         if(abs(z_score[j]) >= 2):
#             fil_lab.append(int(label[j]))
#             fil_score.append(score[j])
#     filtered.append([preds[0].split(".")[0], fil_lab])

# #%%
# fields = ['plot_id', 'species_ids']
# with open(FILTERED, 'w', newline='') as csvfile:
#     csvwriter = csv.writer(csvfile, delimiter=';')
#     csvwriter.writerow(fields)
#     csvwriter.writerows(filtered)