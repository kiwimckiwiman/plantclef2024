import os
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import math
import scipy.stats as stats

MODEL_PATH = r"C:\Users\User\plantclef\code\model\model3"
EMBEDS = r"D:\plantclef2024\embedsv2"
CLASSES = os.listdir(EMBEDS)
MODELS = os.listdir(MODEL_PATH)
VAL = r"D:\plantclef2024\merged_embeds_all\val"


device = torch.device('cuda')
print(f"using GPU: {torch.cuda.is_available()}")
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

def accuracy(outputs, labels):
    def cal_up(x):
        return 1 if x > 0.95 else x
    def cal_down(x):
        return 0 if x < 0.05 else x
    ones = 0
    correct = 0
    cal_up_lambda = np.vectorize(cal_up)
    cal_down_lambda = np.vectorize(cal_down)
    corrected = cal_up_lambda(cal_down_lambda(outputs))
    for i in range(len(outputs)):
        if labels[i] == 1:
            ones += 1
            if corrected[i] == 1:
                correct += 1
            
    return (torch.tensor(correct/ones)), corrected 

def one_hot_to_label(x):
    labels = []
    for i in range(len(x)):
        if x[i] == 1:
            labels.append(CLASSES[i])
    return labels

def softmax(scores):
    softmax_scores = []
    denom = 0
    for i in scores:
        denom += math.exp(i)
    for i in scores:
        softmax_scores.append((math.exp(i)/denom)*100)
    return softmax_scores

def load_class_mapping(class_list_file):
    with open(class_list_file) as f:
        class_index_to_class_name = {i: line.strip() for i, line in enumerate(f)}
    return class_index_to_class_name

CLASSES_LIST = r"C:\Users\User\plantclef\code\model\pretrained_models\class_mapping.txt"
cid_to_spid = load_class_mapping(CLASSES_LIST)

for m in MODELS:
    print(f"Now validating: {m}")
    mlp = MLP()
    mlp.load_state_dict(torch.load(os.path.join(MODEL_PATH, m)))
    mlp = mlp.to(device)
    mlp.eval()
    print(f"{m} | Model loaded")
    
    total = 2500
    top_n = 0
    top_10 = 0
    top_test = 0
    for file in tqdm(os.listdir(VAL)[:2500]):
        embed = torch.tensor(np.load(os.path.join(VAL, file))[0])
        out = mlp.forward(embed.to(device))
        
        outputs = []
        for i in out.clone().detach().squeeze(1):
            outputs.append(torch.FloatTensor(i.cpu().numpy()))
        outputs = torch.stack(outputs, dim=0)
        
        one_hot = []
        for class_name in CLASSES:
            item_labels = file.split(".")[0].split("_")[:-1]
            if class_name in item_labels:
                one_hot.append(1)
            else:
                one_hot.append(0)
                
        true_labels = file.split(".")[0].split("_")[:-1]
        pred = sorted(range(len(outputs[0])), key=lambda k: outputs[0][k], reverse=True)
        scores = sorted(outputs[0], reverse=True)
        pred_labels = []
        
        for i in pred[:10]:
            pred_labels.append(CLASSES [i])
        
        
        corr = 0
        for i in true_labels:
            if i in pred_labels[:len(true_labels)]:
                corr+= 1
        top_n += (corr/len(true_labels))
        
        corr_10 = 0
        for i in true_labels:
            if i in pred_labels[:10]:
                corr_10+= 1
        top_10 += (corr_10/len(true_labels))
        
        #calculate how test would be
        
        softmax_arr = softmax(scores)
        
        z_score = stats.zscore(softmax_arr)
        fil_lab = []
        for j in range(len(z_score)):
            if(abs(z_score[j]) >= 2):
                fil_lab.append([cid_to_spid[j]])
        
        
        correct_test = 0
        for l in true_labels:
            if l in fil_lab:
                correct_test += 1
       
        top_test += correct_test/len(true_labels)
        
    print(f"{m} top_n accuracy: {top_n/total}")
    print(f"{m} top_10 accuracy: {top_10/total}")
    print(f"{m} top_test accuracy: {top_test/total}")
    print("======================================")
    

        