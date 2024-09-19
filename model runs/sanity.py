import cv2
import os
import timm
from PIL import Image
import torch
import tqdm as tqdm
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import random
TRAIN_PATH = r"D:\plantclef2024\merged_sample"
CLASSES = r"D:\plantclef2024\PlantCLEF2024singleplanttrainingdata/train"
#%%
print(f"is cuda enabled: {torch.cuda.is_available()}")
#load dino
model = timm.create_model(
    'vit_base_patch14_reg4_dinov2.lvd142m',
    pretrained=True)
device = torch.device('cuda')
model = model.eval()
print("Model loaded")
model = model.to(device)

data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)

classes = os.listdir(CLASSES)[:10]
def dataloader(embeds, labels):
    img_list = random.sample(os.listdir(TRAIN_PATH), 10)
    embeds = []
    labels = []
    tru = []
    for i in tqdm.tqdm(img_list):
        img = Image.open(os.path.join(TRAIN_PATH, i))
        output = model.forward_features(transforms(img).unsqueeze(0).to(device))
        output = model.forward_head(output, pre_logits=True)
        img.close()
        
        embeds.append(output)
        tru.append(i.split(".")[0])
        one_hot = []
        for j in classes:
            if j in i.split(".")[0].split("_"):
                one_hot.append(1)
            else:
                one_hot.append(0)
        labels.append(one_hot)
    return embeds, torch.Tensor(labels)  
#%%
#dataloader(embeds = [], labels = [])
#%%
#build model
class mlp(nn.Module):
    def __init__(self):
        super(mlp, self).__init__()
        self.input = nn.Linear(768, 1024)
        self.hidden = nn.Linear(1024,1024)
        self.output = nn.Linear(1024, 10)
        self.relu = nn.ReLU()
        self.soft = nn.Softmax()
        
    def forward(self, x):
        #x = F.normalize(x, p=2, dim=1, eps=1e-12)
        x = self.relu(self.input(x))
        x = self.relu(self.hidden(x))
        x = self.soft(self.output(x))
        return x

mlp_model = mlp()
print(mlp_model)
mlp_model.to(device)
#%%
def cce(target_label, pred_label):
    sum = 0
    for i in range(len(target_label)):
        sum += -1*pred_label[i].log()*target_label[i]
    return sum/len(target_label)

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
loss_fn = nn.CrossEntropyLoss()
EPOCHS = 200
BATCH_SIZE = 10
VAL_BATCH_SIZE = 8
#%%
torch.autograd.set_detect_anomaly(True)
embeds = []
true_labels = []
embeds, true_labels = dataloader(embeds, true_labels)
print(true_labels)
for n in range(EPOCHS):
    print(f"epoch: {n}")
    mlp_model.train()
    train_loss = 0.0
    #train_acc = 0.0
    for i in range(len(embeds)):
        optimizer.zero_grad()
        pred = mlp_model.forward(embeds[i].detach().cpu().to(device))
        true = torch.reshape(true_labels[i], (1, 10)).detach().cpu().to(device)
        loss = loss_fn(true, pred)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
    print(f"avg train loss: {train_loss/len(embeds)}")    
    