## Imports
import torch
import torchvision ## Contains some utilities for working with the image data
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
#%matplotlib inline
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import tqdm as tqdm
import random
import os
import timm
from PIL import Image
import numpy as np

TRAIN_PATH = r"D:\plantclef2024\merged_sample"
AVG = r"D:\plantclef2024\embeds\avg_all.pt"
CLASSES = r"D:\plantclef2024\PlantCLEF2024singleplanttrainingdata/train"

mean_embed = torch.load(AVG)
classes = os.listdir(CLASSES)[:10]

no_classes = len(classes)
bacth_size = 16

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


def accuracy(outputs, labels):
    pred = outputs.detach().cpu().numpy()[0]
    truth = labels.detach().cpu().numpy()[0]
    def calibrate_upper(x):
        return 1 if x > 0.99 else x
    def calibrate_lower(x):
        return 0 if x < 0.002 else x
    cal_up = np.vectorize(calibrate_upper)
    cal_down = np.vectorize(calibrate_lower)
    pred = cal_down(cal_up(pred))
    acc = 0
    for i in range(len(pred)):
        if pred[i] == truth[i]:
            acc += 1
    return(torch.tensor(acc/len(pred)))

def train_loader():
    print("Generating embeddings")
    embeds = []
    labels = []
    img_list = "D:\plantclef2024\merged_sample/1355868_1355869_1355870.png"
    i = "1355868_1355869_1355870.png"
    for k in tqdm.tqdm(range(10)):
        img = Image.open(img_list)
        output = model.forward_features(transforms(img).unsqueeze(0).to(device))
        output = model.forward_head(output, pre_logits=True)
        img.close()
        embed = torch.subtract(output, mean_embed.to(device))
        embeds.append(embed)
        one_hot = []
        for j in classes:
            if j in i.split(".")[0].split("_"):
                one_hot.append(1)
            else:
                one_hot.append(0)
        labels.append(one_hot)
    return embeds, torch.Tensor(labels) 
        

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(768, 1024)
        self.hidden = nn.Linear(1024,1024)
        self.output = nn.Linear(1024, no_classes)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
    
    def forward(self, xb):
        xb = xb.reshape(-1, 768)
        x = self.relu(self.input(xb))
        x = self.relu(self.hidden(x))
        x = self.output(x)
        return(x)
    
mlp = MLP()
mlp = mlp.to(device)

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()
embeds, labels = train_loader()
#val_embeds, val_labels = train_loader()
def fit(epochs, lr, model, opt_func = torch.optim.SGD):
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        train_loss = 0
        train_acc = 0
        val_loss = 0
        val_acc = 0
        ## Training Phase
        mlp.train()
        for i in range(len(embeds)):
            
            embed = embeds[i].detach().cpu().to(device)
            label = torch.reshape(labels[i], (1, no_classes)).detach().cpu().to(device)
            
            out = mlp.forward(embed) ## Generate predictions
            
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(out, label)
            acc = accuracy(out, label)
        
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.item()
            train_acc += acc.item()
            
        print(f"Epoch [{epoch}] | train_loss: {round((train_loss/len(embeds)),2)} train_acc: {round((train_acc/len(embeds)),2)}")
        
        mlp.eval()
        for i in range(len(embeds)):
            embed = embeds[i].detach().cpu().to(device)
            label = torch.reshape(labels[i], (1, no_classes)).detach().cpu().to(device)
            
            out = mlp.forward(embed)
            
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(out, label)
            acc = accuracy(out, label)
            val_loss += loss.item()
            val_acc += acc.item()
            
        print(f"Epoch [{epoch}] | val_loss: {round((val_loss/len(embeds)),2)} val_acc: {round((val_acc/len(embeds)),2)}")
        
        writer.add_scalar('Loss/train', train_loss/len(embeds), epoch)
        writer.add_scalar('Loss/val', val_loss/len(embeds), epoch)
        writer.add_scalar('Acc/train', train_acc/len(embeds), epoch)
        writer.add_scalar('Acc/val', val_acc/len(embeds), epoch)
        writer.close()
fit(200, 0.001, mlp)