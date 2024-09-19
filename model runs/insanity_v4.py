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
    def cal_up(x):
        return 1 if x > 0.95 else x
    def cal_down(x):
        return 0 if x < 0.05 else x
    ones = 0
    correct = 0
    for i in range(len(outputs)):
        cal_up_lambda = np.vectorize(cal_up)
        cal_down_lambda = np.vectorize(cal_down)
        corrected = cal_up_lambda(cal_down_lambda(outputs[i]))
        for j in range(len(outputs[0])):
            if labels[i][j] == 1:
                ones += 1
                if corrected[j] == 1:
                    correct += 1
    return(torch.tensor(correct/ones))

def data_loader(embeds, labels):
    print("Generating embeddings")
    img_list = "D:\plantclef2024\merged_sample/1355868_1355869_1355870.png"
    i = "1355868_1355869_1355870.png"
    for k in tqdm.tqdm(range(10)):
        img = Image.open(img_list)
        output = model.forward_features(transforms(img).unsqueeze(0).to(device))
        output = model.forward_head(output, pre_logits=True)
        img.close()
        embed = torch.subtract(output, mean_embed.to(device))
        embeds.append(torch.tensor(embed))
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
        self.output = nn.Linear(1024, no_classes)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(1024)
    
    def forward(self, xb):
        xb = xb.reshape(-1, 768)
        x = self.relu(self.input(xb))
        x = self.output(x)
        return(x)
    
mlp = MLP()
mlp = mlp.to(device)
from torch.utils.tensorboard import SummaryWriter
import time
embeds, labels = data_loader(embeds = [], labels = [])
writer = SummaryWriter()
def fit(epochs, lr, model, opt_func = torch.optim.SGD):
    optimizer = opt_func(model.parameters(), lr)
    iter_step = 8
    for epoch in range(epochs):
        time.sleep(0.1)
        # init
        train_loss_batch = 0
        train_acc_batch = 0
        val_loss_batch = 0
        val_acc_batch = 0
        
        # Training Phase
        mlp.train()
        out = mlp.forward(torch.stack(embeds, dim=0).to(device))
        
        # reshape for acc calculations
        outputs = []
        for i in out.clone().detach().squeeze(1):
            outputs.append(torch.FloatTensor(i.cpu().numpy()))
        outputs = torch.stack(outputs, dim=0)
        
        # loss calc
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(out.squeeze(1), labels.to(device))/iter_step
        loss.backward() # backprop
    
        acc = accuracy(outputs, labels) # acc calc
        
        # record vals
        train_loss_batch += loss.item()
        train_acc_batch += acc.item()
        
        # iter batch
        if((epoch+1)%iter_step == 0) or (epoch+1 == epochs):
            optimizer.step()
            optimizer.zero_grad()
        
        # Validation Phase
        mlp.eval()
        val_out = mlp.forward(torch.stack(embeds, dim=0).to(device))
        
        # reshape for acc calculations
        val_outputs = []
        for i in val_out.clone().detach().squeeze(1):
            val_outputs.append(torch.FloatTensor(i.cpu().numpy()))
        val_outputs = torch.stack(val_outputs, dim=0)
        
        val_loss = loss_fn(val_out.squeeze(1), labels.to(device)) # loss calc
        
        val_acc = accuracy(outputs, labels)# acc calc
        
        # record vals
        val_loss_batch += val_loss.item()
        val_acc_batch += val_acc.item()
        
        # display/record
        print(f"Epoch [{epoch}] of [{epochs}]| train_loss: {round(train_loss_batch,2)} train_acc: {round(train_acc_batch,2)}")
        print(f"Epoch [{epoch}] of [{epochs}]| val_loss: {round(val_loss_batch,2)} val_acc: {round(val_acc_batch,2)}")
        print("===============================================================================")
        writer.add_scalar('Loss/train', train_loss_batch, epoch)
        writer.add_scalar('Loss/val', val_loss_batch, epoch)
        writer.add_scalar('Acc/train', train_acc_batch, epoch)
        writer.add_scalar('Acc/val', val_acc_batch, epoch)
        writer.close()
fit(200, 0.001, mlp)