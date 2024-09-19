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
classes = os.listdir(CLASSES)[:5]

no_classes = len(classes)
bacth_size = 8

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
    def calibrate(x):
        return 1 if x >0.7 else 0
    apply = np.vectorize(calibrate)
    pred_max = apply(pred)
    acc = 0
    for i in range(len(pred_max)):
        if pred_max[i] == truth[i]:
            acc += 1
    return(torch.tensor(acc/len(pred)))

def train_loader():
    print("Generating embeddings")
    img_list = random.sample(os.listdir(TRAIN_PATH), bacth_size)
    embeds = []
    labels = []
    tru = []
    for i in tqdm.tqdm(img_list):
        img = Image.open(os.path.join(TRAIN_PATH, i))
        output = model.forward_features(transforms(img).unsqueeze(0).to(device))
        output = model.forward_head(output, pre_logits=True)
        img.close()
        embed = torch.subtract(output, mean_embed.to(device))
        embeds.append(embed)
        tru.append(i.split(".")[0])
        one_hot = []
        for j in classes:
            if j in i.split(".")[0].split("_"):
                one_hot.append(1)
            else:
                one_hot.append(0)
        labels.append(one_hot)
    return embeds, torch.Tensor(labels) 
        

class MnistModel(nn.Module):
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
    
    def training_step(self, image, labels):
        out = self(image) ## Generate predictions
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(out, labels)
        acc = accuracy(out, labels)
        return(loss, {'train_loss':loss, 'train_acc': acc})
    
    def validation_step(self, val_embeds, val_labels):
        out = self(val_embeds)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(out, val_labels)
        acc = accuracy(out, val_labels)
        return({'val_loss':loss, 'val_acc': acc})
    
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return({'val_loss': epoch_loss.item(), 'val_acc' : epoch_acc.item()})
    
    def train_epoch_end(self, outputs):
        batch_losses = [x['train_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['train_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return({'train_loss': epoch_loss.item(), 'train_acc' : epoch_acc.item()})
    
    def val_epoch_out(self, epoch,result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))
    
    def train_epoch_out(self, epoch,result):
        print("Epoch [{}], train_loss: {:.4f}, train_acc: {:.4f}".format(epoch, result['train_loss'], result['train_acc']))    
    
mlp = MnistModel()
mlp = mlp.to(device)
def evaluate(model, val_embeds, val_labels):
    outputs = [model.validation_step(val_embeds[i].detach().cpu().to(device), torch.reshape(val_labels[i], (1, no_classes)).detach().cpu().to(device)) for i in range(len(val_embeds))]
    return(model.validation_epoch_end(outputs))

embeds, labels = train_loader()
def fit(epochs, lr, model, opt_func = torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        train_res = []
        ## Training Phase
        for i in range(len(embeds)):
            loss, res = mlp.training_step(embeds[i].detach().cpu().to(device), torch.reshape(labels[i], (1, no_classes)).detach().cpu().to(device))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_res.append(res)
        train_end = mlp.train_epoch_end(train_res)
        ## Validation phase
        result = evaluate(mlp, embeds, labels)
        history.append(result)
        
        mlp.train_epoch_out(epoch, train_end)
        mlp.val_epoch_out(epoch, result)
    return(history)

history1 = fit(200, 0.001, mlp)