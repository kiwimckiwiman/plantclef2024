#%%Imports
import torch
import patch_images_unblended as img_gen
import torch.nn as nn
import tqdm as tqdm
import os
import timm
from PIL import Image
import numpy as np
from torch.utils.tensorboard import SummaryWriter
#%% Paths
TRAIN_PATH = r"D:\plantclef2024\PlantCLEF2024singleplanttrainingdata/train"
AVG = r"D:\plantclef2024\embeds\avg_all.pt"
CLASSES = r"D:\plantclef2024\PlantCLEF2024singleplanttrainingdata/train"

#%% Parameters
PATCH_HEIGHT = 500
PATCH_WIDTH = 500
MIN_CLASS = 1
MAX_CLASS = 1
MIN_SIZE = 150
batch_size = 8
#%% 

mean_embed = torch.load(AVG)
classes = os.listdir(CLASSES)

no_classes = len(classes)
#%% Patch generator
args = TRAIN_PATH, PATCH_HEIGHT, PATCH_WIDTH, MIN_CLASS, MAX_CLASS, MIN_SIZE
img_generator = img_gen.patched_img(*args)

#%% Load DinoV2
print(f"is cuda enabled: {torch.cuda.is_available()}")
model = timm.create_model(
    'vit_base_patch14_reg4_dinov2.lvd142m',
    pretrained=True)
device = torch.device('cuda')
model = model.eval()
print("Model loaded")
model = model.to(device)

data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)

#%% Accuracy
def accuracy(outputs, labels):
    pred = outputs.detach().cpu().numpy()[0]
    truth = labels.detach().cpu().numpy()[0]
    def calibrate_upper(x):
        return 1 if x > 0.9 else x
    def calibrate_lower(x):
        return 0 if x < 0.002 else x
    cal_up = np.vectorize(calibrate_upper)
    cal_down = np.vectorize(calibrate_lower)
    pred = cal_down(cal_up(pred))
    acc = 0
    truth_labels = 0
    avg_pred = 0
    for i in range(len(pred)):
        if truth[i] == 1:
            truth_labels += 1
            if pred[i] == truth[i]:
                acc += 1
                avg_pred += pred[i]
    return(torch.tensor(acc/truth_labels), (avg_pred/truth_labels))

#%% Data Loader
def data_loader(embeds, labels):
    print("Generating embeddings")
    for i in tqdm.tqdm(range(0, batch_size)):
        #get patch
        patch, label = img_generator.generate_patched_img()
        
        #process through dino
        img = Image.fromarray(np.uint8(patch*255))
        output = model.forward_features(transforms(img).unsqueeze(0).to(device))
        output = model.forward_head(output, pre_logits=True)
        img.close()
        
        #normalise embed with mean of mean per class
        embed = torch.subtract(output, mean_embed.to(device))
        embeds.append(embed)
        
        #one hot encode labels
        one_hot = []
        for j in classes:
            if j in label:
                one_hot.append(1)
            else:
                one_hot.append(0)
        labels.append(one_hot)
    return embeds, torch.Tensor(labels), one_hot, label 

def onehot_to_class(onehot):
    obtained = []
    for i in range(len(classes)):
        if onehot[i] == 1:
           obtained.append(classes[i])
    return obtained
            

#%% Initialise MLP
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
writer = SummaryWriter()
# Training loop
def fit(epochs, lr, model, opt_func = torch.optim.SGD):
    optimizer = opt_func(model.parameters(), lr)
    iter_step = 8
    for epoch in range(epochs):
        # initialise 
        embeds = []
        labels= []
        
        train_loss = 0
        train_acc = 0
        val_loss = 0
        val_acc = 0
        avg_train_pred_val = 0
        avg_test_pred_val = 0
        # Training Phase
        mlp.train()
        embeds, labels, onehot, label = data_loader(embeds, labels)
        for i in range(len(embeds)):
            
            embed = embeds[i].detach().cpu().to(device)
            label = torch.reshape(labels[i], (1, no_classes)).detach().cpu().to(device)
            
            out = mlp.forward(embed)
            
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(out, label)
            acc, val = accuracy(out, label)
            
            loss2 = loss/iter_step
            loss2.backward()
            
            if((epoch+1)%iter_step == 0) or (epoch+1 == epochs):
                optimizer.step()
                optimizer.zero_grad()
            train_loss += loss.item()
            train_acc += acc.item()
            avg_train_pred_val += val
        mlp.eval()
        embeds, labels, onehot, label = data_loader(embeds = [], labels = [])
        for i in range(len(embeds)):
            embed = embeds[i].detach().cpu().to(device)
            label = torch.reshape(labels[i], (1, no_classes)).detach().cpu().to(device)
            
            out = mlp.forward(embed)
            
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(out, label)
            acc, val = accuracy(out, label)
            val_loss += loss.item()
            val_acc += acc.item()
            avg_test_pred_val += val
        print(f"Epoch [{epoch}] of [{epochs}]| train_loss: {round((train_loss/len(embeds)),2)} train_acc: {round((train_acc/len(embeds)),2)} avg_pred_score: {round((avg_train_pred_val/len(embeds)),2)}")
        print(f"Epoch [{epoch}] of [{epochs}]| val_loss: {round((val_loss/len(embeds)),2)} val_acc: {round((val_acc/len(embeds)),2)} avg_pred_score: {round((avg_test_pred_val/len(embeds)),2)}")
        print("===============================================================================")
        writer.add_scalar('Loss/train', train_loss/len(embeds), epoch)
        writer.add_scalar('Loss/val', val_loss/len(embeds), epoch)
        writer.add_scalar('Acc/train', train_acc/len(embeds), epoch)
        writer.add_scalar('Acc/val', val_acc/len(embeds), epoch)
        writer.close()
fit(30000, 0.005, mlp)