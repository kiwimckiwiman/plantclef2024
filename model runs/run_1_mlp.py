import patch_images_unblended as img_gen
import cv2
import os
import timm
from PIL import Image
import torch
import tqdm as tqdm
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

# Parameters:
TRAIN_PATH = r"D:\plantclef2024\PlantCLEF2024singleplanttrainingdata/train"
PATCH_HEIGHT = 500
PATCH_WIDTH = 500
MIN_CLASS = 1
MAX_CLASS = 4
MIN_SIZE = 150
#patched image generator
args = TRAIN_PATH, PATCH_HEIGHT, PATCH_WIDTH, MIN_CLASS, MAX_CLASS, MIN_SIZE
img_generator = img_gen.patched_img(*args)

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
#%%
#generate embeddings

def get_batch(batch_size, patch_embeds, patch_labels):
    print("Generating embeddings")
    for i in tqdm.tqdm(range(0, batch_size)):
        patch, label = img_generator.generate_patched_img()
        img = Image.fromarray(np.uint8(patch*255))
        output = model.forward_features(transforms(img).unsqueeze(0).to(device))
        output2 = model.forward_head(output, pre_logits=True)
        patch_embeds.append(output2)
        patch_labels.append(get_one_hot(label).to(device))
    return patch_embeds, patch_labels

def get_one_hot(classes):
    one_hot = []
    for i in os.listdir(TRAIN_PATH):
        if i in classes:
            one_hot.append(1)
        else:
            one_hot.append(0)
    return torch.Tensor(one_hot)
#%%
#build model
class mlp(nn.Module):
    def __init__(self):
        super(mlp, self).__init__()
        self.input = nn.Linear(768, 1024)
        self.hidden = nn.Linear(1024,1024)
        self.output = nn.Linear(1024, 7806)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.input(x))
        x = self.relu(self.hidden(x))
        x = self.output(x)
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

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.BCEWithLogitsLoss()
EPOCHS = 50
BATCH_SIZE = 16
VAL_BATCH_SIZE = 8
#%%
torch.autograd.set_detect_anomaly(True)
for n in range(EPOCHS):
    print(f"epoch: {n}")
    mlp_model.train()
    embeds = []
    true_labels = []
    embeds, true_labels = get_batch(BATCH_SIZE, embeds, true_labels)
    train_loss = 0.0
    #train_acc = 0.0
    for i in range(len(embeds)):
        optimizer.zero_grad()
        pred = mlp_model.forward(embeds[i].detach().cpu().to(device))
        loss = loss_fn(torch.reshape(true_labels[i], (1, 7806)).detach().cpu().to(device), pred)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        #train_acc = (torch.argmax(torch.reshape(true_labels[i], (1, 7806)), 1) == torch.argmax(pred, 1)).float().mean()
    print(f"avg train loss: {train_loss/len(embeds)}")    
    #print(f"avg train acc: {train_acc/len(embeds)}")
    
    # val_embeds, val_true_labels = get_batch(VAL_BATCH_SIZE)
    # mlp_model.eval()
    # val_loss = 0.0
    # val_acc = 0.0
    # with torch.no_grad():
    #     for i in range(len(val_embeds)):
    #         pred = model.forward(val_embeds[i].detach().cpu().to(device))
    #         loss = loss_fn(torch.reshape(val_true_labels[i], (1, 7806)).detach().cpu().to(device), pred)
    #         val_acc += (torch.reshape(val_true_labels[i], (1, 7806)).detach().cpu().to(device) == pred).float().sum()
    #         val_loss += loss
    
    # print(f"avg val loss: {val_loss/len(val_embeds)}")
    # print(f"avg val acc: {val_acc/len(val_embeds)}")