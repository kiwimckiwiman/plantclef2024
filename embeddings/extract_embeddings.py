import os
import timm
from PIL import Image
import torch
import tqdm as tqdm
#%%
TRAIN = r"D:\plantclef2024\PlantCLEF2024singleplanttrainingdata/train"
EMBEDS = r"D:\plantclef2024\embeds"
#%%
model = timm.create_model(
    'vit_base_patch14_reg4_dinov2.lvd142m',
    pretrained=True)
device = torch.device('cuda')
model = model.to(device)
model = model.eval()
print("Model loaded")
#%%

data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)

#%%
curr= 1
for i in os.listdir(TRAIN):
    #if (os.path.exists(os.path.join(EMBEDS, i)) == False):
        #os.mkdir(os.path.join(EMBEDS, i))
    print("converting class " + i + " | " + str(curr) + " of " + str(len(os.listdir(TRAIN))))
    for j in tqdm.tqdm(os.listdir(os.path.join(TRAIN, i))):
        img = Image.open(os.path.join(os.path.join(TRAIN, i), j))
        output = model.forward_features(transforms(img).unsqueeze(0).to(device))
        output = model.forward_head(output, pre_logits=True)
        print(output)
        #torch.save(output, os.path.join(os.path.join(EMBEDS, i), (j.split(".")[0] + ".pt")))
        img.close()
    if(curr == 1):
        break
    else:
        curr = curr +1