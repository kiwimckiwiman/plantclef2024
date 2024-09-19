import os
import timm
from PIL import Image
import torch
import tqdm as tqdm
#%%
TRAIN = r"C:\Users\ASUS\Desktop\plantclef\image_samples\samples"
EMBEDS = r"C:\Users\ASUS\Desktop\plantclef\embed_samples\embedv2"
CHKPNT = r"C:\Users\ASUS\Desktop\plantclef\models\vit\vit_base_patch14_reg4_dinov2_lvd142m_pc24_onlyclassifier_then_all\model_best.pth.tar"
#%%
print(f"Using GPU: {torch.cuda.is_available()}")
model = timm.create_model(
    'vit_base_patch14_reg4_dinov2.lvd142m',
    pretrained=False,
    num_classes=7806,
    checkpoint_path=CHKPNT
    )
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
    if (os.path.exists(os.path.join(EMBEDS, i)) == False):
        os.mkdir(os.path.join(EMBEDS, i))
    print("converting class " + i + " | " + str(curr) + " of " + str(len(os.listdir(TRAIN))))
    for j in tqdm.tqdm(os.listdir(os.path.join(TRAIN, i))):
        img = Image.open(os.path.join(os.path.join(TRAIN, i), j))
        output = model.forward_features(transforms(img).unsqueeze(0).to(device))
        output = model.forward_head(output, pre_logits=True)
        torch.save(output, os.path.join(os.path.join(EMBEDS, i), (j.split(".")[0] + ".pt")))
        img.close()
    if(curr == 1):
        break
    else:
        curr = curr +1