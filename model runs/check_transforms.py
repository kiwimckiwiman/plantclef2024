import torch
import timm
from PIL import Image
import numpy as np
import patch_images_unblended as img_gen
#%%
CHKPNT = r"C:\Users\ASUS\Desktop\plantclef\models\vit\vit_base_patch14_reg4_dinov2_lvd142m_pc24_onlyclassifier_then_all\model_best.pth.tar"

print(f"Using GPU: {torch.cuda.is_available()}")
model = timm.create_model(
    'vit_base_patch14_reg4_dinov2.lvd142m',
    pretrained=False,
    num_classes=7806,
    checkpoint_path=CHKPNT
    )
device = torch.device('cuda')
model = model.eval()
print("Model loaded")
model = model.to(device)

data_config = timm.data.resolve_model_data_config(model)
print(data_config)
transforms = timm.data.create_transform(**data_config, is_training=False)
#%%
img_path = r"C:\Users\ASUS\Desktop\plantclef\image_samples\samples\1355934\0b37da0d0e6bbc23f8a552ec5f95b1895557c857.jpg"
img = Image.open(img_path)
pixels = list(img.getdata())
print(img.size)
trnsf = transforms(img).unsqueeze(0)
print(trnsf)
print(trnsf.shape)
print("========================================")
#%% Parameters
PATCH_HEIGHT = 600
PATCH_WIDTH = 600
MIN_CLASS = 2
MAX_CLASS = 3
MIN_SIZE = 150
batch_size = 8
TRAIN_PATH = r"C:\Users\ASUS\Desktop\plantclef\image_samples\samples"

args = TRAIN_PATH, PATCH_HEIGHT, PATCH_WIDTH, MIN_CLASS, MAX_CLASS, MIN_SIZE
img_generator = img_gen.patched_img(*args)

patch, label = img_generator.generate_patched_img()

print(patch)
img2 = Image.fromarray(np.uint8(patch))
print(img2.size)
trnsf2 = transforms(img2).unsqueeze(0)
print(trnsf2)
print(trnsf2.shape)