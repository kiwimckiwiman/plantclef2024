#%%Imports
import patch_images_unblended as img_gen
import cv2
import os
from PIL import Image
import time
from tqdm import tqdm
import numpy as np
#%% Paths
TRAIN_PATH = r"C:\Users\ASUS\Desktop\miniclef\images"
CLASSES = r"C:\Users\ASUS\Desktop\miniclef\images"
CHKPNT = r"C:\Users\ASUS\Desktop\miniclef\model\vit_base_patch14_reg4_dinov2_lvd142m_pc24_onlyclassifier_then_all\model_best.pth.tar"
MERGED = r"C:\Users\ASUS\Desktop\miniclef\merged_samples"
#%% Parameters
PATCH_HEIGHT = 800
PATCH_WIDTH = 800
MIN_CLASS = 3
MAX_CLASS =6
MIN_SIZE = 150

#%% Patch generator
args = TRAIN_PATH, PATCH_HEIGHT, PATCH_WIDTH, MIN_CLASS, MAX_CLASS, MIN_SIZE
img_generator = img_gen.patched_img(*args)

#%% Get embeddings

# predefined class A: 1355934
# predefined class B: 1356331

for i in tqdm(range(0,100)):
    time.sleep(0.1)
    patch, label, _ = img_generator.generate_patched_img()
    recolor = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
    generator = Image.fromarray(recolor)
    cv2.imshow("patch", np.array(generator))
    cv2.waitKey(0)
    #path = os.path.join(MERGED, str(i) + ".jpg")
    #generator.save(path)

