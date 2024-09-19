import numpy as np
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import pandas as pd
import timm
from PIL import Image
from tqdm import tqdm
import random
import cv2
#%%

#%%
TRAIN = r"D:\plantclef2024\PlantCLEF2024singleplanttrainingdata\train"
EMBEDS = r"D:\plantclef2024\merged_embeds_all"
CHKPNT = r"C:\Users\User\plantclef\code\model\pretrained_models\vit_base_patch14_reg4_dinov2_lvd142m_pc24_onlyclassifier_then_all\model_best.pth.tar"
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
class patch_creator():
    def __init__(self, top_left, top_right, bottom_right, bottom_left):
        self.top_left = top_left
        self.top_right = top_right
        self.bottom_right = bottom_right
        self.bottom_left = bottom_left

    def width(self):
        return self.top_right[0] - self.top_left[0]

    def height(self):
        return self.bottom_right[1] - self.top_right[1]

    def coords(self):
          print(f'top left: {self.top_left}, top right: {self.top_right}, bottom right: {self.bottom_right}, bottom left: {self.bottom_left}')

class patches():
    def __init__(self, height, width, patch_no, min_size):
        self.height = height
        self.width = width
        self.patch_no = patch_no
        self.min_size = min_size
        self.all_patches = []
        self.create_patches()

    def get_orientation(self):
        bool_list = [True]
        for i in range(1, self.patch_no - 1):
            bool_list.append(bool_list[-1] if i % 2 == 0 else not bool_list[-1])
        k = random.randint(0,100)%2
        if k == 0:
        	bool_list = list(map(lambda x: not x, bool_list))
        return bool_list

    def create_patches(self):
        curr = []
        curr.append(patch_creator([0,0], [self.width, 0], [self.width, self.height], [0, self.height]))
        orientations = self.get_orientation()
        for i in range(0, self.patch_no - 1):
            orientation = orientations[i]
            curr_patch = curr[i]
            top_left = curr_patch.top_left
            top_right = curr_patch.top_right
            bottom_right = curr_patch.bottom_right
            bottom_left = curr_patch.bottom_left
            if orientation == True:
                if(curr_patch.width() >= self.min_size*2):
                    x = random.randrange(round((curr_patch.width() - self.min_size)/2), round((curr_patch.width() + self.min_size)/2), round(self.min_size/3))

                    x_split_1 = [x, curr_patch.top_left[1]]
                    x_split_2 = [x, curr_patch.bottom_left[1]]

                    patch_1 = patch_creator(top_left, x_split_1, x_split_2, bottom_left)
                    patch_2 = patch_creator(x_split_1, top_right, bottom_right, x_split_2)

                    curr.append(patch_1)
                    curr.append(patch_2)
            else:
                if(curr_patch.height() >= self.min_size*2):
                    y = random.randrange(round((curr_patch.height() - self.min_size)/2), round((curr_patch.height() + self.min_size)/2), round(self.min_size/3))

                    y_split_1 = [curr_patch.top_left[0], y]
                    y_split_2 = [curr_patch.top_right[0], y]

                    patch_1 = patch_creator(top_left, top_right, y_split_2, y_split_1)
                    patch_2 = patch_creator(y_split_1, y_split_2, bottom_right, bottom_left)

                    curr.append(patch_1)
                    curr.append(patch_2)
        del curr [0:(self.patch_no-1)]
        self.all_patches = curr

class patched_img():
    def __init__(self, train_path, height, width, min_class, max_class, min_size):
        self.train_path = train_path
        self.height = height
        self.width = width
        self.min_class = min_class
        self.max_class = max_class
        self.min_size = min_size

    def generate_patched_img(self):
        def fit_to_patch(image, patch):
            patch_width = patch.width()
            patch_height = patch.height()
            image_height, image_width = image.shape[:2]
            if image_height >= patch_height:
                if image_width >= patch_width:
                    resized_image = image
                elif image_width < patch_width:
                    new_height = round((image_height * patch_width)/image_width)
                    resized_image = cv2.resize(image, (patch_width, new_height))
            elif image_height < patch_height:
                new_width = round((image_width * patch_height)/image_height)
                resized_image = cv2.resize(image, (new_width, patch_height))

            resized_image_height, resized_image_width = resized_image.shape[:2]
            x1 = round(abs(resized_image_width - patch_width)/2)
            x2 = x1 + patch_width
            y1 = round(abs(resized_image_height - patch_height)/2)
            y2 = y1 + patch_height
            cropped_image = resized_image[y1:y2, x1:x2]
            return cropped_image

        imgs_count = random.randint(self.min_class, self.max_class)

        classes = os.listdir(self.train_path)

        selected = random.sample(classes, imgs_count)
        imgs = []
        img_name = []
        for i in selected:
            class_path = os.path.join(self.train_path, i)
            random_img = random.choice(os.listdir(class_path))
            img_path = os.path.join(class_path, random_img)
            img = cv2.imread(img_path)
            imgs.append(img)
            img_name.append(os.path.join(class_path, random_img))
        patch_list = patches(self.height, self.width, imgs_count, self.min_size)
        canvas = np.full((self.height, self.width, 3), (255, 255, 255), dtype=np.uint8)
        for i in range(0, imgs_count):
            patch = fit_to_patch(imgs[i], patch_list.all_patches[i])
            x = patch_list.all_patches[i].top_left[0]
            y =  patch_list.all_patches[i].top_left[1]
            patch_height, patch_width, _ = patch.shape
            canvas[y:y+patch_height, x:x+patch_width] = patch

        return canvas, selected, img_name
#%%
PATCH_HEIGHT = 500
PATCH_WIDTH = 500
MIN_CLASS = 1
MAX_CLASS = 4
MIN_SIZE = 150
#%%
args = TRAIN, PATCH_HEIGHT, PATCH_WIDTH, MIN_CLASS, MAX_CLASS, MIN_SIZE
img_generator = patched_img(*args)
#%%
import time
for i in tqdm(range(198894)):
        patch, label, _ = img_generator.generate_patched_img()
        recolor = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(recolor)
        output = model.forward_features(transforms(img).unsqueeze(0).to(device))
        output = model.forward_head(output, pre_logits=True)
        np.save(os.path.join(EMBEDS, ("_".join(label) + "_" + str(time.time()).split(".")[0] + ".npy")), output.cpu().detach().numpy())
        img.close()