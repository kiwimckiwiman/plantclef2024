import os
from patchify import patchify
from tqdm import tqdm
from PIL import Image
import numpy as np
from math import ceil, sqrt
import cv2

TEST = r"D:\plantclef2024\PlantCLEF2024test"
PATCH_4 = r"D:\plantclef2024\patch_4"
PATCH_16 = r"D:\plantclef2024\patch_16"
PATCH_64 = r"D:\plantclef2024\patch_64"


def split_image_into_patches(image, n):
    # Image dimensions
    H, W, C = image.shape  # Assuming image is H x W x C (Height x Width x Channels)
    # Determine the number of patches in each dimension
    n_h = ceil(sqrt(n))
    n_w = ceil(n / n_h)
    
    # Calculate the size of each patch
    patch_height = H // n_h
    patch_width = W // n_w
    
    patches = []
    
    for i in range(n_h):
        for j in range(n_w):
            # Calculate the start and end indices for the current patch
            start_h = i * patch_height
            end_h = (i + 1) * patch_height if i != n_h - 1 else H
            start_w = j * patch_width
            end_w = (j + 1) * patch_width if j != n_w - 1 else W
            
            # Extract the patch
            patch = image[start_h:end_h, start_w:end_w, :]
            patches.append(patch)
    
    return patches


for name in tqdm(os.listdir(TEST)):
    folder = os.path.join(PATCH_4, name)
    if (os.path.exists(folder) == False):
        os.mkdir(folder)
    img = cv2.cvtColor(cv2.imread(os.path.join(TEST, name)), cv2.COLOR_BGR2RGB)
    patches = split_image_into_patches(img, 4)
    count = 0
    for i in patches:
      patch = Image.fromarray(i)
      patch.save(os.path.join(folder, name + "_" +str(count) + ".jpg"))
      count += 1
    
   
for name in tqdm(os.listdir(TEST)):
    folder = os.path.join(PATCH_16, name)
    if (os.path.exists(folder) == False):
        os.mkdir(folder)
    img = cv2.cvtColor(cv2.imread(os.path.join(TEST, name)), cv2.COLOR_BGR2RGB)
    patches = split_image_into_patches(img, 16)
    count = 0
    for i in patches:
      patch = Image.fromarray(i)
      patch.save(os.path.join(folder, name + "_" +str(count) + ".jpg"))
      count += 1
      
for name in tqdm(os.listdir(TEST)):
    folder = os.path.join(PATCH_64, name)
    if (os.path.exists(folder) == False):
        os.mkdir(folder)
    img = cv2.cvtColor(cv2.imread(os.path.join(TEST, name)), cv2.COLOR_BGR2RGB)
    patches = split_image_into_patches(img, 64)
    count = 0
    for i in patches:
      patch = Image.fromarray(i)
      patch.save(os.path.join(folder, name + "_" +str(count) + ".jpg"))
      count += 1