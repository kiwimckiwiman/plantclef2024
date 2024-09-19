import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from ast import literal_eval
import math
import random
import csv
import cv2
from PIL import Image

TEST = r"D:\plantclef2024\PlantCLEF2024test"
TRAIN = r"D:\plantclef2024\PlantCLEF2024singleplanttrainingdata\train"
df = pd.read_csv(r"C:\Users\User\plantclef\1_random_50_top_10.csv")

probabilities = []
for index, row in df.iterrows():
    name = row['img_name']
    preds = literal_eval(row['top_10'])
    probabilities.append([name, preds])
    
# print(probabilities[0])

for pred in probabilities:
    input("next")
    top_5 = pred[1][:5]
    for j in top_5:
        imgs_display = []
        folder = os.path.join(TRAIN, j[0])
        imgs = random.sample(os.listdir(folder), 4)
        for x in imgs:
            imgs_display.append(Image.open(os.path.join(folder, x)))
        fig = plt.figure(figsize=(10, 7)) 
        plt.title(f"image: {pred[0]} | class: {j[0]} | probability: {j[1]}")
        for i in range(len(imgs)):
            fig.add_subplot(2, 2, i+1) 
            plt.imshow(imgs_display[i]) 
            plt.axis('off')
    plt.show()
    plt.close()   