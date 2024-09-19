import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from ast import literal_eval
import math
import random
import csv
import cv2
import matplotlib.pyplot as plt 

TEST = r"C:\Users\ASUS\Desktop\miniclef\test_images\test_actual"
TRAIN = r"C:\Users\ASUS\Desktop\miniclef\images"
df = pd.read_csv(r"C:\Users\ASUS\Desktop\miniclef\code\benchmarks\results\1_random_50_top_10.csv")

probabilities = []
for index, row in df.iterrows():
    name = row['img_name']
    preds = literal_eval(row['top_10'])
    probabilities.append([name, preds])
    
# print(probabilities[0])

for i in probabilities:
    input("next")
    img = cv2.imread(os.path.join(TEST, i[0]))
    #img = cv2.resize(img, (1000, 1000))
    plt.title(f"image: {i[0]}")
    plt.imshow(img) 
    plt.axis('off')
    plt.show()
    plt.close()   
