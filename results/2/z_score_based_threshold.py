import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from ast import literal_eval
import math
import scipy.stats as stats
import csv

df = pd.read_csv(r"C:\Users\ASUS\Desktop\miniclef\code\benchmarks\results\vit_whole_image_top_50.csv")
RESULTS = r"C:\Users\ASUS\Desktop\miniclef\code\benchmarks\prepared\filtered_bench_1.csv"
#df = df['top_50']

count = 1
filtered = []
for index, row in df.iterrows():
    score = []
    label = []
    img_name = row['img_name']
    preds = literal_eval(row['top_50'])
    for i in preds:
        label.append(i[0])
        score.append(i[1])
        
    z_score = stats.zscore(score)
    fil_lab = []
    fil_score = []
    for j in range(len(z_score)):
        if(abs(z_score[j]) >= 2):
            fil_lab.append(int(label[j]))
            fil_score.append(score[j])
    
    filtered.append([img_name.split(".")[0], fil_lab])

#%%
fields = ['plot_id', 'species_ids']
with open(RESULTS, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter=';')
    csvwriter.writerow(fields)
    csvwriter.writerows(filtered)