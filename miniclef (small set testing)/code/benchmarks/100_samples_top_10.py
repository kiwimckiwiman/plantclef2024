import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from ast import literal_eval
import math
import random
import csv

df = pd.read_csv(r"C:\Users\ASUS\Desktop\miniclef\code\benchmarks\results\vit_whole_image_top_50.csv")
#df = df['top_50']

probabilities = []
for index, row in df.iterrows():
    name = row['img_name']
    preds = literal_eval(row['top_50'])
    probabilities.append([name, preds])

samples_100 = random.sample(probabilities, 100)
print(len(samples_100))
samples_100_top_10 = []
for i in samples_100:
    samples_100_top_10.append([i[0], i[1][:10]])
    
for j in samples_100_top_10:
    print(j)
    break

half_1 = samples_100_top_10[:50]
half_2 = samples_100_top_10[50:]

print(len(half_1))
print(len(half_2))

#%%
RESULTS1 = r"C:\Users\ASUS\Desktop\miniclef\code\benchmarks\results\1_random_50_top_10.csv"
RESULTS2 = r"C:\Users\ASUS\Desktop\miniclef\code\benchmarks\results\2_random_50_top_10.csv"

fields = ['img_name', 'top_10']
with open(RESULTS1, 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(fields)
    csvwriter.writerows(half_1)
    
with open(RESULTS2, 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(fields)
    csvwriter.writerows(half_2)