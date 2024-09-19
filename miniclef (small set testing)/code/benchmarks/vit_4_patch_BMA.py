import pandas as pd
from ast import literal_eval
import math
import seaborn as sns

df = pd.read_csv(r"C:\Users\ASUS\Desktop\miniclef\code\benchmarks\results\vit_patch_4_image_top_50.csv")
RESULTS = r"C:\Users\ASUS\Desktop\miniclef\code\benchmarks\prepared\vit_4_patch_BMA_z_score.csv"


def variance(data):
     n = len(data)
     mean = sum(data) / n
     deviations = [(x - mean) ** 2 for x in data]
     variance = sum(deviations) / n
     return variance

all_variance = []
top_pred = []
count = 0
for index, row in df.iterrows():
    img_name = row['img_name']
    preds = literal_eval(row['top_50'])
    img_label = []
    img_score = []
    for i in preds:
        count += 1
        score = []
        label = []
        for p in i:
            score.append(p[1]/100)
        all_variance.append(abs(math.log(variance(score))))
        top_pred.append(score[0])

print(all_variance[0])  
print(top_pred[0]) 

g = sns.scatterplot(x=all_variance, y=top_pred, s=40,  edgecolor='k', alpha=0.8, legend="full")

s = pd.Series(all_variance)
print(s.describe())
