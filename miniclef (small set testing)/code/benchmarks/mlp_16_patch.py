import os
import pandas as pd
from ast import literal_eval
from tqdm import tqdm
import seaborn as sns
import math
FOLDER_64 = r"C:\Users\ASUS\Desktop\miniclef\code\benchmarks\16"
#%%
variance = []
top_pred = []
for img_name in tqdm(os.listdir(FOLDER_64)):
    df = pd.read_csv(os.path.join(FOLDER_64, img_name))
    for index, row in df.iterrows():
        variance.append(row['variance'])
        preds = literal_eval(row['top_100'])
        img_label = []
        img_score = []
        top_pred.append(preds[0][1])
#%%
print(len(variance))
print(len(top_pred))

g = sns.scatterplot(x=variance, y=top_pred, s=40,  edgecolor='k', alpha=0.8, legend="full")
#%%
s = pd.Series(variance)
def confidence(x,a):
    return (math.sqrt((a-x)/a))*100


