import os
import pandas as pd
from ast import literal_eval
from tqdm import tqdm
import seaborn as sns
import math
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
FOLDER_64 = r"C:\Users\ASUS\Desktop\miniclef\code\benchmarks\64"
#%%

def confidence(x, a):
    a = a + 1
    return (math.sqrt((a-x)/a))

# for img_name in tqdm(os.listdir(FOLDER_64)):
#     df = pd.read_csv(os.path.join(FOLDER_64, img_name))
#     variance = []
#     top_pred = []
#     for index, row in df.iterrows():
#         variance.append(row['variance'])
#         preds = literal_eval(row['top_100'])
#         img_label = []
#         img_score = []  
    
#     log_var = [abs(math.log(i, 10)) for i in variance]
#     confidence = [confidence(i, max(log_var)) for i in log_var]
    
#     g = sns.scatterplot(x=log_var, y=confidence, s=40,  edgecolor='k', alpha=0.8, legend="full")
#     print(pd.Series(log_var).describe())
#     print(pd.Series(confidence).describe())
#     break

conf = []
top_pred = []
variance = []
predictions = []
total = []
df = pd.read_csv(os.path.join(FOLDER_64, 'CBN-can-D3-20230705.csv'))

for index, row in df.iterrows():
    conf.append(abs(math.log(row['variance'], 10)))
    preds = literal_eval(row['top_100'])
    variance.append(row['variance'])
    top_pred.append(preds[0][1])
    predictions.append([i[1] for i in preds])
    score_count = 0
    for i in preds[:5]:
        score_count += i[1]
    total.append(score_count)
#%%

pred_conf = [round(confidence(i, pd.Series(conf).describe()[-1]), 2) for i in conf]
pred_conf = np.array(pred_conf).reshape(8,8)

total = [round(i, 2) for i in total]
total = np.array(total).reshape(8,8)

top_pred = [round(i, 2) for i in top_pred]
top_pred = np.array(top_pred).reshape(8,8)
#%%

fig, ax = plt.subplots(figsize=(13,7))
ax.set_xticks([])
ax.set_yticks([])
ax.axis('off')
sns.heatmap(pred_conf,annot=pred_conf,fmt="",cmap='RdYlGn',linewidths=0.30,ax=ax)
plt.show()

fig, ax = plt.subplots(figsize=(13,7))
ax.set_xticks([])
ax.set_yticks([])
ax.axis('off')
sns.heatmap(total,annot=total,fmt="",cmap='RdYlGn',linewidths=0.30,ax=ax)
plt.show()

fig, ax = plt.subplots(figsize=(13,7))
ax.set_xticks([])
ax.set_yticks([])
ax.axis('off')
sns.heatmap(top_pred,annot=top_pred,fmt="",cmap='RdYlGn',linewidths=0.30,ax=ax)
plt.show()

dataset = [[predictions[0].index(i), i] for i in predictions[0]]
#%%
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(predictions)

# Apply K-Means Clustering
kmeans = KMeans(n_clusters=2, random_state=42)
labels = kmeans.fit_predict(predictions)

# Plot the clusters
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='viridis')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('K-Means Clustering of Raw Predictions')
plt.show()

cluster_counts = np.bincount(labels)
print("Number of points in each cluster:", cluster_counts)

labels = np.array(labels).reshape(8,8)
fig, ax = plt.subplots(figsize=(13,7))
ax.set_xticks([])
ax.set_yticks([])
ax.axis('off')
sns.heatmap(labels,annot=labels,fmt="",cmap='RdYlGn',linewidths=0.30,ax=ax)
plt.show()
