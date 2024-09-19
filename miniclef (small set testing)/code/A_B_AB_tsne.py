import os
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# predefined class A/1: 1355934
# predefined class B/2: 1356331
# predefined class C/3: 1356672
EMBEDS = r"C:\Users\ASUS\Desktop\miniclef\embeds"
MERGED_EMBEDS = r"C:\Users\ASUS\Desktop\miniclef\merged_embeds\AB"

embeds = []
exp_labels = []
count = 1
for i in tqdm(os.listdir(EMBEDS)[:3]):
    print(f"taking from class: {i} label {count}")
    folder = os.path.join(EMBEDS, i)
    for j in os.listdir(folder):
        embeds.append(np.load(os.path.join(folder, j)))
        exp_labels.append(count)
    count += 1

print(f"taking from: {MERGED_EMBEDS}")
for i in tqdm(os.listdir(MERGED_EMBEDS)):
    embeds.append(np.load(os.path.join(folder, j)))
    exp_labels.append(12)
    

real_embeddings = np.array(embeds)
real_embeddings = real_embeddings.reshape(len(exp_labels), 768)
tsne = TSNE(n_components=2 ,perplexity = 30 ,random_state = 1).fit_transform(real_embeddings)

tsne_df = pd.DataFrame({'tsne_1': tsne[:,0], 'tsne_2': tsne[:,1], 'label': exp_labels})
fig, ax = plt.subplots(1)
sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', palette=sns.color_palette("hls", 4), data=tsne_df, legend="full", ax=ax,s=120)
tsne_df_new = tsne_df.to_numpy()
lim = (tsne_df_new.min()-5, tsne_df_new.max()+5)

ax.set_xlim(-50,50)
ax.set_ylim(-50,50)
ax.set_aspect('equal')
ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
plt.show()