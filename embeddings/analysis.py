import numpy as np
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import pandas as pd
import tqdm as tqdm
#%%
EMBEDS = r"D:\plantclef2024\embeds"
#%%

for i in os.listdir(EMBEDS):
    real_embeddings = []
    exp_labels = []
    for j in os.listdir(os.path.join(EMBEDS, i)):
        embed = torch.load(os.path.join(os.path.join(EMBEDS, i), j))
        real_embeddings.append(embed.cpu().detach().numpy())
        exp_labels.append(0)
    
    real_embeddings = np.array(real_embeddings)
    real_embeddings = real_embeddings.reshape(len(exp_labels), 768)
    tsne = TSNE(n_components=2 ,perplexity = 30,random_state = 1).fit_transform(real_embeddings)

    tsne_df = pd.DataFrame({'tsne_1': tsne[:,0], 'tsne_2': tsne[:,1], 'label': exp_labels})
    fig, ax = plt.subplots(1)
    sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', palette=sns.color_palette("hls", 6), data=tsne_df, legend="full", ax=ax,s=120)
    tsne_df_new = tsne_df.to_numpy()
    lim = (tsne_df_new.min()-5, tsne_df_new.max()+5)

    # you can change the limit
    ax.set_xlim(-30,30)
    ax.set_ylim(-30,30)
    ax.set_aspect('equal')
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    ax.set_title(i)
    plt.show()
#%%
for i in os.listdir(EMBEDS):
    real_embeddings = []
    exp_labels = []
    for j in os.listdir(os.path.join(EMBEDS, i)):
        embed = torch.load(os.path.join(os.path.join(EMBEDS, i), j))
        real_embeddings.append(embed.cpu().detach().numpy())
        exp_labels.append(0)
        
    all_avg = []
    for e in tqdm.tqdm(real_embeddings):
        if len(all_avg) == 0:
            all_avg = torch.from_numpy(e)
        else:
            all_avg = torch.add(all_avg, torch.from_numpy(e))
    all_avg = all_avg.apply_(lambda x: (x/len(all_avg)))
    
    real_embeddings.append(all_avg.cpu().detach().numpy())
    exp_labels.append(1)
    real_embeddings = np.array(real_embeddings)
    real_embeddings = real_embeddings.reshape(len(exp_labels), 768)
    tsne = TSNE(n_components=2 ,perplexity = 30,random_state = 1).fit_transform(real_embeddings)

    tsne_df = pd.DataFrame({'tsne_1': tsne[:,0], 'tsne_2': tsne[:,1], 'label': exp_labels})
    fig, ax = plt.subplots(1)
    sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', palette=sns.color_palette("hls", 6), data=tsne_df, legend="full", ax=ax,s=120)
    tsne_df_new = tsne_df.to_numpy()
    lim = (tsne_df_new.min()-5, tsne_df_new.max()+5)

    # you can change the limit
    ax.set_xlim(-30,30)
    ax.set_ylim(-30,30)
    ax.set_aspect('equal')
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    ax.set_title(i)
    plt.show()