import os
import numpy
from tqdm import tqdm
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

EMBEDS = r"C:\Users\ASUS\Desktop\miniclef\embeds"
AVG = r"C:\Users\ASUS\Desktop\miniclef\avg_embeds"

class_name = "1361068"

EMBED_CURR = os.path.join(EMBEDS, class_name)
AVG_CURR = os.path.join(AVG, class_name)

embeds = []
labels = []

for i in os.listdir(EMBED_CURR):
    embeds.append(numpy.load(os.path.join(EMBED_CURR, i)))
    labels.append(0)
    
embeds.append(numpy.load(os.path.join(AVG_CURR, class_name + "_average.npy")))
labels.append(1)

embeds = numpy.array(embeds)
embeds = embeds.reshape(len(embeds), 768)
tsne = TSNE(n_components=2 ,perplexity = 30 ,random_state = 1).fit_transform(embeds)

tsne_df = pd.DataFrame({'tsne_1': tsne[:,0], 'tsne_2': tsne[:,1], 'label': labels})
fig, ax = plt.subplots(1)
sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', palette=sns.color_palette("hls", 2), data=tsne_df, legend="full", ax=ax,s=120)
tsne_df_new = tsne_df.to_numpy()
lim = (tsne_df_new.min()-5, tsne_df_new.max()+5)

ax.set_xlim(-150,150)
ax.set_ylim(-150,150)
ax.set_aspect('equal')
ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
plt.title(class_name + " individual")
plt.show()