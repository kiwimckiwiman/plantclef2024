import os
import numpy
from tqdm import tqdm
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

EMBEDS = r"C:\Users\ASUS\Desktop\miniclef\embeds"
AVG = r"C:\Users\ASUS\Desktop\miniclef\avg_embeds"

for i in os.listdir(EMBEDS):
    EMBED_CURR = os.path.join(EMBEDS, i)
    AVG_CURR = os.path.join(AVG, i)
    #if embed folder does not exist
    if (os.path.exists(AVG_CURR) == False):
        os.mkdir(AVG_CURR)
    embeds = []
    for j in tqdm(os.listdir(EMBED_CURR)):
        embed = numpy.load(os.path.join(EMBED_CURR, j))
        embeds.append(embed)
        
    all_avg = []
    
    for e in tqdm(embeds):
        if len(all_avg) == 0:
            all_avg = e
        else:
            all_avg = numpy.add(all_avg, e)
            
    def avg(x):
        return x/100
    
    average_lambda = numpy.vectorize(avg)
    
    all_avg = average_lambda(all_avg)
    numpy.save(os.path.join(AVG_CURR, (i + "_average")), all_avg)
    
    labels = []
    for k in range(100):
        labels.append(0)
    labels.append(1)
    embeds.append(all_avg)
    
    embeds = numpy.array(embeds)
    embeds = embeds.reshape(len(embeds), 768)
    tsne = TSNE(n_components=2 ,perplexity = 30 ,random_state = 1).fit_transform(embeds)

    tsne_df = pd.DataFrame({'tsne_1': tsne[:,0], 'tsne_2': tsne[:,1], 'label': labels})
    fig, ax = plt.subplots(1)
    sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', palette=sns.color_palette("hls", 2), data=tsne_df, legend="full", ax=ax,s=120)
    tsne_df_new = tsne_df.to_numpy()
    lim = (tsne_df_new.min()-5, tsne_df_new.max()+5)

    ax.set_xlim(-15,15)
    ax.set_ylim(-15,15)
    ax.set_aspect('equal')
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    filename = AVG_CURR+"/extracted_embeddings_tsne" + str(i) + ".png"
    plt.title(str(i))
    plt.savefig(filename)
    plt.show()
    