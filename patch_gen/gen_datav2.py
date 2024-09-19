import numpy as np
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import pandas as pd
import timm
from PIL import Image
import tqdm as tqdm
import random
#%%
TRAIN = r"D:\plantclef2024\PlantCLEF2024singleplanttrainingdata/train"
EMBEDS = r"D:\plantclef2024\embedsv2"
CHKPNT = r"C:\Users\User\plantclef\code\model\pretrained_models\vit_base_patch14_reg4_dinov2_lvd142m_pc24_onlyclassifier_then_all\model_best.pth.tar"
#%%
print(f"Using GPU: {torch.cuda.is_available()}")
model = timm.create_model(
    'vit_base_patch14_reg4_dinov2.lvd142m',
    pretrained=False,
    num_classes=7806,
    checkpoint_path=CHKPNT
    )
device = torch.device('cuda')
model = model.to(device)
model = model.eval()
print("Model loaded")
#%%
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)
#%%
curr= 1
failed = []
for file in os.listdir(TRAIN):
    TRAIN_CURR_FOLDER = os.path.join(TRAIN, file)
    EMBED_CURR_FOLDER = os.path.join(EMBEDS, file)
    #if embed folder does not exist
    if (os.path.exists(EMBED_CURR_FOLDER) == False):
        os.mkdir(EMBED_CURR_FOLDER)
        
    curr_list = os.listdir(EMBED_CURR_FOLDER)
    processed = False
    for i in curr_list:
        if "average" in i:
            processed = True
    
    if processed:
        print("already converted " + file)
    else:
        
        print("converting class " + file + " | " + str(curr) + " of " + str(len(os.listdir(TRAIN))))
        try:
            #extract embeddings
            for j in tqdm.tqdm(os.listdir(TRAIN_CURR_FOLDER)):
                try:
                    img = Image.open(os.path.join(TRAIN_CURR_FOLDER, j))
                    output = model.forward_features(transforms(img).unsqueeze(0).to(device))
                    output = model.forward_head(output, pre_logits=True)
                    torch.save(output, os.path.join(EMBED_CURR_FOLDER, (j.split(".")[0] + ".pt")))
                    img.close()
                except:
                    print("error with " + j)
                
            print(file + ": embeddings extracted!")
            #get embeds
            real_embeddings = []
            exp_labels = []
            for j in os.listdir(EMBED_CURR_FOLDER):
                embed = torch.load(os.path.join(os.path.join(EMBEDS, file), j))
                real_embeddings.append(embed.cpu().detach().numpy())
                exp_labels.append(0)
            print(file + ": calculating average")
            #average
            all_avg = []
            if(len(real_embeddings) > 1):
                for e in tqdm.tqdm(real_embeddings):
                    if len(all_avg) == 0:
                        all_avg = torch.from_numpy(e)
                    else:
                        all_avg = torch.add(all_avg, torch.from_numpy(e))
                all_avg = all_avg.apply_(lambda x: (x/len(all_avg)))
            else:
                all_avg = torch.from_numpy(real_embeddings[0])
            
            print(file + ":generating t-sne")
            #t-sne
            if(len(real_embeddings) > 40):
                real_embeddings.append(all_avg.cpu().detach().numpy())
                exp_labels.append(1)
                real_embeddings = np.array(real_embeddings)
                real_embeddings = real_embeddings.reshape(len(exp_labels), 768)
                tsne = TSNE(n_components=2 ,perplexity = 30 ,random_state = 1).fit_transform(real_embeddings)
            
                tsne_df = pd.DataFrame({'tsne_1': tsne[:,0], 'tsne_2': tsne[:,1], 'label': exp_labels})
                fig, ax = plt.subplots(1)
                sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', palette=sns.color_palette("hls", 2), data=tsne_df, legend="full", ax=ax,s=120)
                tsne_df_new = tsne_df.to_numpy()
                lim = (tsne_df_new.min()-5, tsne_df_new.max()+5)
            
                ax.set_xlim(-30,30)
                ax.set_ylim(-30,30)
                ax.set_aspect('equal')
                ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
                ax.set_title(file)
                filename = EMBED_CURR_FOLDER+"/extracted_embeddings_" + file + ".png"
                plt.savefig(filename)
                #plt.show()
                
            real_embeddings = []
            print(file + ": saving averages...")
            #saving embeds + image
            torch.save(all_avg, os.path.join(EMBED_CURR_FOLDER, (file + "_average_embedding.pt")))
            if(len(random_100_avg) != 0):
                torch.save(all_avg, os.path.join(EMBED_CURR_FOLDER, (file + "_100_random_average_embedding.pt")))
            print("=================================")
        except:
            print("error with " + file + "!")
            failed.append(file)
        
    curr = curr + 1
    print("currently failed to extract: " + str(failed))

print("=================================================")
print("failed: " + str(failed))