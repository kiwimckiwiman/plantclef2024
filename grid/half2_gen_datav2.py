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
TRAIN = r"D:\plantclef2024\PlantCLEF2024singleplanttrainingdata/test"
EMBEDS = r"D:\plantclef2024\embeds_test"
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
img_list = os.listdir(TRAIN)
half = img_list
for file in half:
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
            img_list = os.listdir(TRAIN_CURR_FOLDER)
            for j in tqdm.tqdm(range(len(img_list))):
                if(j == 100):
                    break
                try:
                    img = Image.open(os.path.join(TRAIN_CURR_FOLDER, img_list[j]))
                    output = model.forward_features(transforms(img).unsqueeze(0).to(device))
                    output = model.forward_head(output, pre_logits=True)
                    np.save(os.path.join(EMBED_CURR_FOLDER, (img_list[j].split(".")[0] + ".npy")), output.cpu().detach().numpy())
                    img.close()
                except:
                    print("error with " + img_list[i])
                    continue
            print(file + ": embeddings extracted!")
            #get embeds
            real_embeddings = []
            exp_labels = []
            for j in os.listdir(EMBED_CURR_FOLDER):
                embed = np.load(os.path.join(EMBED_CURR_FOLDER, j))
                real_embeddings.append(embed)
                exp_labels.append(0)
            print(file + ": calculating average")
            
            #average
               
            all_avg = []
            if(len(real_embeddings) > 1):
                for e in tqdm.tqdm(real_embeddings):
                    if len(all_avg) == 0:
                        all_avg = e
                    else:
                        all_avg = np.add(all_avg, e)
                def avg(x):
                    return x/100
                
                average_lambda = np.vectorize(avg)
                
                all_avg = average_lambda(all_avg)
            else:
                all_avg = real_embeddings[0]
            
            # print(file + ": deleting embeddings")
            # #file deletion
            # for i in os.listdir(EMBED_CURR_FOLDER):
            #     os.remove(os.path.join(EMBED_CURR_FOLDER, i))
                
            print(file + ":generating t-sne")
            #t-sne
            if(len(real_embeddings) > 40):
                real_embeddings.append(all_avg)
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
            np.save(os.path.join(EMBED_CURR_FOLDER, (file + "_average_embedding.npy")), all_avg)
            print("=================================")
        except:
            print("error with " + file + "!")
            failed.append(file)
        
    curr = curr + 1
    print("currently failed to extract: " + str(failed))

print("=================================================")
print("failed: " + str(failed))