import torch
import timm
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import csv
#%%
TEST_IMGS = r"C:\Users\ASUS\Desktop\miniclef\test_images\test_actual"
MODEL = r"C:\Users\ASUS\Desktop\miniclef\model\vit_base_patch14_reg4_dinov2_lvd142m_pc24_onlyclassifier_then_all\model_best.pth.tar"
CLASSES = r"C:\Users\ASUS\Desktop\miniclef\model\class_mapping.txt"
SPECIES = r"C:\Users\ASUS\Desktop\miniclef\model\species_id_to_name.txt"
RESULTS = r"C:\Users\ASUS\Desktop\miniclef\code\benchmarks\results\vit_whole_image_top_50.csv"
print(len(os.listdir(TEST_IMGS)))
#%%
def load_class_mapping(class_list_file):
    with open(class_list_file) as f:
        class_index_to_class_name = {i: line.strip() for i, line in enumerate(f)}
    return class_index_to_class_name


def load_species_mapping(species_map_file):
    df = pd.read_csv(species_map_file, sep=';', quoting=1, dtype={'species_id': str})
    df = df.set_index('species_id')
    return  df['species'].to_dict()

cid_to_spid = load_class_mapping(CLASSES)
spid_to_sp = load_species_mapping(SPECIES)

print(len(cid_to_spid))
#%%
print(f"Using GPU: {torch.cuda.is_available()}")
model = timm.create_model(
    'vit_base_patch14_reg4_dinov2.lvd142m',
    pretrained=False,
    num_classes=len(cid_to_spid),
    checkpoint_path=MODEL
    )
device = torch.device('cuda')
model = model.to(device)
model = model.eval()
print("Model loaded")
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)

top_n = 50
all_pred = []
for i in tqdm(os.listdir(TEST_IMGS)):
    img = Image.open(os.path.join(TEST_IMGS, i))
    plt.figure()
    plt.imshow(img)
    plt.show()
    output = model(transforms(img).unsqueeze(0).to(device))  # unsqueeze single image into batch of 1
    topN_probabilities, topN_class_indices = torch.topk(output.softmax(dim=1) * 100, k=top_n)
    topN_probabilities = topN_probabilities.cpu().detach().numpy()
    topN_class_indices = topN_class_indices.cpu().detach().numpy()
    img.close()
    predictions = []
    for proba, cid in zip(topN_probabilities[0], topN_class_indices[0]):
        species_id = cid_to_spid[cid]
        species = spid_to_sp[species_id]
        predictions.append([species_id, proba])
    all_pred.append([i, predictions])

#%%
fields = ['img_name', 'top_50']
with open(RESULTS, 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(fields)
    csvwriter.writerows(all_pred)