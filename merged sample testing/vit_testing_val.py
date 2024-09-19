import os
import timm
from PIL import Image
import torch
import pandas as pd
from tqdm import tqdm
TRAIN = r"D:\plantclef2024\PlantCLEF2024singleplanttrainingdata/train" 
TEST = r"D:\plantclef2024\PlantCLEF2024singleplanttrainingdata/test"
VAL = r"D:\plantclef2024\PlantCLEF2024singleplanttrainingdata/val"
CHKPNT = r"C:\Users\User\plantclef\code\model\pretrained_models\vit_base_patch14_reg4_dinov2_lvd142m_pc24_onlyclassifier_then_all\model_best.pth.tar"
SPECIES = r"C:\Users\User\plantclef\code\model\pretrained_models\species_id_to_name.txt"
CLASSES = r"C:\Users\User\plantclef\code\model\pretrained_models\class_mapping.txt"
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
    num_classes=7806,
    checkpoint_path=CHKPNT
    )
device = torch.device('cuda')
model = model.to(device)
model = model.eval()
print("Model loaded")
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)
#%%
# test = r"D:\plantclef2024\PlantCLEF2024singleplanttrainingdata\test\1744928\067d8f35fa8f822d2cee6e395b3b000c4189fd03.jpg"
# class_true = '1744928'
# top_n = 1
# img = Image.open(test)
# img = transforms(img).unsqueeze(0).to(device)
# output = model(img)  # unsqueeze single image into batch of 1
# topN_probabilities, topN_class_indices = torch.topk(output.softmax(dim=1) * 100, k=top_n)
# topN_probabilities = topN_probabilities.cpu().detach().numpy()
# topN_class_indices = topN_class_indices.cpu().detach().numpy()

# for proba, cid in zip(topN_probabilities[0], topN_class_indices[0]):
#     species_id = cid_to_spid[cid]
#     species = spid_to_sp[species_id]
#     if(species_id == class_true):
#         print('yes')

#%%
val_acc = 0
running = 0
top_n = 1
for i in os.listdir(VAL):
    folder_acc = 0
    curr = 0
    curr_folder = os.path.join(VAL, i)
    running += len(os.listdir(curr_folder))
    for j in (os.listdir(curr_folder)):
        if curr == 20:
            break
        try:
            img_path = os.path.join(curr_folder, j)
            img = Image.open(img_path)
            output = model(transforms(img).unsqueeze(0).to(device))  # unsqueeze single image into batch of 1
            topN_probabilities, topN_class_indices = torch.topk(output.softmax(dim=1) * 100, k=top_n)
            topN_probabilities = topN_probabilities.cpu().detach().numpy()
            topN_class_indices = topN_class_indices.cpu().detach().numpy()
            img.close()
            for proba, cid in zip(topN_probabilities[0], topN_class_indices[0]):
                species_id = cid_to_spid[cid]
                species = spid_to_sp[species_id]
                if(species_id == i):
                    folder_acc += 1
                    val_acc += 1
            curr += 1
        except:
            print()
    print(f"Accuracy for class {i}:{folder_acc/curr}")
print(f"Accuracy for train set: {val_acc/running}")
#%%
train_acc = 0
running = 0
top_n = 1
for i in os.listdir(TRAIN):
    folder_acc = 0
    curr = 0
    curr_folder = os.path.join(TRAIN, i)
    running += len(os.listdir(curr_folder))
    for j in (os.listdir(curr_folder)):
        if curr == 20:
            break
        try:
            img_path = os.path.join(curr_folder, j)
            img = Image.open(img_path)
            output = model(transforms(img).unsqueeze(0).to(device))  # unsqueeze single image into batch of 1
            topN_probabilities, topN_class_indices = torch.topk(output.softmax(dim=1) * 100, k=top_n)
            topN_probabilities = topN_probabilities.cpu().detach().numpy()
            topN_class_indices = topN_class_indices.cpu().detach().numpy()
            img.close()
            for proba, cid in zip(topN_probabilities[0], topN_class_indices[0]):
                species_id = cid_to_spid[cid]
                species = spid_to_sp[species_id]
                if(species_id == i):
                    folder_acc += 1
                    train_acc += 1
            curr += 1
        except:
            print()
    print(f"Accuracy for class {i}:{folder_acc/curr}")
print(f"Accuracy for train set: {train_acc/running}")

#%%
test_acc = 0
running = 0
top_n = 1
for i in os.listdir(TEST):
    folder_acc = 0
    curr = 0
    curr_folder = os.path.join(TEST, i)
    running += len(os.listdir(curr_folder))
    for j in (os.listdir(curr_folder)):
        if curr == 20:
            break
        try:
            img_path = os.path.join(curr_folder, j)
            img = Image.open(img_path)
            output = model(transforms(img).unsqueeze(0).to(device))  # unsqueeze single image into batch of 1
            topN_probabilities, topN_class_indices = torch.topk(output.softmax(dim=1) * 100, k=top_n)
            topN_probabilities = topN_probabilities.cpu().detach().numpy()
            topN_class_indices = topN_class_indices.cpu().detach().numpy()
            img.close()
            for proba, cid in zip(topN_probabilities[0], topN_class_indices[0]):
                species_id = cid_to_spid[cid]
                species = spid_to_sp[species_id]
                if(species_id == i):
                    folder_acc += 1
                    test_acc += 1
            curr += 1
        except:
            print()
    print(f"Accuracy for class {i}:{folder_acc/curr}")
print(f"Accuracy for train set: {test_acc/running}")