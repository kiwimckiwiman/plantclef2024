import torch
import timm
import os
import numpy as np
from PIL import Image
import csv
import pandas as pd
import warnings
import cv2

#%%

CLASSES_ = r"C:\Users\User\plantclef\code\model\pretrained_models\class_mapping.txt"
SPECIES = r"C:\Users\User\plantclef\code\model\pretrained_models\species_id_to_name.txt"

# Patches
PATCHES = r"D:\plantclef2024\patch_64"

CHKPNT = r"C:\Users\User\plantclef\code\model\pretrained_models\vit_base_patch14_reg4_dinov2_lvd142m_pc24_onlyclassifier_then_all\model_best.pth.tar"

OUTPUT = r"C:\Users\User\plantclef\code\benchmark\64\pls.txt"

#%%
# Functions for classifying with ViT
def load_class_mapping(class_list_file):
    with open(class_list_file) as f:
        class_index_to_class_name = {i: line.strip() for i, line in enumerate(f)}
    return class_index_to_class_name


def load_species_mapping(species_map_file):
    df = pd.read_csv(species_map_file, sep=';', quoting=1, dtype={'species_id': str})
    df = df.set_index('species_id')
    return  df['species'].to_dict()

cid_to_spid = load_class_mapping(CLASSES_)
spid_to_sp = load_species_mapping(SPECIES)

print(len(cid_to_spid))
#%%

#Functions
def preprocess_image(image_input_filepath, center_only=False, crop_fraction=1):
    img = []
    try:
        im = cv2.imread(image_input_filepath)
    
        if im is None:
           im = cv2.cvtColor(np.asarray(Image.open(image_input_filepath).convert('RGB')),cv2.COLOR_RGB2BGR)
        im = cv2.resize(im,(518,518))
    
        if np.ndim(im) == 2:
            im = cv2.cvtColor(im,cv2.COLOR_GRAY2RGB)          
        else:
            im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
               
        # Flip image
        flip_img = cv2.flip(im, 1)
        
        im1 = im[0:450,0:450,:]
        im2 = im[0:450,-450:,:]
        im3 = im[-450:,0:450,:]
        im4 = im[-450:,-450:,:]
        im5 = im[33:483,33:483,:]
        
        imtemp = [cv2.resize(ims,(518,518)) for ims in (im1,im2,im3,im4,im5)]
                
        [img.append(ims) for ims in imtemp]


        flip_im1 = flip_img[0:450,0:450,:]
        flip_im2 = flip_img[0:450,-450:,:]
        flip_im3 = flip_img[-450:,0:450,:]
        flip_im4 = flip_img[-450:,-450:,:]
        flip_im5 = flip_img[33:483,33:483,:]
        
        flip_imtemp = [cv2.resize(imf,(518,518)) for imf in (flip_im1,flip_im2,flip_im3,flip_im4,flip_im5)]
                
        [img.append(imf) for imf in flip_imtemp]  
    except:
        raise ValueError("Cannot read image input")

    # Convert the list of images to a numpy array and normalize
    output = np.asarray(img, dtype=np.float32) / 255.0
    

    return output

#Models
device = torch.device('cuda')
print(f"is cuda enabled: {torch.cuda.is_available()}")

#ViT feature extractor
def init_vit_classifier():
    model = timm.create_model(
        'vit_base_patch14_reg4_dinov2.lvd142m',
        pretrained=False,
        num_classes=len(cid_to_spid),
        checkpoint_path=CHKPNT
    )
    model = model.eval()
    model = model.to(device)
    print("ViT Model loaded")
    return model
    
#ViT feature extractor
def vit_only(patch, model):
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)
    
    img = Image.open(patch)
    output = model(transforms(img).unsqueeze(0).to(device))  #unsqueeze single image into batch of 1
    n = 10 #top-n to get variance from (hyper param)
    probabilities, indices = torch.topk(output.softmax(dim=1) * 100, k=n)
    img.close()
    
    preds = []
    for i in range(n):
        preds.append([cid_to_spid[indices[i]], probabilities[i]])
    return preds

#%%

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
    
final_preds = []
vit = init_vit_classifier()
patches = os.listdir(PATCHES)
count = 0
for image in patches:
    count += 1
    print("====================================")
    print(f" Vit: Now predicting {image} | {count} of 1695")
    folder = os.path.join(PATCHES, image)
    for patch in os.listdir(folder):
        img_path = os.path.join(folder, patch)
        sub_patches = preprocess_image(img_path)
        predictions = []
        classes_present = []
        for sp in sub_patches:
            sub_img = Image.fromarray(sp)
            preds = vit_only(img_path, vit)
            for pair in preds:
                if pair[0] not in classes_present:
                    classes_present.append(pair[0])
            predictions.append(preds)
        
        avg_patch_score = []
        for c in classes_present:
            count = 0
            score = 0
            for pred in predictions:
                for p in pred:
                    if(p[0] == c):
                        count += 1
                        score += p[1]
            avg_patch_score.append(score/count)
            
        avg_patch_score = np.array(avg_patch_score)
        classes_present = np.array(classes_present)
        
        score_indices = np.argsort(avg_patch_score)[::-1][:10] #sort scores
        sorted_score = avg_patch_score[score_indices] #keep indices
        
        sorted_label = [classes_present[i] for i in score_indices]
        
        split_patch_name = patch.split("_")
        final_patch_name = split_patch_name[0] + "_patch_" + split_patch_name[1]
        final_preds.append([final_patch_name, sorted_label, sorted_score])

file_name = "sub_patch_res.csv"
OUTPUT = r"C:\Users\User\plantclef\code\benchmark"

with open(os.path.join(OUTPUT, file_name), 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter=';')
    csvwriter.writerows(final_preds)