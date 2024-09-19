import torch
import timm
import os
import numpy as np
from PIL import Image
import csv
import pandas as pd
import warnings
import cv2
from tqdm import tqdm
import sys

CLASSES_ = r"C:\Users\User\plantclef\code\model\pretrained_models\class_mapping.txt"
SPECIES = r"C:\Users\User\plantclef\code\model\pretrained_models\species_id_to_name.txt"

# Patches
PATCHES = r"D:\plantclef2024\patch_16"

CHKPNT = r"C:\Users\User\plantclef\code\model\pretrained_models\vit_base_patch14_reg4_dinov2_lvd142m_pc24_onlyclassifier_then_all\model_best.pth.tar"

OUTPUT = r"C:\Users\User\plantclef\code\benchmark\64\pls.txt"

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
        
        im1 = im[0:300,0:300,:]
        im2 = im[0:300,-300:,:]
        im3 = im[-300:,0:300,:]
        im4 = im[-300:,-300:,:]
        im5 = im[73:443,73:443,:]
        
        imtemp = [cv2.resize(ims,(518,518)) for ims in (im1,im2,im3,im4,im5)]
                
        [img.append(ims) for ims in imtemp]


        flip_im1 = flip_img[0:300,0:300,:]
        flip_im2 = flip_img[0:300,-300:,:]
        flip_im3 = flip_img[-300:,0:300,:]
        flip_im4 = flip_img[-300:,-300:,:]
        flip_im5 = flip_img[73:443,73:443,:]
        flip_imtemp = [cv2.resize(imf,(518,518)) for imf in (flip_im1,flip_im2,flip_im3,flip_im4,flip_im5)]
                
        [img.append(imf) for imf in flip_imtemp]  
    except:
        raise ValueError("Cannot read image input")

    # Convert the list of images to a numpy array and normalize
    output = np.asarray(img, dtype=np.float32)
    

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
    
    output = model(transforms(patch).unsqueeze(0).to(device))  #unsqueeze single image into batch of 1
    n = 10 #top-n to get variance from (hyper param)
    probs, indices = torch.topk(output.softmax(dim=1) * 100, k=n)
    probs = probs.cpu().detach().numpy()
    indices = indices.cpu().detach().numpy()
    preds = []
    for proba, cid in zip(probs[0], indices[0]):
        species_id = cid_to_spid[cid]
        preds.append([species_id, proba])
    return preds

#%%

def plsplsplspls2(process):
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
        
    final_preds = []
    vit = init_vit_classifier()
    patch_list = os.listdir(PATCHES)
    def split_into_batch(lst, chunk_size):
        return list(zip(*[iter(lst)] * chunk_size))
    count = 0
    patch_batch = split_into_batch(patch_list, int(1695/5))
    for image in patch_batch[int(process)]:
        count += 1
        print("====================================")
        print(f" Vit 10_subsample {process}: Now predicting {image} | {count} of 1695")
        folder = os.path.join(PATCHES, image)
        for patch in os.listdir(folder):
            img_path = os.path.join(folder, patch)
            sub_patches = preprocess_image(img_path)
            predictions = []
            classes_present = []
            for sp in sub_patches:
                sub_img = Image.fromarray(np.uint8(sp))
                preds = vit_only(sub_img, vit)
                sub_img.close()
                for pair in preds:
                    if pair[0] not in classes_present:
                        classes_present.append(pair[0])
                predictions.append(preds)
    
            avg_patch_score = []
            for c in classes_present:
                count_avg = 0
                score = 0
                for pred in predictions:
                    for p in pred:
                        if(p[0] == c):
                            count_avg += 1
                            score += p[1]
                avg_patch_score.append(score/count_avg)
    
            avg_patch_score = np.array(avg_patch_score)
            classes_present = np.array(classes_present)
            
            score_indices = np.argsort(avg_patch_score)[::-1][:10] #sort scores
            sorted_score = avg_patch_score[score_indices] #keep indices
            sorted_score = [str(i) for i in sorted_score]
            
            sorted_label = [classes_present[i] for i in score_indices]
            
            split_patch_name = patch.split("_")
            final_patch_name = split_patch_name[0] + "_patch_" + split_patch_name[1]
            final_preds.append([final_patch_name, sorted_label, sorted_score])
        
    file_name = "sub_patch_res_" + str(process) + ".csv"
    OUTPUT = r"C:\Users\User\plantclef\code\benchmark"
    
    with open(os.path.join(OUTPUT, file_name), 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=';')
        csvwriter.writerows(final_preds)
    
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python sub_script.py <identifier>")
        sys.exit(1)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    
    identifier = sys.argv[1]
    plsplsplspls2(identifier)