# -*- coding: utf-8 -*-
"""
Created on Wed May 22 11:22:22 2024

@author: user
"""
from collections import Counter
import pandas as pd
import csv
import matplotlib.pyplot as plt
#TXT_PREDICTION = r"F:\20240502_plantclef2024\1_submission\predictions\whole_image\run2_cont_160k\predictions_official_testset_whole_image_10_crops_run2_cont_160k_visualised.txt"
#TXT_PREDICTION = r"F:\20240502_plantclef2024\1_submission\predictions\4patches\run2_160k\predictions_official_testset_4patches_10_crops_run2_cont_160k_visualised.txt"
TXT_PREDICTION = r"C:\Users\User\plantclef\code\benchmark\10_sub_sample_vit.txt"
CSV_MAP_PATH = r"C:\Users\User\plantclef\code\benchmark\PlantCLEF2024_mapping_with_labels.csv"
CSV_SUBMISSION = r"C:\Users\User\plantclef\code\benchmark\10_sub_sample_vit_processed.csv"

MAP_DF = pd.read_csv(CSV_MAP_PATH, sep=';', encoding="utf-8")
MAP_LABELS = MAP_DF['species_id'].to_list()
MAP_LABELS = [str(x) for x in MAP_LABELS]
MAP_GENUS = MAP_DF['genus'].to_list()



with open(TXT_PREDICTION, 'r') as txt:
    lines = [x.strip() for x in txt.readlines()]

files_list = [x.split(";")[0] for x in lines]
labels_list_ = [x.split(";")[1] for x in lines]
q = [x.strip("[]") for x in labels_list_]
qq = [x.replace("'", "") for x in q]
qqq = [x.split(",") for x in qq]
labels_list = [[value.strip() for value in x] for x in qqq]

probabilities_list_ = [x.split(";")[2] for x in lines]
p = [x.strip("[]") for x in probabilities_list_]
pp = [x.replace("'", "") for x in p]
ppp = [x.split(",") for x in pp]
probabilities_list = [[float(value) for value in x] for x in ppp]


files_dictionary = {}

for filename, labels, probabilities in zip(files_list, labels_list, probabilities_list):
    base_name = '_'.join(filename.split('_')[:-2])
    
    if base_name not in files_dictionary:
        files_dictionary[base_name] = {"filenames":[], "labels":[], "probabilities":[]}

    
    files_dictionary[base_name]["filenames"].append(filename)
    files_dictionary[base_name]["labels"].append(labels)
    files_dictionary[base_name]["probabilities"].append(probabilities)

# =============================================================================
# Get final predictions
# =============================================================================
total_composite_scores = []
counter = 0
prediction_dictionary = {}
for base_filename in files_dictionary:
    counter += 1
    print(counter, base_filename)
    if base_filename not in prediction_dictionary:
#        prediction_dictionary[base_filename] = {"final_labels":[], "final_scores":[]}
        prediction_dictionary[base_filename] = []

    all_labels_arr = files_dictionary[base_filename]["labels"]
    all_probabilities_arr = files_dictionary[base_filename]["probabilities"]
    
    all_labels = [] # if 16 patches, there will be 160 labels (Top-10 results)
    all_probabilities = [] # if 16 patches, there will be 160 prob scores (Top-10 results)
    for lbl_list in all_labels_arr:
        for label in lbl_list:
            all_labels.append(label)
            
    for prob_list in all_probabilities_arr:
        for prob in prob_list:
            all_probabilities.append(prob)



    # Normalize probabilities
    total_probability = sum(all_probabilities)
    probabilities = [p / total_probability for p in all_probabilities]
    
    # Calculate occurrence count
    label_counts = Counter(all_labels)
    
    # Assign weights (adjust these based on your preference)
    probability_weight = 0.3
    occurrence_weight = 0.7
    
    # Calculate composite scores
    composite_scores = {}
    for label in label_counts:
        probability_score = probabilities[all_labels.index(label)]
        occurrence_score = label_counts[label]
        composite_scores[label] = (probability_weight * probability_score) + (occurrence_weight * occurrence_score)
    
    # Sort labels based on composite scores
    sorted_labels = sorted(composite_scores, key=lambda x: composite_scores[x], reverse=True)
    
    # Select top 10 labels
    top_10_labels = sorted_labels[:10]
    
    
    selected_labels = []
    selected_scores = []
    # Print top 10 labels
    print("Top 10 labels based on composite score:")
    for label in top_10_labels:
        composite_score = composite_scores[label]
        print(f"{label}: {composite_score}")
        
        if composite_score > 0:#3#1.41:#0.62:
            selected_labels.append(label)
            selected_scores.append(composite_score)
    
    selected_genus = []
    for l in selected_labels:
        index = MAP_LABELS.index(l)
        genus = MAP_GENUS[index]
        selected_genus.append(genus)



    seen = set()
    final_labels = []
    final_scores = []
    for i, (gen, lbl, score) in enumerate(zip(selected_genus, selected_labels, selected_scores)):
        # If the element is not seen before, add it to the unique lists
        if gen not in seen:
            final_labels.append(int(lbl))
            final_scores.append(score)
            seen.add(gen)
            total_composite_scores.append(score)
       
    
#    prediction_dictionary[base_filename]["final_labels"] = final_labels
#    prediction_dictionary[base_filename]["final_scores"] = final_scores
    prediction_dictionary[base_filename] = final_labels
    


plt.hist(total_composite_scores, bins=10, edgecolor='black')
plt.xlabel('Composite score')
plt.ylabel('Occurrence')
plt.title('Distribution of Composite Scores')
plt.show()

# =============================================================================
# Write CSV
# =============================================================================
df = pd.DataFrame(prediction_dictionary.items(), columns=['plot_id', 'species_ids'])
df.to_csv(CSV_SUBMISSION, sep=';', index=False, header=True, quoting=csv.QUOTE_NONE)



























   

