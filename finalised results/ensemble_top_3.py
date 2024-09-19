import pandas as pd
import csv
from ast import literal_eval
#%%
MLP_5 = r"C:\Users\ASUS\Desktop\plantclef\finalised results\mlp_run_8_0.07_300_single_class_64_16_BMA_z_score.csv"
MLP_1 = r"C:\Users\ASUS\Desktop\plantclef\finalised results\mlp_run_6_0.05_100_single_class_64_16_BMA_z_score.csv"
VIT = r"C:\Users\ASUS\Desktop\plantclef\finalised results\vit_64_16_BMA_z_score.csv"
OUTPUT = r"C:\Users\ASUS\Desktop\plantclef\finalised results\vit_mlp_1_mlp_8_64_16_merged.csv"
#%%

mlp1 = pd.read_csv(MLP_1, delimiter=";")
mlp5 = pd.read_csv(MLP_5, delimiter=";")
vit = pd.read_csv(VIT, delimiter=";")

#%%

results = []
for index, row in mlp5.iterrows():
    preds = literal_eval(row['species_ids'])
    results.append([row['plot_id'], preds])

for index, row in vit.iterrows():
    preds = literal_eval(row['species_ids'])
    curr = results[index][1]
    for i in preds:
        if i not in curr:
            curr.append(i)
    results[index][1] = curr

for index, row in mlp1.iterrows():
    preds = literal_eval(row['species_ids'])
    curr = results[index][1]
    for i in preds:
        if i not in curr:
            curr.append(i)
    results[index][1] = curr

fields = ['plot_id', 'species_ids']
with open(OUTPUT, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=';')
        csvwriter.writerow(fields)
        csvwriter.writerows(results)