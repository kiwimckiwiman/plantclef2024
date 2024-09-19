import os
import pandas as pd
from ast import literal_eval
import numpy as np
import matplotlib.pyplot as plt
import random
df = pd.read_csv(r"C:\Users\ASUS\Desktop\miniclef\code\benchmarks\results\vit_patch_4_image_top_50.csv")
#df = df['top_50']

probabilities = []
for index, row in df.iterrows():
    preds = literal_eval(row['top_50'])
    for i in preds:
        prob = []
        for p in i:
            prob.append(p[1])
        probabilities.append(prob)

print(probabilities[0])
#%%
def entropy(probabilities):
    probabilities = np.array(probabilities)
    return -(np.sum(probabilities * np.log2(probabilities + 1e-9)))

def sigmoid(x):
    return 1/(1+np.exp(x))


probabilities = random.sample(probabilities, 50)
x_ent = []
y_ent = []
x_sig_ent = []
y_sig_ent = []
random.shuffle(probabilities)
for pred in probabilities:
    x_ent.append(pred[0])
    y_ent.append(entropy(pred))
print(y_ent)
#%%
# for pred in probabilities:
#     x_sig_ent.append(prob[0])
#     y_sig_ent.append(sigmoid(entropy(pred)))

# fig1 = plt.figure(figsize = (150, 50))
 
# # creating the bar plot
# plt.bar(x_ent, y_ent, color ='purple', 
#         width = 0.1)
# plt.xlabel("Predictions")
# plt.ylabel("Entropy")
# plt.show()
print("creating graph")
fig2 = plt.figure(figsize = (300, 100))

# for i in range(len(x_ent)):
#     print(x_ent[i], y_ent[i])
#     print(x_ent[i], sigmoid(y_ent[i]))
#     print("=========")
# creating the bar plot
plt.bar(x_ent[:50], y_ent[:50], color ='red', 
        width = 0.1)
plt.xlabel("Predictions")
plt.ylabel("Entropy (Sigmoid)")
plt.show()