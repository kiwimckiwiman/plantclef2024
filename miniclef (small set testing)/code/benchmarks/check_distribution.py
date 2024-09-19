import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from ast import literal_eval
import math

df = pd.read_csv(r"C:\Users\ASUS\Desktop\miniclef\code\benchmarks\results\vit_patch_4_image_top_50.csv")
#df = df['top_50']

def sort_into_buckets(probabilities, bucket_size=1):
    num_buckets = int(100 / bucket_size) + 1
    buckets = np.zeros(num_buckets, dtype=int)
    for prob in probabilities:
        bucket_index = int(prob // bucket_size)
        buckets[bucket_index] += 1

    return buckets

# buckets = sort_into_buckets(probabilities)

# for i, count in enumerate(buckets):
#     print(f"Bucket {i * 0.5} to {(i + 1) * 0.5}: {count}")
probabilities = []
for index, row in df.iterrows():
    preds = literal_eval(row['top_50'])
    for i in preds:
        count = 1
        for j in i:
            if count == 10:
                break
            else:
                count += 1
                probabilities.append(j[1])

print(len(probabilities))
buckets = sort_into_buckets(probabilities)

y_axis = []
x_axis = []
for i, count in enumerate(buckets):
    if count == 0:
        y_axis.append(count)
    else:
        y_axis.append(math.log(count))
    x_axis.append(f"{i * 1} to {(i + 1) * 1}")
print(len(x_axis))
print(len(y_axis))

fig = plt.figure(figsize = (150, 50))
 
# creating the bar plot
plt.bar(x_axis, y_axis, color ='purple', 
        width = 0.1)
plt.xlabel("Probabilities (log(count))")
plt.ylabel("Occurences")
#plt.savefig(r"C:\Users\ASUS\Desktop\miniclef\code\benchmarks\results\vit_whole_image_top_50_bucket_1.png")