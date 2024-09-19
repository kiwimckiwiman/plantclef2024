import os
import torch
import re
import tqdm as tqdm
EMBEDS = r"D:\plantclef2024\embeds"

total = []

for i in tqdm.tqdm(os.listdir(EMBEDS)):
    path = os.path.join(EMBEDS, i)
    for j in os.listdir(path):
        x = re.search("\d+_average", j)
        if x:
            embed = torch.load(os.path.join(path, j))
            if len(total) == 0:
                total = embed
            else:
                torch.add(total, embed)
#%%
print(total)
#%%
avg = total
avg.apply_(lambda x: x/7806)
print(avg)
torch.save(avg, os.path.join(EMBEDS, "avg_all.pt"))