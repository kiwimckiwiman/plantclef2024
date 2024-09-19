import os
import numpy as np
from tqdm import tqdm
from numpy import dot
from numpy.linalg import norm

# predefined class A: 1355934
# predefined class B: 1356331
AVG = r"C:\Users\ASUS\Desktop\miniclef\avg_embeds"

averages = []
classes = []
AB = []

for i in tqdm(os.listdir(AVG)):
    if i == "AB":
        AB = np.load(os.path.join(os.path.join(AVG, i),i+"_average.npy"))
    else:
        averages.append(np.load(os.path.join(os.path.join(AVG, i),i+"_average.npy")))
        classes.append(i)
     
cosine_score = {}
euclidean_score = {}

for j in range(len(averages)):
    cos_sim = dot(AB[0], averages[j][0])/(norm(AB[0])*norm(averages[j][0]))
    cosine_score[classes[j]] = cos_sim
    euclidean_score[classes[j]] = norm(AB[0]-averages[j][0])

print(sorted(cosine_score.items(), key=lambda x:x[1], reverse=True)[:5])
print(f"cosine score for class A (1355934) vs AB {cosine_score['1355934']}")
print(f"cosine score for class B (1356331) vs AB {cosine_score['1356331']}")
print(f"cosine score for class 1364133 vs AB {cosine_score['1364133']}")
print("========================================================")
print(sorted(euclidean_score.items(), key=lambda x:x[1])[:5])
print(f"euclidean score for class A (1355934) vs AB {euclidean_score['1355934']}")
print(f"euclidean score for class B (1356331) vs AB {euclidean_score['1356331']}")
print(f"euclidean score for class 1364133 vs AB {euclidean_score['1364133']}")

