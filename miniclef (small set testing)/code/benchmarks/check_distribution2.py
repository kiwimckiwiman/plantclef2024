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
        for j in i:
            probabilities.append(j[1])        
    
print(len(probabilities))
buckets = sort_into_buckets(probabilities)

y_axis = []
x_axis = []
for i, count in enumerate(buckets):
    y_axis.append((math.log(count + 1)))
    x_axis.append(f"{i * 1} to {(i + 1) * 1}")
print(len(x_axis))
print(len(y_axis))

fig = plt.figure(figsize = (150, 50))
 
# creating the bar plot
plt.bar(x_axis, y_axis, color ='purple', 
        width = 0.1)
plt.xlabel("Probabilities")
plt.ylabel("Occurences  (log(count))")

x_data = [i for i in range(0, 101)]

#plt.savefig(r"C:\Users\ASUS\Desktop\miniclef\code\benchmarks\results\vit_patch_image_top_50_bucket_1.png")
y_axis.reverse()
from scipy.optimize import curve_fit
# define the true objective function
def objective(x, a, b, c, d):
 return a * x + b * x**2 + c*x**3 + d
 print("======================================")
# curve fit
popt, _ = curve_fit(objective, x_data, y_axis)
# summarize the parameter values
a, b, c, d = popt
print('y = %.5f * x + %.5f * x^2 + %.5f * x^3 + %.5f' % (a, b, c, d))
# plot input vs output
# define a sequence of inputs between the smallest and largest known inputs
x_line = np.arange(min(x_data), max(x_data), 1)
# calculate the output for the range
y_line = objective(x_line, a, b, c, d)
# create a line plot for the mapping function
plt.plot(x_line, y_line, '--', color='red')
plt.show()
