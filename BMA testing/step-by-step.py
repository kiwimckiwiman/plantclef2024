import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from ast import literal_eval
import math
import seaborn as sns
import csv
import scipy.stats as stats

folder = r"C:\Users\ASUS\Desktop\raw"

def conv_var(x):
    return abs(math.log(x, 10))

def confidence(x, a):
    a = a + 0.5
    return (math.sqrt((a-x)/a))

def variance(data):
     n = len(data)
     mean = sum(data) / n
     deviations = [(x - mean) ** 2 for x in data]
     var = sum(deviations) / n
     return var

def softmax(scores):
    softmax_scores = []
    denom = 0
    for i in scores:
        denom += math.exp(i)
    for i in scores:
        softmax_scores.append((math.exp(i)/denom)*100)
    return softmax_scores
 #%%
for file in os.listdir(folder):
    probabilities = []
    df = pd.read_csv(os.path.join(folder, file), delimiter=";")
    for index, row in df.iterrows():
        preds = literal_eval(row['raw_preds'])
        p = []
        count = 1
        for i in preds:
            p.append(i)
            if count == 1000:
                break
            else:
                count += 1
        probabilities.append(p)
        
    #%%
    all_classes = []
    variances = []
    for i in probabilities:
        score = []
        for pair in i:
            if pair[0] not in all_classes:
                all_classes.append(pair[0])
            score.append(pair[1])
        variances.append(variance(score))
        
    conv_vars = [conv_var(i) for i in variances]
    conf = [confidence(i, pd.Series(conv_vars).describe()[-1]) for i in conv_vars]
    total_conf = np.sum(np.array(conf))
    
    #%%
    for i in range(len(probabilities)):
        score = []
        for pair in probabilities[i]:
            pair[1] = pair[1] * conf[i] / total_conf
    #%%
    BMA = []
    
    for c in all_classes:
        total = 0
        for p in probabilities:
            for pair in p:
                if pair[0] == c:
                    total += pair[1]
        BMA.append([c, total])
    
    #%%
    BMA_score = []
    BMA_label = []
    for i in BMA: #split label and score for sorting
        BMA_score.append(i[1])
        BMA_label.append(i[0])
    
    BMA_score = np.array(BMA_score)
    BMA_label = np.array(BMA_label)
    
    BMA_indices = np.argsort(BMA_score)[::-1] #sort scores
    sorted_BMA_score = BMA_score[BMA_indices] #keep indices
    
    sorted_BMA_label = [BMA_label[i] for i in BMA_indices] #get labels based on score sorted indices
    
    
    # creating the bar plot
    plt.bar(sorted_BMA_label[:10], sorted_BMA_score[:10], color ='purple', 
            width = 0.5)
    plt.title(file)
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("Classes")
    plt.ylabel("Probability")
    plt.show()
    
    
    z_score = stats.zscore(sorted_BMA_score[:100]) #z_score computation for threshold
    
    
    # creating the bar plot
    plt.bar(sorted_BMA_label[:10], abs(z_score[:10]), color ='purple', width = 0.5)
    plt.title(file)
    plt.axhline(2, color='r') # horizontal
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("Classes")
    plt.ylabel("z-score")
    plt.show()
    fil_lab = []
    for j in range(len(z_score)):
        if(abs(z_score[j]) >= 2): #anything above a standard deviation of 2 (hyper parameter)
            fil_lab.append([int(sorted_BMA_label[j]), sorted_BMA_score[j]])
            
