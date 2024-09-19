import os
import numpy as np
import random
import torch
from tqdm import tqdm
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import math
from operator import itemgetter

EMBEDS = r"C:\Users\ASUS\Desktop\miniclef\embeds"
AVG = r"C:\Users\ASUS\Desktop\miniclef\avg_embeds"
CLASSES = os.listdir(EMBEDS)
#%%
print(len(CLASSES))

dataset = []

for i in CLASSES:
    for j in tqdm(os.listdir(os.path.join(EMBEDS, i))):
        embed = np.load(os.path.join(os.path.join(EMBEDS, i), j))
        one_hot = []
        for class_name in CLASSES:
            if i == class_name:
                one_hot.append(1)
            else:
                one_hot.append(0)
        
        dataset.append([torch.tensor(embed[0]), torch.FloatTensor(one_hot)])
        
print(len(dataset))
#%%
random.shuffle(dataset)

#%%
device = torch.device('cuda')
print(f"using GPU: {torch.cuda.is_available()}")
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(768, 1024)
        self.output = nn.Linear(1024, len(CLASSES))
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(1024)
    def forward(self, xb):
        xb = xb.reshape(-1, 768)
        x = self.relu(self.bn(self.input(xb)))
        x = self.output(x)
        return(x)
    
mlp = MLP()
mlp = mlp.to(device)
print("Model loaded")
#%%
train_len = round(0.6*len(dataset))
test_len = round(0.3*len(dataset))
val_len = round(0.1*len(dataset))

train, test, val = torch.utils.data.random_split(dataset, [train_len, test_len, val_len])
#%% Accuracy
def one_hot_to_label(x):
    labels = []
    for i in range(len(x)):
        if x[i] == 1:
            labels.append(CLASSES[i])
    return labels

def accuracy(outputs, labels):
    def cal_up(x):
        return 1 if x > 0.95 else x
    def cal_down(x):
        return 0 if x < 0.05 else x
    top_n = 0
    top_10 = 0
    tp_acc = 0
    for score in range(len(outputs)):
        cal_up_lambda = np.vectorize(cal_up)
        cal_down_lambda = np.vectorize(cal_down)
        
        raw_outputs = np.array(outputs[score]) #straight from model
        
        true_labels = one_hot_to_label(labels[score]) #convert true one-hot to labels
        
        corrected = cal_up_lambda(cal_down_lambda(raw_outputs)) #calibrate raw to one-hot
        
        # sort in descending order
        top_10_indexes = np.argpartition(raw_outputs, -10)[-10:]
        top_10_indexes_sorted = top_10_indexes[np.argsort(raw_outputs[top_10_indexes])[::-1]]
        #top_10_values = raw_outputs[top_10_indexes_sorted]
        
        #get top 10 highest predicted labels
        pred_labels = []
        for i in top_10_indexes_sorted:
            pred_labels.append(CLASSES [i])
        
        #calculate top_n (where n = number of true classes)
        correct_n = 0
        for l in true_labels:
            if l in pred_labels[:len(true_labels)]:
                correct_n += 1
       
        top_n += correct_n/len(true_labels)
        #calculate top_10 (where true labels are within the top 10 predicted classes
        correct_10 = 0
        for l in true_labels:
            if l in pred_labels[:10]:
                correct_10 += 1
        
        top_10 += correct_10/len(true_labels)
           
        #calculate tp/tp+tn
        true_pos = 0
        total = 0
        for k in range(len(corrected)):
            if(corrected[k] == 1):
                total += 1
                if(corrected[k] == labels[score][k]):
                    true_pos += 1
            
        if(total == 0):
            tp_acc += 0
        else:
            tp_acc += true_pos/total
    return((top_n/len(outputs)), (top_10/len(outputs)), (tp_acc/len(outputs)))   
        
def split_into_batch(lst, chunk_size):
    return list(zip(*[iter(lst)] * chunk_size))
      
writer = SummaryWriter()
        
def fit(epochs, batch_size, lr, model, opt_func = torch.optim.SGD):
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        train_loss_batch = 0
        train_acc_top_n_batch = 0
        train_acc_top_10_batch = 0
        train_acc_tp_acc_batch = 0
        
        val_loss_batch = 0
        val_acc_top_n_batch = 0
        val_acc_top_10_batch = 0
        val_acc_tp_acc_batch = 0
        
        train_batch = split_into_batch(train, batch_size)
        for batch in tqdm(train_batch):
            embeds = []
            labels = []
            for embed, label in batch:
                embeds.append(embed)
                labels.append(label)
            
            # Training Phase
            mlp.train()
            out = mlp.forward(torch.stack(embeds, dim=0).to(device))
            
            # reshape for acc calculations
            outputs = []
            for i in out.clone().detach().squeeze(1):
                outputs.append(torch.FloatTensor(i.cpu().numpy()))
            outputs = torch.stack(outputs, dim=0)
            
            # loss calc
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(out.squeeze(1), torch.stack(labels, dim=0).to(device))
            loss.backward() # backprop
            
            #optimize
            optimizer.step()
            optimizer.zero_grad()
            
            top_n, top_10, tp_acc = accuracy(outputs, labels) # acc calc
            train_acc_top_n_batch += top_n
            train_acc_top_10_batch += top_10
            train_acc_tp_acc_batch += tp_acc
            
            # record vals
            train_loss_batch += loss.item()
                
        
        test_batch = split_into_batch(test, batch_size)
        for batch in tqdm(test_batch):
            val_embeds = []
            val_labels = []
            for embed, label in batch:
                val_embeds.append(embed)
                val_labels.append(label)
            
            # Testing Phase
            mlp.eval()
            val_out = mlp.forward(torch.stack(val_embeds, dim=0).to(device))
            
            # reshape for acc calculations
            val_outputs = []
            for i in val_out.clone().detach().squeeze(1):
                val_outputs.append(torch.FloatTensor(i.cpu().numpy()))
            val_outputs = torch.stack(val_outputs, dim=0)
            
            val_loss = loss_fn(val_out.squeeze(1), torch.stack(val_labels, dim=0).to(device)) # loss calc
            
            val_top_n, val_top_10, val_tp_acc = accuracy(val_outputs, val_labels)# acc calc
            val_acc_top_n_batch += val_top_n
            val_acc_top_10_batch += val_top_10
            val_acc_tp_acc_batch += val_tp_acc
            
            # record vals
            val_loss_batch += val_loss.item()
            
        # display/record
        print(f"Epoch [{epoch}] of [{epochs}]| train_loss: {round((train_loss_batch/len(train_batch)),2)} top_n: {round((train_acc_top_n_batch/len(train_batch)),2)} top_10: {round((train_acc_top_10_batch/len(train_batch)),2)} tp_acc: {round((train_acc_tp_acc_batch/len(train_batch)),2)}")
        print(f"Epoch [{epoch}] of [{epochs}]| val_loss: {round((val_loss_batch/len(test_batch)),2)} top_n: {round((val_acc_top_n_batch/len(test_batch)),2)} top_10: {round((val_acc_top_10_batch/len(test_batch)),2)} tp_acc: {round((val_acc_tp_acc_batch/len(test_batch)),2)}")
        print("===============================================================================")
        writer.add_scalar('Loss/train', (train_loss_batch/len(train_batch)), epoch)
        writer.add_scalar('Loss/val', (val_loss_batch/len(test_batch)), epoch)
        writer.add_scalar('Top_n_Acc/train', (train_acc_top_n_batch/len(train_batch)), epoch)
        writer.add_scalar('Top_n_Acc/val', (val_acc_top_n_batch/len(test_batch)), epoch)
        writer.add_scalar('Top_10_Acc/train', (train_acc_top_10_batch/len(train_batch)), epoch)
        writer.add_scalar('Top_10_Acc/val', (val_acc_top_10_batch/len(test_batch)), epoch)
        writer.add_scalar('Tp_Acc/train', (train_acc_tp_acc_batch/len(train_batch)), epoch)
        writer.add_scalar('Tp_Acc/val', (val_acc_tp_acc_batch/len(test_batch)), epoch)
        writer.close()
fit(50, 64, 0.005, mlp)
#%%

unseen_embeds = []
unseen_labels = []

for embed, label in val:
    unseen_embeds.append(embed)
    unseen_labels.append(label)
    
mlp.eval()

unseen_out = mlp.forward(torch.stack(unseen_embeds, dim=0).to(device))

# reshape for acc calculations
unseen_outputs = []
for i in unseen_out.clone().detach().squeeze(1):
    unseen_outputs.append(torch.FloatTensor(i.cpu().numpy()))
unseen_outputs = torch.stack(unseen_outputs, dim=0)

unseen_acc = accuracy(unseen_outputs, unseen_labels)# acc calc
print(f"unseen accuracy: {unseen_acc}")
#%%
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def from_output_to_classes(x, top_n):
    class_name, L_sorted = zip(*sorted(enumerate(x), key=itemgetter(1), reverse=True))
    classes = []
    for i in class_name[:top_n]:
        classes.append(CLASSES[i])
    return classes, L_sorted[:top_n]  

def from_onehot_to_classes(x):
    classes = []
    for i in range(len(x)):
        if x[i] == 1:
            classes.append(CLASSES[i])
    return classes  

to_sigmoid = np.vectorize(sigmoid)
sample, label = val[random.randint(0, len(val)-1)]

inference =  mlp.forward(sample.to(device))
class_name, score = from_output_to_classes(to_sigmoid(inference.cpu().detach().numpy())[0], 5)
print(class_name)
print(score)
print(from_onehot_to_classes(label))
#%%

AB = r"C:\Users\ASUS\Desktop\miniclef\merged_embeds\AB"

merged_embeds = []

for i in os.listdir(AB):
    merged_embeds.append(np.load(os.path.join(AB, i)))

print(len(merged_embeds))

# predefined class A: 1355934
# predefined class B: 1356331

top_5_acc = 0
top_2_acc = 0
for sample in merged_embeds:
    inference =  mlp.forward(torch.tensor(sample[0]).to(device))
    class_name, score = from_output_to_classes(to_sigmoid(inference.cpu().detach().numpy())[0], 5)
    if ('1355934' in class_name) and ('1356331' in class_name):
        top_5_acc += 1
    class_name, score = from_output_to_classes(to_sigmoid(inference.cpu().detach().numpy())[0], 2)
    if ('1355934' in class_name) and ('1356331' in class_name):
        top_2_acc += 1
        
print(f"top_5 acc = {top_5_acc/100}")
print(f"top_2 acc = {top_2_acc/100}")

sample_1 = torch.tensor(merged_embeds[random.randint(0,100)][0])
inference =  mlp.forward(sample_1.to(device))
class_name, score = from_output_to_classes(to_sigmoid(inference.cpu().detach().numpy())[0], 5)
print(class_name)
print(score)

inference =  mlp.forward(sample_1.to(device))
class_name, score = from_output_to_classes(to_sigmoid(inference.cpu().detach().numpy())[0], 2)
print(class_name)
print(score)