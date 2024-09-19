import os
import numpy as np
import random
import torch
from tqdm import tqdm
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import math
from operator import itemgetter
import scipy.stats as stats

EMBEDS = r"D:\plantclef2024\embedsv2"
TEST = r"D:\plantclef2024\embeds_test"
AVG = r"D:\plantclef2024\avg_all.pt"
MODEL = r"C:\Users\User\plantclef\code\model\models\mlp_run_8_0.07_sig_100_single_class.pth"
CLASSES = os.listdir(EMBEDS)
TEST_CLASSES = os.listdir(TEST)
#%%
print(len(CLASSES))

def get_list(x, i):
    embeds = []
    for j in (os.listdir(os.path.join(x, i))):
        if "average" in j or ".png" in j:
            continue
        else:
            embeds.append(j)
    return embeds

def get_train():
    dataset = []
    for i in tqdm(CLASSES):
        emb = get_list(EMBEDS, i)
        selected = random.sample(emb, 1)
        embed = np.load(os.path.join(os.path.join(EMBEDS, i), selected[0]))
        one_hot = []
        for class_name in CLASSES:
            if i == class_name:
                one_hot.append(1)
            else:
                one_hot.append(0)
        dataset.append([torch.tensor(embed[0]), torch.FloatTensor(one_hot)])
    random.shuffle(dataset) 
    return dataset

def get_test():
    dataset = []
    for i in tqdm(TEST_CLASSES):
        emb = get_list(TEST, i)
        selected = random.sample(emb, 1)
        embed = np.load(os.path.join(os.path.join(TEST, i), selected[0]))
        one_hot = []
        for class_name in CLASSES:
            if i == class_name:
                one_hot.append(1)
            else:
                one_hot.append(0)
        dataset.append([torch.tensor(embed[0]), torch.FloatTensor(one_hot)])
    random.shuffle(dataset) 
    return dataset

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
        self.sig = nn.Sigmoid()
    def forward(self, xb):
        xb = xb.reshape(-1, 768)
        x = self.relu(self.bn(self.input(xb)))
        x = self.sig(self.output(x))
        return(x)
    
mlp = MLP()
mlp = mlp.to(device)
print("Model loaded")

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
    def softmax(scores):
        softmax_scores = []
        denom = 0
        for i in scores:
            denom += math.exp(i.item())
        for i in scores:
            softmax_scores.append((math.exp(i.item())/denom)*100)
        return softmax_scores
    
    def load_class_mapping(class_list_file):
        with open(class_list_file) as f:
            class_index_to_class_name = {i: line.strip() for i, line in enumerate(f)}
        return class_index_to_class_name
    CLASSES_LIST = r"C:\Users\User\plantclef\code\model\pretrained_models\class_mapping.txt"
    cid_to_spid = load_class_mapping(CLASSES_LIST)
    
    top_n = 0
    top_10 = 0
    tp_acc = 0
    top_test = 0
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
        
        #calculate how test would be
        
        sorted_indexes = np.argsort(raw_outputs)[::-1]
        sorted_array = raw_outputs[sorted_indexes]
        softmax_arr = softmax(sorted_array)
        
        z_score = stats.zscore(softmax_arr)
        fil_lab = []
        for j in range(len(z_score)):
            if(abs(z_score[j]) >= 2):
                fil_lab.append([cid_to_spid[sorted_indexes[i]]])
        
        
        correct_test = 0
        for l in true_labels:
            if l in fil_lab:
                correct_test += 1
       
        top_test += correct_test/len(true_labels)
    return((top_n/len(outputs)), (top_10/len(outputs)), (top_test/len(outputs)))   
        
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
        
        train_acc_count = 0
        test_acc_count = 0
        
        count = 0
        train_batch = split_into_batch(get_train(), batch_size)
        for batch in tqdm(train_batch):
            count += 1
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
            
            if(count%10 == 0):
                train_acc_count += 1
                top_n, top_10, tp_acc = accuracy(outputs, labels) # acc calc
                train_acc_top_n_batch += top_n
                train_acc_top_10_batch += top_10
                train_acc_tp_acc_batch += tp_acc
            
            # record vals
            train_loss_batch += loss.item()
                
        
        count = 0
        test_batch = split_into_batch(get_test(), batch_size)
        for batch in tqdm(test_batch):
            count += 1
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
            
            if(count%10 == 0):
                test_acc_count += 1
                val_top_n, val_top_10, val_tp_acc = accuracy(val_outputs, val_labels)# acc calc
                val_acc_top_n_batch += val_top_n
                val_acc_top_10_batch += val_top_10
                val_acc_tp_acc_batch += val_tp_acc
            
            # record vals
            val_loss_batch += val_loss.item()
            
        # display/record
        print(f"Epoch [{epoch}] of [{epochs}]| train_loss: {round((train_loss_batch/len(train_batch)),2)} top_n: {round((train_acc_top_n_batch/train_acc_count),2)} top_10: {round((train_acc_top_10_batch/train_acc_count),2)} tp_acc: {round((train_acc_tp_acc_batch/train_acc_count),2)}")
        print(f"Epoch [{epoch}] of [{epochs}]| val_loss: {round((val_loss_batch/len(test_batch)),2)} top_n: {round((val_acc_top_n_batch/test_acc_count),2)} top_10: {round((val_acc_top_10_batch/test_acc_count),2)} tp_acc: {round((val_acc_tp_acc_batch/test_acc_count),2)}")
        print("===============================================================================")
        writer.add_scalar('Loss/train', (train_loss_batch/len(train_batch)), epoch)
        writer.add_scalar('Loss/val', (val_loss_batch/len(test_batch)), epoch)
        writer.add_scalar('Top_n_Acc/train', (train_acc_top_n_batch/train_acc_count), epoch)
        writer.add_scalar('Top_n_Acc/val', (val_acc_top_n_batch/test_acc_count), epoch)
        writer.add_scalar('Top_10_Acc/train', (train_acc_top_10_batch/train_acc_count), epoch)
        writer.add_scalar('Top_10_Acc/val', (val_acc_top_10_batch/test_acc_count), epoch)
        writer.add_scalar('Tp_Acc/train', (train_acc_tp_acc_batch/train_acc_count), epoch)
        writer.add_scalar('Tp_Acc/val', (val_acc_tp_acc_batch/test_acc_count), epoch)
        writer.close()
        
        #save
        if(epoch+1 == 100):
            PATH = r"C:\Users\User\plantclef\code\model\mlp_run_8_0.07_sig_100.pth"
            torch.save(mlp.state_dict(), PATH)
        if(epoch+1 == 150):
            PATH = r"C:\Users\User\plantclef\code\model\mlp_run_8_0.07_sig_150.pth"
            torch.save(mlp.state_dict(), PATH)
        if(epoch+1 == 200):
            PATH = r"C:\Users\User\plantclef\code\model\mlp_run_8_0.07_sig_200.pth"
            torch.save(mlp.state_dict(), PATH)
        if(epoch+1 == 250):
            PATH = r"C:\Users\User\plantclef\code\model\mlp_run_8_0.07_sig_250.pth"
            torch.save(mlp.state_dict(), PATH)
        if(epoch+1 == 300):
            PATH = r"C:\Users\User\plantclef\code\model\mlp_run_8_0.07_sig_300.pth"
            torch.save(mlp.state_dict(), PATH)
        if(epoch+1 == 350):
            PATH = r"C:\Users\User\plantclef\code\model\mlp_run_8_0.07_sig_350.pth"
            torch.save(mlp.state_dict(), PATH)
        if(epoch+1 == 400):
            PATH = r"C:\Users\User\plantclef\code\model\mlp_run_8_0.07_sig_400.pth"
            torch.save(mlp.state_dict(), PATH)
fit(400, 128, 0.07, mlp)
