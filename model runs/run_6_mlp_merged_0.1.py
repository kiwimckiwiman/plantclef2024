import os
import numpy as np
import random
import torch
from tqdm import tqdm
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

TRAIN = r"D:\plantclef2024\merged_embeds_all\train"
TEST = r"D:\plantclef2024\merged_embeds_all\test"
VAL = r"D:\plantclef2024\merged_embeds_all\val"
EMBEDS = r"D:\plantclef2024\embedsv2"
MODEL = r"C:\Users\User\plantclef\code\model\models\mlp_run_6_0.05_100_merged.pth"
CLASSES = os.listdir(EMBEDS)
#%%
print(len(CLASSES))
train_list = os.listdir(TRAIN)
test_list = os.listdir(TEST)

print(len(train_list))
print(len(test_list))
random.shuffle(train_list)
random.shuffle(test_list)

def get_train():
    selected = random.sample(train_list, 6400)
    random.shuffle(selected)
    return selected


def get_test():
    selected = random.sample(test_list, 3840)
    random.shuffle(selected)
    return selected
        
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
mlp.load_state_dict(torch.load(MODEL))
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
    optimizer = opt_func(model.parameters(), lr, weight_decay=0.01)
    for epoch in range(epochs):
        train_loss_batch = 0
        train_acc_top_n_batch = 0
        train_acc_top_10_batch = 0
        train_acc_tp_acc_batch = 0
        
        val_loss_batch = 0
        val_acc_top_n_batch = 0
        val_acc_top_10_batch = 0
        val_acc_tp_acc_batch = 0
        
        train_batch = split_into_batch(get_train(), batch_size)
        for batch in tqdm(train_batch):
            embeds = []
            labels = []
            for item in batch:
                try:
                    embeds.append(torch.tensor(np.load(os.path.join(TRAIN, item))[0]))
                    one_hot = []
                    for class_name in CLASSES:
                        item_labels = item.split(".")[0].split("_")[:-1]
                        if class_name in item_labels:
                            one_hot.append(1)
                        else:
                            one_hot.append(0)
                    labels.append(torch.FloatTensor(one_hot))
                except:
                    continue
            
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
            
            # record vals
            train_loss_batch += loss.item()
            train_acc_top_n_batch += top_n
            train_acc_top_10_batch += top_10
            train_acc_tp_acc_batch += tp_acc
            
                
        
        test_batch = split_into_batch(get_test(), batch_size)
        for batch in tqdm(test_batch):
            val_embeds = []
            val_labels = []
            for item in batch:
                try:
                    val_embeds.append(torch.tensor(np.load(os.path.join(TEST, item))[0]))
                    one_hot = []
                    for class_name in CLASSES:
                        item_labels = item.split(".")[0].split("_")[:-1]
                        if class_name in item_labels:
                            one_hot.append(1)
                        else:
                            one_hot.append(0)
                    val_labels.append(torch.FloatTensor(one_hot))
                    
                except:
                    continue
            
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
            
            # record vals
            val_loss_batch += val_loss.item()
            val_acc_top_n_batch += val_top_n
            val_acc_top_10_batch += val_top_10
            val_acc_tp_acc_batch += val_tp_acc
            
        
        # display/record
        print(f"Epoch [{epoch}] of [{epochs}] (continued after 100)| train_loss: {round((train_loss_batch/len(train_batch)),2)} top_n: {round((train_acc_top_n_batch/len(train_batch)),2)} top_10: {round((train_acc_top_10_batch/len(train_batch)),2)} tp_acc: {round((train_acc_tp_acc_batch/len(train_batch)),2)}")
        print(f"Epoch [{epoch}] of [{epochs}]  (continued after 100)| val_loss: {round((val_loss_batch/len(test_batch)),2)} top_n: {round((val_acc_top_n_batch/len(train_batch)),2)} top_10: {round((val_acc_top_10_batch/len(train_batch)),2)} tp_acc: {round((val_acc_tp_acc_batch/len(train_batch)),2)}")
        
        writer.add_scalar('Loss/train', (train_loss_batch/len(train_batch)), epoch)
        writer.add_scalar('Loss/val', (val_loss_batch/len(test_batch)), epoch)
        writer.add_scalar('Top_n_Acc/train', (train_acc_top_n_batch/len(train_batch)), epoch)
        writer.add_scalar('Top_n_Acc/val', (val_acc_top_n_batch/len(test_batch)), epoch)
        writer.add_scalar('Top_10_Acc/train', (train_acc_top_10_batch/len(train_batch)), epoch)
        writer.add_scalar('Top_10_Acc/val', (val_acc_top_10_batch/len(test_batch)), epoch)
        writer.add_scalar('Tp_Acc/train', (train_acc_tp_acc_batch/len(train_batch)), epoch)
        writer.add_scalar('Tp_Acc/val', (val_acc_tp_acc_batch/len(test_batch)), epoch)
        writer.close()
        
        #save
        if(epoch+1 == 100):
            PATH = r"C:\Users\User\plantclef\code\model\mlp_run_6_0.1_200_merged.pth"
            torch.save(mlp.state_dict(), PATH)
        if(epoch+1 == 150):
            PATH = r"C:\Users\User\plantclef\code\model\mlp_run_6_0.1_250_merged.pth"
            torch.save(mlp.state_dict(), PATH)
        if(epoch+1 == 200):
            PATH = r"C:\Users\User\plantclef\code\model\mlp_run_6_0.1_300_merged.pth"
            torch.save(mlp.state_dict(), PATH)
        if(epoch+1 == 250):
            PATH = r"C:\Users\User\plantclef\code\model\mlp_run_6_0.1_350_merged.pth"
            torch.save(mlp.state_dict(), PATH)
        if(epoch+1 == 300):
            PATH = r"C:\Users\User\plantclef\code\model\mlp_run_6_0.1_400_merged.pth"
            torch.save(mlp.state_dict(), PATH)
        if(epoch+1 == 350):
            PATH = r"C:\Users\User\plantclef\code\model\mlp_run_6_0.1_450_merged.pth"
            torch.save(mlp.state_dict(), PATH)
        if(epoch+1 == 400):
            PATH = r"C:\Users\User\plantclef\code\model\mlp_run_6_0.1_500_merged.pth"
            torch.save(mlp.state_dict(), PATH)
fit(400, 128, 0.1, mlp)