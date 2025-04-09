import torch
import torch.nn as nn
import torch.utils.data as Data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import os

data_path=os.getcwd()

input_size = 2
output_size = 1
num_layers = 1
hidden_size = 64
batch_size = 64
train_step = 20000
h_state = None
use_gpu = torch.cuda.is_available()
print("GPU:", use_gpu)

data=np.loadtxt(data_path+r"/data/Q.txt")
data=data[:,0:2]

scale=StandardScaler()
data=scale.fit_transform(data)

listt=np.loadtxt(data_path+r"/data/list.txt")

data_x=[]
number=0
for i in range(listt.shape[0]):
    a = int(listt[i])
    b = torch.from_numpy(data[number:number+a,:].astype(np.float32))
    data_x.append(b)
    number+=a

data_target_temp=np.loadtxt(data_path+r"/data/Q.txt")
data_runoff=data_target_temp[:,2]

data_y=[]
number=0
for i in range(listt.shape[0]):
    aa = int(listt[i])
    bb = torch.from_numpy(data_runoff[number:number+aa].astype(np.float32))
    data_y.append(bb)
    number+=aa

train_x=[]
train_y=[]
test_x=[]
test_y=[]
for i in range(len(data_x)):
    if (i+1)%4.0==0:
        test_x.append(data_x[i])
        test_y.append(data_y[i])
    else:
        train_x.append(data_x[i])
        train_y.append(data_y[i])

class MyData(Data.Dataset):
    def __init__(self, train_x, train_y):
        self.train_x = train_x
        self.train_y = train_y

    def __len__(self):
        return len(self.train_x)

    def __getitem__(self, item):
        return self.train_x[item], self.train_y[item]

train_data = MyData(train_x, train_y)

def collate_fn(train_data):
    train_data.sort(key=lambda data: len(data[0]), reverse=True)
    train_x = []
    train_y = []
    for data in train_data:
        train_x.append(data[0])
        train_y.append(data[1])
    data_length = [len(data) for data in train_x]
    train_x = torch.nn.utils.rnn.pad_sequence(train_x, batch_first=True, padding_value=0)
    train_y = torch.nn.utils.rnn.pad_sequence(train_y, batch_first=True, padding_value=0)
    return train_x, train_y, data_length

train_loader=Data.DataLoader(train_data, shuffle=True, batch_size=batch_size, collate_fn=collate_fn)

class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size, bias=True)

    def forward(self, x, h0: torch.Tensor, c0: torch.Tensor):
        lstm_out, (h_s, h_c) = self.lstm(x, (h0, c0))
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        out = self.out(out)
        return out

lstm = LSTM(input_size, hidden_size, num_layers, output_size)
optimizer = torch.optim.RMSprop(lstm.parameters(), lr=0.001)
loss_func = nn.MSELoss()
train_loss_all = []

for epoch in range(train_step):
    train_loss=0
    train_num=0
    for (bx, by, length) in train_loader:
        bx = torch.nn.utils.rnn.pack_padded_sequence(bx, length, batch_first=True)
        h0 = torch.zeros(num_layers, by.shape[0], hidden_size)
        c0 = torch.zeros(num_layers, by.shape[0], hidden_size)
        out=lstm.forward(bx,h0,c0)
        loss=loss_func(out[:,:,0],by[:,:])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss+=loss.item()*by.size(0)
        train_num+=by.size(0)
    train_loss_all.append(train_loss/train_num)
    print(epoch)
    print(train_loss/train_num)

plt.figure(figsize = (8, 6))
plt.plot(train_loss_all, 'ro-', label = 'Train loss')
plt.legend()
plt.grid()
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.show()

torch.save(lstm.state_dict(),data_path+r"/data/lstm_parameters.pkl")

new_lstm = LSTM(input_size, hidden_size, num_layers, output_size)
new_lstm.load_state_dict(torch.load(data_path+r"/data/lstm_parameters.pkl"))

case_number = 0
data_number = 0
MSE_train = 0.0
output_all_name = data_path + r"/data/result_train.txt"
output_all = open(output_all_name, 'w')
peak_all_name = data_path + r"/data/peak_train.txt"
peak_all = open(peak_all_name, 'w')
for i in range(len(train_x)):
    case_number += 1
    output_name=data_path+r"/data/result_train"+str(i)+".txt"
    output = open(output_name, 'w')
    h0 = torch.zeros(num_layers, 1, hidden_size)
    c0 = torch.zeros(num_layers, 1, hidden_size)
    a = train_x[i].unsqueeze(0)
    b = torch.nn.utils.rnn.pack_padded_sequence(a, [train_x[i].shape[0]], batch_first=True)
    c = new_lstm(b, h0, c0)

    simulation = c[0,:,0].detach().numpy()
    observation = train_y[i].numpy()
    mean_observation = np.mean(observation)
    numerator = np.sum((observation - simulation) ** 2)
    denominator = np.sum((observation - mean_observation) ** 2)
    if denominator>0:
        nse = 1 - (numerator / denominator)
    else:
        nse = -9999
    print(np.max(simulation), np.max(observation), nse, file=peak_all)
    for k in range(c.shape[1]):
        data_number+=1
        MSE_train += (c[0, k, 0].item() - train_y[i][k].item()) ** 2
        print(k,c[0,k,0].item(),train_y[i][k].item(),file=output)
        print(data_number, c[0, k, 0].item(), train_y[i][k].item(), file=output_all)
    output.close()
output_all.close()
peak_all.close()
print("Number of train set is:",case_number)
print("MSE of train is:",MSE_train/data_number)

case_number = 0
data_number = 0
MSE_test = 0.0
output_all_name = data_path + r"/data/result_test.txt"
output_all = open(output_all_name, 'w')
peak_all_name = data_path + r"/data/peak_test.txt"
peak_all = open(peak_all_name, 'w')
for i in range(len(test_x)):
    case_number += 1
    output_name=data_path+r"/data/result_test"+str(i)+".txt"
    output = open(output_name, 'w')
    h0 = torch.zeros(num_layers, 1, hidden_size)
    c0 = torch.zeros(num_layers, 1, hidden_size)
    a = test_x[i].unsqueeze(0)
    b = torch.nn.utils.rnn.pack_padded_sequence(a, [test_x[i].shape[0]], batch_first=True)
    c = new_lstm(b, h0, c0)

    simulation = c[0,:,0].detach().numpy()
    observation = test_y[i].numpy()
    mean_observation = np.mean(observation)
    numerator = np.sum((observation - simulation) ** 2)
    denominator = np.sum((observation - mean_observation) ** 2)
    if denominator>0:
        nse = 1 - (numerator / denominator)
    else:
        nse = -9999
    print(np.max(simulation), np.max(observation), nse, file=peak_all)
    for k in range(c.shape[1]):
        data_number += 1
        MSE_test += (c[0,k,0].item()-test_y[i][k].item())**2
        print(k,c[0,k,0].item(),test_y[i][k].item(),file=output)
        print(data_number, c[0, k, 0].item(), test_y[i][k].item(), file=output_all)
    output.close()
output_all.close()
peak_all.close()
print("Number of test set is:",case_number)
print("MSE of test is:",MSE_test/data_number)