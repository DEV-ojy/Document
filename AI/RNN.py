import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

df = pd.read_csv('/kospi.csv')
df.head() # 데이터를 불러왔음

scaler = MinMaxScaler()
df[['Open','High','Low','Close','Volume']] = scaler.fit_transform(df[['Open','High','Low','Close','Volume']])
df.head()
#불러온 데이터변수(date제외)들을 최대최소 정규화를 진행한다

df.info()
#데이터프레임에 관한 정보를 알려준다

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'{device} is available')
#텐서 데이터를 만들기 전 gpu가 잘 활성화 되어 있는지 device 함수를 통해 확인

x = df[['Open','High','Low','Volume']].values
y = df['Close'].values
#데이터 셋을 target 기준으로 분리시켜주도록 한다

def seq_data(x,y,sequence_length):
  x_seq = []
  y_seq = []
  for i in range(len(x) - sequence_length):
    x_seq.append(x[i:i+sequence_length])
    y_seq.append(y[i+sequence_length])

    return torch.FloatTensor(x_seq).to(device),torch.FloatTensor(y_seq).to(device).view([-1,1])
# float형 tensor로 변형, gpu사용간으하게 .to(device)를 사용.   

split  =200
sequence_length = 5

x_seq,y_seq = seq_data(x,y,sequence_length)

x_train_seq = x_seq[:split]
y_train_seq = y_seq[:split]
x_test_seq = x_seq[split:]
y_test_seq = y_seq[split:]
print(x_train_seq.size(), y_train_seq.size())
print(x_test_seq.size(), y_test_seq.size())

train = torch.utils.data.TensorDataset(x_train_seq,y_train_seq)
test = torch.utils.data.TensorDataset(x_test_seq,y_test_seq)

batch_size = 20
train_loader =  torch.utils.data.DataLoader(dataset=train,batch_size=batch_size,shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test,batch_size=batch_size,shuffle=False)

input_size = x_seq.size(2)
num_layers =2 
hidden_size = 8

class VanillaRNN(nn.Module):
  def __init__(self,input_size,hidden_size,sequence_length,num_layers,device):
    super(VanillaRNN,self).__init__()
    self.device = device
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.rnn = nn.RNN(input_size,hidden_size,num_layers,batch_first=True)
    self.fc = nn.Sequential(nn.Linear(hidden_size * sequence_length,1),nn.Sigmoid())

  def forward(self,x):
    h0 = torch.zeros(self.num_layers,x.size()[0],self.hidden_size).to(self.device)
    #초기 hidden state 설정하기
    out,_=self.rnn(x,h0)
    #out:RNN의 마지막 레이어로부터 나온 output feature를 반환한다 hn:hidden state를 반환한다
    out = out.reshape(out.shape[0],-1) # many to many 전략
    out = self.fc(out)

    return out
model = VanillaRNN(input_size=input_size,
                   hidden_size=hidden_size,
                   sequence_length=sequence_length,
                   num_layers=num_layers,
                   device=device).to(device)   


criterion = nn.MSELoss()

lr = 1e-3
num_epochs = 200
optimizer = optim.Adam(model.parameters(), lr=lr)

loss_graph = [] #그래프 그릴 목적인 loss
n=len(train_loader)

for epoch in range(num_epochs):
  running_loss = 0.0

  for data in train_loader:
    seq, target = data #배치데이터
    out = model(seq) #모델에 넣고
    loss = criterion(out,target) #output 가지고 loss 구하고

    optimizer.zero_grad()
    loss.backward() #loss가 최소가 되게하는
    optimizer.step() #가중치 업데이트 해주고
    running_loss += loss.item()#한 배치의 loss 더해주고

  loss_graph.append(running_loss / n ) #한 epoch에 모든 배치들에 대한 평균 loss리스트에 담고

  if epoch % 100 ==0:
    print('[epoch:%d] loss: %.4f'%(epoch,running_loss/n))

plt.figure(figsize=(20,10))
plt.plot(loss_graph)
plt.show()


def plotting(train_loader,test_loader,actual):
  with torch.no_grad():
    train_pred = []
    test_pred = []

    for data in train_loader:
      seq, target = data
      out = model(seq)
      train_pred += out.cpu().numpy().tolist()

    for data in test_loader:
      seq, target = data
      out=model(seq)
      test_pred += out.cpu().numpy().tolist()

  total = train_pred + test_pred
  plt.figure(figsize=(20,10))
  plt.plot(np.ones(100)*len(train_pred),np.linspace(0,1,100),'--',linewidth=0.6)
  plt.plot(actual,'--')
  plt.plot(total,'b',linewidth=0.6)

  plt.legend(['train boundary','actual','prediction'])
  plt.show()

plotting(train_loader,test_loader,df['Close'][sequence_length:])
