# 문자 단위 RNN(Char RNN) - 더 많은 데이터

## 문자 단위 RNN(Char RNN)

필요한 도구 임포트 

```py
import torch
import torch.nn as nn
import torch.optim as optim
```

### 1.훈련데이터 전처리하기 
```py
sentence = ("if you want to build a ship, don't drum up people together to "
            "collect wood and don't assign them tasks and work, but rather "
            "teach them to long for the endless immensity of the sea.")
```

```py
char_set = list(set(sentence)) # 중복을 제거한 문자 집합 생성
char_dic = {c: i for i, c in enumerate(char_set)} # 각 문자에 정수 인코딩

print(char_dic) # 공백도 여기서는 하나의 원소
```

```
{'k': 0, 'o': 1, 'r': 2, 'a': 3, 'f': 4, 'b': 5, 'g': 6, 'w': 7, ',': 8, ' ': 9, 'h': 10, 'l': 11,
"'": 12, 'e': 13, '.': 14, 'd': 15, 's': 16, 'y': 17, 'u': 18, 't': 19, 'n': 20, 'i': 21, 'm': 22, 'c': 23, 'p': 24}
```

```py
dic_size = len(char_dic)
print('문자 집합의 크기 : {}'.format(dic_size))
```

```
문자 집합의 크기 : 25
```

```py
# 하이퍼파라미터 설정
hidden_size = dic_size
sequence_length = 10  # 임의 숫자 지정
learning_rate = 0.1
```

```py
# 데이터 구성
x_data = []
y_data = []

for i in range(0, len(sentence) - sequence_length):
    x_str = sentence[i:i + sequence_length]
    y_str = sentence[i + 1: i + sequence_length + 1]
    print(i, x_str, '->', y_str)

    x_data.append([char_dic[c] for c in x_str])  # x str to index
    y_data.append([char_dic[c] for c in y_str])  # y str to index
```
```py
0 if you wan -> f you want
1 f you want ->  you want 
2  you want  -> you want t
3 you want t -> ou want to
4 ou want to -> u want to 
... 중략 ...
165 ity of the -> ty of the 
166 ty of the  -> y of the s
167 y of the s ->  of the se
168  of the se -> of the sea
169 of the sea -> f the sea.  
```

```py
print(x_data[0])
print(y_data[0])
```

```
[21, 4, 9, 17, 1, 18, 9, 7, 3, 20] # if you wan에 해당됨.
[4, 9, 17, 1, 18, 9, 7, 3, 20, 19] # f you want에 해당됨.
```

```py
x_one_hot = [np.eye(dic_size)[x] for x in x_data] # x 데이터는 원-핫 인코딩
X = torch.FloatTensor(x_one_hot)
Y = torch.LongTensor(y_data)

print('훈련 데이터의 크기 : {}'.format(X.shape))
print('레이블의 크기 : {}'.format(Y.shape))
```

```
훈련 데이터의 크기 : torch.Size([170, 10, 25])
레이블의 크기 : torch.Size([170, 10])
```

```py
print(X[0])
```
```
tensor([[0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], # i
        [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], # f
        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], # 공백
        [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], # y
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.], # o
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], # y
        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], # 공백
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.], # w
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.], # a
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]) # n
```

### 2.모델 구현하기
```py
class Net(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, layers): # 현재 hidden_size는 dic_size와 같음.
        super(Net, self).__init__()
        self.rnn = torch.nn.RNN(input_dim, hidden_dim, num_layers=layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x):
        x, _status = self.rnn(x)
        x = self.fc(x)
        return x
```
```py
net = Net(dic_size, hidden_size, 2) # 이번에는 층을 두 개 쌓습니다.
```
```py
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), learning_rate)

outputs = net(X)
print(outputs.shape) # 3차원 텐서

print(outputs.view(-1, dic_size).shape) # 2차원 텐서로 변환.
```
```
torch.Size([170, 10, 25])

torch.Size([1700, 25])
```
```py
print(Y.shape)
print(Y.view(-1).shape)
```
```
torch.Size([170, 10])
torch.Size([1700])
```
```py
for i in range(100):
    optimizer.zero_grad()
    outputs = net(X) # (170, 10, 25) 크기를 가진 텐서를 매 에포크마다 모델의 입력으로 사용
    loss = criterion(outputs.view(-1, dic_size), Y.view(-1))
    loss.backward()
    optimizer.step()

    # results의 텐서 크기는 (170, 10)
    results = outputs.argmax(dim=2)
    predict_str = ""
    for j, result in enumerate(results):
        if j == 0: # 처음에는 예측 결과를 전부 가져오지만
            predict_str += ''.join([char_set[t] for t in result])
        else: # 그 다음에는 마지막 글자만 반복 추가
            predict_str += char_set[result[-1]]

    print(predict_str)
```
```
hahhahrrhhhahaahahhhhahhahhhhhhhhhahhahhhhhhhahrahhahhhahhhhaahhhrhahhahahhahhhhhhhhaahhhhhhh
ahhhhahhhhahhhrhhhhhhahhhahahhhhaahahhahhhhaahahhahahhhahhhhhhahhahahhhhhhhahhhhahhhaa
... 중략 ...
p you want to build a ship, don't drum up people together to collect wood and don't assign them tasks and work,
but rather teach them to long for the endless immensity of the sea
```
