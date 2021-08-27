# 문자 단위 RNN(Char RNN)
이번챕터에서는 모든 시점의 입력에 대해서 모든 시점에 대해서 출력을ㅇ 하는 다대다 RNN을 구현해보겠습니다
다대다 RNN은 대표적으로 품사 태깅, 개체명 인식등에서 사용됩니다

## 1. 문자 단위 RNN(Char RNN)

RNN의 입출력의 단위가 단어 레벨이 아니라 문자 레벨로 하여 RNN을 구현한다면, 이를 문자 단위 RNN이라고 합니다
RNN구조자체가 달라진 것은 아니고 입출력의 단위가 문자로 바뀌었을 뿐입니다 문자 단위 RNN을 다대다 구조로 구현해봅시다 

먼저 필요한 도구를 임포트 합니다 
```py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
```

### 1. 훈련 데이터 전처리하기

여기서는 문자 시퀀스 apple을 입력받으면 pple!를 출력하는 RNN을 구현해볼 겁니다 이렇게 구현하는 어떤 의미가 있지는 않습니다
그저 RNN의 동작을 이핵가기 위한 목적입니다 

입력 데이터와 레이블 데이터에 대해서 문자 집합을 만듭니다 여기서 문자 집합은 중복을 제거한 문자들의 집합입니다 
```py
input_str = 'apple'
label_str = 'pple!'
char_vocab = sorted(list(set(input_str+label_str)))
vocab_size = len(char_vocab)
print ('문자 집합의 크기 : {}'.format(vocab_size))
```
```
문자 집합의 크기 : 5
```

현재 문자 집합에는 총 5개의 문자가 있습니다 !,a,e,l,p입니다 이제 하이퍼파라미터를 정의해줍니다
이때 입력은 원-핫 벡터를 사용할 것이므로 입력의 크기는 문자 집합의 크기여야합니다 
```py
input_size = vocab_size # 입력의 크기는 문자 집합의 크기
hidden_size = 5
output_size = 5
learning_rate = 0.1
```

이제 문자 집합에 고유한 정수를 부여합니다
```py
char_to_index = dict((c, i) for i, c in enumerate(char_vocab)) # 문자에 고유한 정수 인덱스 부여
print(char_to_index)
```
```
{'!': 0, 'a': 1, 'e': 2, 'l': 3, 'p': 4}
```
!은 0, a는 1, e는 2, l은 3, p는 4가 부여되었습니다 나중에 예측 결과를 다시 문자 시퀀스로 보기위해서 반대로 정수로부터 문자를
얻을 수 있는 index_to_char을 만듭니다 
```py
index_to_char={}
for key, value in char_to_index.items():
    index_to_char[value] = key
print(index_to_char)
```
```
{0: '!', 1: 'a', 2: 'e', 3: 'l', 4: 'p'}
```
이제 입력 데이터와 레이블 데이터의 각 문자들을 정수로 맵핑합니다
```py
x_data = [char_to_index[c] for c in input_str]
y_data = [char_to_index[c] for c in label_str]
print(x_data)
print(y_data)
```
```
[1, 4, 4, 3, 2] # a, p, p, l, e에 해당됩니다.
[4, 4, 3, 2, 0] # p, p, l, e, !에 해당됩니다.
```
파이토치의 nn.RNN()은 기본적으로 3차원 텐서를 입력받습니다 그렇기 때문에 배치차원을 추가해줍니다
```py
# 배치 차원 추가
# 텐서 연산인 unsqueeze(0)를 통해 해결할 수도 있었음.
x_data = [x_data]
y_data = [y_data]
print(x_data)
print(y_data)
```
```
[[1, 4, 4, 3, 2]]
[[4, 4, 3, 2, 0]]
```
입력 시퀀스의 각 문자들을 원-핫 벡터로 바꿔줍니다
```py
x_one_hot = [np.eye(vocab_size)[x] for x in x_data]
print(x_one_hot)
```
```
[array([[0., 1., 0., 0., 0.],
       [0., 0., 0., 0., 1.],
       [0., 0., 0., 0., 1.],
       [0., 0., 0., 1., 0.],
       [0., 0., 1., 0., 0.]])]
```
이제 데이터와 레이블 데이터를 텐서로 바꿔줍니다
```py
X = torch.FloatTensor(x_one_hot)
Y = torch.LongTensor(y_data)

print('훈련 데이터의 크기 : {}'.format(X.shape))
print('레이블의 크기 : {}'.format(Y.shape))
```

이제 각 텐서의 크기를 확인해봅시다
```
훈련 데이터의 크기 : torch.Size([1, 5, 5])
레이블의 크기 : torch.Size([1, 5])
```
