# 문자 단위 RNN(Char RNN) - 더 많은 데이터

앞서 한 것보다 더 많은 데이터 문자 단위 RNN을 구현합니다 


## 1.문자 단위 RNN(Char RNN)

먼저 필요한 도구들을 임포트 합니다 

```py
import torch
import torch.nn as nn
import torch.optim as optim
```

다음과 같이 임의의 샘플을 만듭니다 

### 1. 훈련 데이터 전처리하기

```py
sentence = ("if you want to build a ship, don't drum up people together to "
            "collect wood and don't assign them tasks and work, but rather "
            "teach them to long for the endless immensity of the sea.")
```

문자 집합을 생성하고, 각 문자에 고유한 정수를 부여합니다 

```py
char_set = list(set(sentence)) # 중복을 제거한 문자 집합 생성
char_dic = {c: i for i, c in enumerate(char_set)} # 각 문자에 정수 인코딩

print(char_dic) # 공백도 여기서는 하나의 원소
```
```
{'k': 0, 'o': 1, 'r': 2, 'a': 3, 'f': 4, 'b': 5, 'g': 6, 'w': 7, ',': 8, ' ': 9, 'h': 10, 'l': 11, "'": 
12, 'e': 13, '.': 14, 'd': 15, 's': 16, 'y': 17, 'u': 18, 't': 19, 'n': 20, 'i': 21, 'm': 22, 'c': 23, 'p': 24}
```

각 문자에 정수가 부여됐다면 총 25개의 문자가 존재합니다 문자 집합의 크기를 확인해봅시다 
```py
dic_size = len(char_dic)
print('문자 집합의 크기 : {}'.format(dic_size))
```
```
문자 집합의 크기 : 25
```

문자 집합의 크기는 25이며 입력을 원-핫벡터로 사용할 것이므로 이는 매 시점마다 들어갈 입력의 크기이기도 합니다 이제 하이퍼파라미터를 설정
hidden_size(은닉 상태의 크기)를 입력의 크기와 동일하게 줬는데, 이는 사용자의 선택으로 다른 값을 줘도 무방

그리고 sequence_length라는 변수를 선언했는데 우리가 앞서 만든 샘플을 10개 단위로 끊어서 샘플을 만들예정이기 때문입니다 
```py
# 하이퍼파라미터 설정
hidden_size = dic_size
sequence_length = 10  # 임의 숫자 지정
learning_rate = 0.1
```

다음은 임의로 지정한 sequence_length 값인 10의 단위로 샘플들을 잘라서 데이터를 만드는 모습을 보여줍니다  
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
```
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

그럼 총 170개의 샘플이 생성되었습니다 그리고 각 샘플의 각 문자들은 고유한 정수로 인코딩이 된 상태입니다 첫번째 샘플의 입력 데이터와
레이블 데이터를 출력해 봅시다 

```py
print(x_data[0])
print(y_data[0])
```
```
[21, 4, 9, 17, 1, 18, 9, 7, 3, 20] # if you wan에 해당됨.
[4, 9, 17, 1, 18, 9, 7, 3, 20, 19] # f you want에 해당됨.
```

한칸씩  쉬프트된 시퀀스가 정상적으로 출력되는 것을 볼 수 있습니다 이제 입력 시퀀스에 대해서 원-핫인코딩을 수행하고
입력 데이터와 레이블 데이터를 텐서로 변환합니다 
```py
x_one_hot = [np.eye(dic_size)[x] for x in x_data] # x 데이터는 원-핫 인코딩
X = torch.FloatTensor(x_one_hot)
Y = torch.LongTensor(y_data)
```

이제 데이터들의 크기를 확인해봅시다 
```py
print('훈련 데이터의 크기 : {}'.format(X.shape))
print('레이블의 크기 : {}'.format(Y.shape))
```
```
훈련 데이터의 크기 : torch.Size([170, 10, 25])
레이블의 크기 : torch.Size([170, 10])
```

원-핫 인코딩 된 결과를 보기 위해서 첫번째 샘플만 출력해봅시다 
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

레이블 데이터의 첫번째 샘플도 출력해봅시다 
```py
print(Y[0])
```
```
tensor([ 1,  2,  5, 21, 14,  2, 16, 19,  9, 12])
```

위 레이블 시퀀스는 f you want에 해당됩니다. 이제 모델을 설계합니다
