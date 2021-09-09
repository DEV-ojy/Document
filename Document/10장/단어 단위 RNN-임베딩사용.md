# 단어 단위 RNN - 임베딩 사용

이번챕트에서는 문자 단위가 아닌 RNN의 입력 단위를 단어 단위로 사용합니다 그리고 단어 단위를 사용함에 따라서 
Pytorch에서 제공하는 임베딩 층를 사용하겠습니다 

## 훈련 데이터 전처리하기

우선 실습을 위한 도구들을 임포트합니다 

```py
import torch
import torch.nn as nn
import torch.optim as optim
```
실습을 위해 임의의 문장을 만듭니다 

```py
sentence = "Repeat is the best medicine for memory".split()
```
우리가 만들 RNN은 'Repeat is the best medicine for'을 입력받으면 'is the best medicine for memory'를 출력하는 RNN입니다
위의 임의의 문장으로부터 단어장을 만듭니다 
```py
vocab = list(set(sentence))
print(vocab)
```
```
['best', 'memory', 'the', 'is', 'for', 'medicine', 'Repeat']
```

이제 단어장의 단어에 고유한 정수 인덱스를 부여합니다 그리고 그와 동시에 모르는 단어를 의미하는 UNK토큰도 추가하겠습니다
```py
word2index = {tkn: i for i, tkn in enumerate(vocab, 1)}  # 단어에 고유한 정수 부여
word2index['<unk>']=0

print(word2index
```
```
{'best': 1, 'memory': 2, 'the': 3, 'is': 4, 'for': 5, 'medicine': 6, 'Repeat': 7, '<unk>': 0}
```

이제 word2index가 우리 사용할 최종 단어장인 셈입니다 word2index에 단어를 입력하면 맵핑되는 정수를 리턴합니다
```py
print(word2index['memory'])
```
```
2
```

단어 'memory'와 맵핑이 되는 정수는 2입니다 예측 단계에서 예측한 문장을 확인하기위해 idx2word도 만듭니다
```py
# 수치화된 데이터를 단어로 바꾸기 위한 사전
index2word = {v: k for k, v in word2index.items()}
print(index2word)
```
```
{1: 'best', 2: 'memory', 3: 'the', 4: 'is', 5: 'for', 6: 'medicine', 7: 'Repeat', 0: '<unk>'}
```
idx2word는 정수로부터 단어를 리턴하는 역할을 합니다 정수 2를 넣어봅시다 
```py
print(index2word[2])

```
```
memory
```

정수 2와 맵핑되는 단어는 memory인 것을 확인할 수 있습니다 이제 데이터의 각 단어를 정수로 인코딩하는 동시에 입력 데이터와
레이블 데이터를 만드는 build_data라는 함수를 만들어보겠습니다 
```py
def build_data(sentence, word2index):
    encoded = [word2index[token] for token in sentence] # 각 문자를 정수로 변환. 
    input_seq, label_seq = encoded[:-1], encoded[1:] # 입력 시퀀스와 레이블 시퀀스를 분리
    input_seq = torch.LongTensor(input_seq).unsqueeze(0) # 배치 차원 추가
    label_seq = torch.LongTensor(label_seq).unsqueeze(0) # 배치 차원 추가
    return input_seq, label_seq
```

만들어진 함수로부터 입력 데이터와 레이블 데이터를 얻습니다 
```py
X, Y = build_data(sentence, word2index)
```
입력 데이터와 레이블 데이터가 정상적으로 생성되었는지 출력해봅시다 
```py
print(X)
print(Y)
```
```
tensor([[7, 4, 3, 1, 6, 5]]) # Repeat is the best medicine for을 의미
tensor([[4, 3, 1, 6, 5, 2]]) # is the best medicine for memory을 의미
```
