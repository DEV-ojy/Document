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


## 모델 구현하기 

이제 모델을 설계합니다 이전 모델들과 달라진 점은 임베딩 층을 추가했다는 점입니다
파이토치에서는 nn.Embedding()을 사용해서 임베딩 층을 구현합니다 
임베딩층은 크게 두가지 인자를 받는데 첫번째는 단어장의 크기이며 두번째는 벡터의 차원입니다 '

```py
class Net(nn.Module):
    def __init__(self, vocab_size, input_size, hidden_size, batch_first=True):
        super(Net, self).__init__()
        self.embedding_layer = nn.Embedding(num_embeddings=vocab_size, # 워드 임베딩
                                            embedding_dim=input_size)
        self.rnn_layer = nn.RNN(input_size, hidden_size, # 입력 차원, 은닉 상태의 크기 정의
                                batch_first=batch_first)
        self.linear = nn.Linear(hidden_size, vocab_size) # 출력은 원-핫 벡터의 크기를 가져야함. 또는 단어 집합의 크기만큼 가져야함.

    def forward(self, x):
        # 1. 임베딩 층
        # 크기변화: (배치 크기, 시퀀스 길이) => (배치 크기, 시퀀스 길이, 임베딩 차원)
        output = self.embedding_layer(x)
        # 2. RNN 층
        # 크기변화: (배치 크기, 시퀀스 길이, 임베딩 차원)
        # => output (배치 크기, 시퀀스 길이, 은닉층 크기), hidden (1, 배치 크기, 은닉층 크기)
        output, hidden = self.rnn_layer(output)
        # 3. 최종 출력층
        # 크기변화: (배치 크기, 시퀀스 길이, 은닉층 크기) => (배치 크기, 시퀀스 길이, 단어장 크기)
        output = self.linear(output)
        # 4. view를 통해서 배치 차원 제거
        # 크기변화: (배치 크기, 시퀀스 길이, 단어장 크기) => (배치 크기*시퀀스 길이, 단어장 크기)
        return output.view(-1, output.size(2))v
```

이제 모델을 위해 하이퍼파라미터를 설정 

```py
# 하이퍼 파라미터
vocab_size = len(word2index)  # 단어장의 크기는 임베딩 층, 최종 출력층에 사용된다. <unk> 토큰을 크기에 포함한다.
input_size = 5  # 임베딩 된 차원의 크기 및 RNN 층 입력 차원의 크기
hidden_size = 20  # RNN의 은닉층 크기
```

모델을 생성합니다 

```py
# 모델 생성
model = Net(vocab_size, input_size, hidden_size, batch_first=True)
# 손실함수 정의
loss_function = nn.CrossEntropyLoss() # 소프트맥스 함수 포함이며 실제값은 원-핫 인코딩 안 해도 됨.
# 옵티마이저 정의
optimizer = optim.Adam(params=model.parameters())
```
