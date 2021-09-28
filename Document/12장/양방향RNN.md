# 양방향 RNN을 이용한 품사 태깅

이번 챕터에서는 파이토치를 사용하여 시퀀스 레이블링의 대표적인 태스크인 품사 태깅 작업을 구현해보겠습니다 

## 1.셋팅하기 

필요한 도구를 임포트 

```py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchtext import data
from torchtext import datasets
import time
import random
```
랜덤시드를 고정 

```py
SEED = 1234
random.seed(SEED)
torch.manual_seed(SEED)
```

cpu 또는 gpu를 사용할 것인지를 확인해줍니다
```py
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

## 2.훈련 데이터에 대한 이해 
토치 텍스트를 사용합니다 

### 1.필드 정의하기 

이번에 사용할 데이터용 총 3개의 열,즉 3개의 필드를 가지고 있습니다 레이블이 총 2개이기 때문인데 이 중 1개만 사용할 것이만
원활하게 데이터를 불러오기 위해서 일단은 3개의 모두 필드를 정의해줍니다 

```py
# 3개의 필드 정의
TEXT = data.Field(lower = True)
UD_TAGS = data.Field(unk_token = None)
PTB_TAGS = data.Field(unk_token = None)

fields = (("text", TEXT), ("udtags", UD_TAGS), ("ptbtags", PTB_TAGS))
```

### 2.데이터셋 만들기

이제 토치텍스트에서 제공하는 훈련 데이터를 불러오는 동시에 데이터셋을 만들어보겠습니다 훈련 데이터, 검증 데이터,
테스트 데이터를 각각 나눠서 저장해줍니다

```py
train_data, valid_data, test_data = datasets.UDPOS.splits(fields)

downloading en-ud-v2.zip
en-ud-v2.zip: 100%|██████████| 688k/688k [00:00<00:00, 4.85MB/s]
extracting
```

훈련 데이터,검증 데이터,테스트 데이터의 크기를 확인해봅시다 
```py
print(f"훈련 샘플의 개수 : {len(train_data)}")
print(f"검증 샘플의 개수 : {len(valid_data)}")
print(f"테스트 샘플의 개수 : {len(test_data)}")
```
```
훈련 샘플의 개수 : 12543
검증 샘플의 개수 : 2002
테스트 샘플의 개수 : 2077
```
데이터셋을 생성하였으니 훈련 데이터의 필드들을 출력해봅시다 
```py
# 훈련 데이터의 3개의 필드 확인
print(train_data.fields)
```
```
{'text': <torchtext.data.field.Field object at 0x7f5c28627828>, 
'udtags': <torchtext.data.field.Field object at 0x7f5c28627860>, 'ptbtags':
<torchtext.data.field.Field object at 0x7f5c286278d0>}
```
훈련 데이터의 첫번째 샘플에서 text와 두개의 레이블을 모두 출력해보겠습니다 
```py
# 첫번째 훈련 샘플의 text 필드
print(vars(train_data.examples[0])['text'])
```
```
['al', '-', 'zaman', ':', 'american', 'forces', 'killed', 'shaikh', 'abdullah', 'al', '-', 'ani', ','
, 'the', 'preacher', 'at', 'the', 'mosque', 'in', 'the', 'town', 'of', 'qaim', ',', 'near', 'the', 
'syrian','border', '.']
```

첫번째 레이블은 udtags입니다 우리가 사용할 레이블입니다 

```py
# 첫번째 훈련 샘플의 udtags 필드
print(vars(train_data.examples[0])['udtags'])
```
```
['PROPN', 'PUNCT', 'PROPN', 'PUNCT', 'ADJ', 'NOUN', 'VERB', 'PROPN', 'PROPN', 'PROPN', 
'PUNCT', 'PROPN','PUNCT', 'DET', 'NOUN', 'ADP', 'DET', 'NOUN', 'ADP', 'DET', 'NOUN', 
'ADP', 'PROPN', 'PUNCT', 'ADP', 'DET','ADJ', 'NOUN', 'PUNCT']
```
두번째 레이블은 ptbdtags입니다 여기서는 사용하지 않을 레이블입니다 
```py
# 첫번째 훈련 샘플의 ptbdtags 필드
print(vars(train_data.examples[0])['ptbtags'])
```
```
['NNP', 'HYPH', 'NNP', ':', 'JJ', 'NNS', 'VBD', 'NNP', 'NNP', 'NNP', 'HYPH', 'NNP', 
',', 'DT', 'NN', 'IN', 'DT', 'NN', 'IN', 'DT', 'NN', 'IN', 'NNP', ',', 'IN', 'DT', 'JJ', 'NN', '.']
```

### 3.단어 집합만들기 

이제 단어 집합을 생성해보겠습니다 그리고 단어 집합을 생성 시에 사전 훈련된 워드 임베딩인 GloVe를 사용해보겠습니다
```py
# 최소 허용 빈도
MIN_FREQ = 5

# 사전 훈련된 워드 임베딩 GloVe 다운로드
TEXT.build_vocab(train_data, min_freq = MIN_FREQ, vectors = "glove.6B.100d")
UD_TAGS.build_vocab(train_data)
PTB_TAGS.build_vocab(train_data)
```
상위 빈도수 20개의 단어만 출력해봅시다  .vocab.freqs.most_common(20)를 통해 출력 가능합니다 
```py
# 상위 빈도수 20개 단어
print(TEXT.vocab.freqs.most_common(20))
```
```
[('the', 9076), ('.', 8640), (',', 7021), ('to', 5137), ('and', 5002), ('a', 3782), ('of', 3622), 
('i', 3379), ('in', 3112), ('is', 2239), ('you', 2156), ('that', 2036), ('it', 1850), ('for', 1842), 
('-', 1426), ('have', 1359), ('"', 1296), ('on', 1273), ('was', 1244), ('with', 1216)]
```
영어에서는 보통 the가 빈도수가 가장 많습니다 토치텍스트는 기본적으로 빈도수가 가장 높은 단어부터 작은 숫자를 부여합니다  
물론,<unk>는 0번 <pad>는 1번으로 자동으로 부여되므로 제외입니다

상위 정수 인덱스를 가진 10개의 단어를 출력합니다 다시 말해 0번부터9번까지의 단어를 출력해보겠습니다 

```py
# 상위 정수 인덱스 단어 10개 출력
print(TEXT.vocab.itos[:10]) 
```
```
['<unk>', '<pad>', 'the', '.', ',', 'to', 'and', 'a', 'of', 'i']
```

이제 레이블의 단어 집합에 대해서 빈도수가 가장 높은 단어들과 그 빈도수를 출력해 보겠습니다   

```py
# 상위 빈도순으로 udtags 출력
print(UD_TAGS.vocab.freqs.most_common())
``` 

```
[('NOUN', 34781), ('PUNCT', 23679), ('VERB', 23081), ('PRON', 18577), ('ADP', 17638), ('DET', 16285), 
('PROPN', 12946), ('ADJ', 12477), ('AUX', 12343), ('ADV', 10548), ('CCONJ', 6707), ('PART', 5567),
('NUM', 3999), ('SCONJ', 3843), ('X', 847), ('INTJ', 688), ('SYM', 599)]  
```
 
상위 정수 인덱스를 가진 10개의 단어를 출력합니다. 다시 말해 0번부터 9번까지의 단어를 출력해보겠습니다
```py
# 상위 정수 인덱스 순으로 출력
print(UD_TAGS.vocab.itos)
```
```
['<pad>', 'NOUN', 'PUNCT', 'VERB', 'PRON', 'ADP', 'DET', 'PROPN', 'ADJ', 'AUX', 
  'ADV', 'CCONJ', 'PART', 'NUM', 'SCONJ', 'X', 'INTJ', 'SYM']
```  
레이블에 속한 단어들의 분포를 출력해보겠습니다 
```py
def tag_percentage(tag_counts): # 태그 레이블의 분포를 확인하는 함수
    total_count = sum([count for tag, count in tag_counts])
    tag_counts_percentages = [(tag, count, count/total_count) for tag, count in tag_counts]

    return tag_counts_percentages
  
  
print("Tag  Occurences Percentage\n")
for tag, count, percent in tag_percentage(UD_TAGS.vocab.freqs.most_common()):
    print(f"{tag}\t{count}\t{percent*100:4.1f}%")  
``` 
```
Tag  Occurences Percentage

NOUN    34781   17.0%
PUNCT   23679   11.6%
VERB    23081   11.3%
PRON    18577    9.1%
ADP 17638    8.6%
DET 16285    8.0%
PROPN   12946    6.3%
ADJ 12477    6.1%
AUX 12343    6.0%
ADV 10548    5.2%
CCONJ   6707     3.3%
PART    5567     2.7%
NUM 3999     2.0%
SCONJ   3843     1.9%
X   847  0.4%
INTJ    688  0.3%
SYM 599  0.3%
```

### 4.데이터로더 만들기 
  
이제 데이터로더를 만듭니다 배치 크기는 64로 합니다 
  
```py
BATCH_SIZE = 64

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data), 
    batch_size = BATCH_SIZE,
    device = device)
```
 
첫번때 미니 배치만 꺼내서 미니 배치의 구성,크기,text를 출력해봅시다  
```py
batch = next(iter(train_iterator))  

batch
```
  
```
[torchtext.data.batch.Batch of size 64 from UDPOS]
    [.text]:[torch.cuda.LongTensor of size 46x64 (GPU 0)]
    [.udtags]:[torch.cuda.LongTensor of size 46x64 (GPU 0)]
    [.ptbtags]:[torch.cuda.LongTensor of size 46x64 (GPU 0)]
```
첫번때 미니 배치의 text의 크기를 출력해봅시다   
```py
batch.text.shape
```
```
torch.Size([46, 64]) 
``` 
첫번때 미니 배치의 크기는 (시퀀스 길이 x 배치 크기)입니다 batch_first =True를 하지 않았으므로 배치 크기가 두번때 차원이됩니닫 
```py
batch.text 
```
```
tensor([[ 732,  167,    2,  ...,    2,   59,  668],
        [  16,  196,  133,  ..., 2991,   46,    1],
        [   1,   29,   48,  ..., 1582,   12,    1],
        ...,
        [   1,    1,    1,  ...,    1,    1,    1],
        [   1,    1,    1,  ...,    1,    1,    1],
        [   1,    1,    1,  ...,    1,    1,    1]], device='cuda:0')
```  

## 3.모델 구현하기 
 
이제 모델을 구현해봅시다 기본적으로 다대다 RNN을 사용할텐데 일단 양방향 여부와 층의 개수는 변수로 두겠습니다 
 
```py
# 이번 모델에서는 batch_first=True를 사용하지 않으므로 배치 차원이 맨 앞이 아님.
class RNNPOSTagger(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout): 
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers = n_layers, bidirectional = bidirectional)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)        
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        # text = [sent len, batch size]
        embedded = self.dropout(self.embedding(text))

        # embedded = [sent len, batch size, emb dim]
        outputs, (hidden, cell) = self.rnn(embedded)

        # output = [sent len, batch size, hid dim * n directions]
        # hidden/cell = [n layers * n directions, batch size, hid dim]
        predictions = self.fc(self.dropout(outputs))

        # predictions = [sent len, batch size, output dim]
        return predictions
```
실제 클래스로부터 모덱 객체로 생성 시에 양방향 여부를 True로 주고 층의 개수를 2개로 합니다 
```py
INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
HIDDEN_DIM = 128
OUTPUT_DIM = len(UD_TAGS.vocab)
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.25

model = RNNPOSTagger(INPUT_DIM, 
                     EMBEDDING_DIM, 
                     HIDDEN_DIM, 
                     OUTPUT_DIM, 
                     N_LAYERS, 
                     BIDIRECTIONAL, 
                     DROPOUT)
```
파라리터 개수를 출력해보겠습니다
```py
# 파라미터 개수 출력
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')
```
```
The model has 1,027,510 trainable parameters
``` 
총 102만 7천 5백 10개의 파라미터가 있습니다 

## 4.사전 훈련된 워드 임베딩 사용하기  
앞서 언급하였던대로 이번 챕터에서는 사전 훈련된 워드 임베딩인 GloVe를 사용합니다 이를 위해서는 토치텍스트의 단어 집합 생성시에 저장해두었던
GloVe 임베딩을 nn.Embedding()에 연결해줄 필요가 있습니다 우선 단어 집합의 단어들에 맵핑된 사전 훈련된 워드 임베딩을 출력해봅시다 
 
```py
pretrained_embeddings = TEXT.vocab.vectors
print(pretrained_embeddings.shape)
```
```
torch.Size([3921, 100])
```
단어 집합에 존재하는 총 3,921개의 단어에 대해서 100차원의 벡터가 맵핑되어져 있습니다 이제 nn.Embedding()에 이를 연결시켜줍니다
```py
model.embedding.weight.data.copy_(pretrained_embeddings) # 임베딩 벡터값 copy
```
```
tensor([[ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0382, -0.2449,  0.7281,  ..., -0.1459,  0.8278,  0.2706],
        ...,
        [-0.1020,  0.7700,  0.1169,  ..., -0.1416, -0.1932, -0.4225],
        [-0.0263,  0.0179, -0.5016,  ..., -0.8688,  0.9409, -0.2882],
        [ 0.1519,  0.4712,  0.0895,  ..., -0.4702, -0.3127,  0.1078]]) 
```
우선 <unk> 토큰의 인덱스와 <pad> 토큰의 인덱스를 저장해둡니다 (물론, 각각 0과 1임을 이미 우리는 알고 있습니다)
```py
UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
print(UNK_IDX)
print(PAD_IDX)
```
```
0
1 
```
그리고 임의로 0번과 1번 단어에는 0벡터를 만들어줍니다
```py
model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM) # 0번 임베딩 벡터에는 0값을 채운다.
model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM) # 1번 임베딩 벡터에는 1값을 채운다.
print(model.embedding.weight.data)
```
```
tensor([[ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0382, -0.2449,  0.7281,  ..., -0.1459,  0.8278,  0.2706],
        ...,
        [-0.1020,  0.7700,  0.1169,  ..., -0.1416, -0.1932, -0.4225],
        [-0.0263,  0.0179, -0.5016,  ..., -0.8688,  0.9409, -0.2882],
        [ 0.1519,  0.4712,  0.0895,  ..., -0.4702, -0.3127,  0.1078]])
```
PAD 토큰과 UNK 토큰의 임베딩 벡터값이 0인것을 볼 수 있습니다 사전 훈련된 워드 임베딩을 사용할 준비가 되었습니다

 
## 5.옵티마이저와 비용 함수 구현하기 
옵티마이저 설계 전에 레이블 데이터의 패딩 토큰의 인덱스도 확인해봅시다
 
```py
TAG_PAD_IDX = UD_TAGS.vocab.stoi[UD_TAGS.pad_token]
print(TAG_PAD_IDX)
```
```
0
```
0인 것을 확인할 수 있습니다 이를 하는 이유는 아래 비용 함수를 선택할 때 인자로 주기 위함입니다 
이제 옵티마이저를 설정합니다 여기서는 Adam을 택했습니다
```py
optimizer = optim.Adam(model.parameters())
```
비용 함수로 크로스엔트로피 함수를 선택합니다 이떄 레이블 데이터의 패딩 토큰은 비용함수의 연산에 포함시키지도 않도록
레이블 데이터의 패딩 토큰을 무시하라고 기재해줍니다 
```py
criterion = nn.CrossEntropyLoss(ignore_index = TAG_PAD_IDX)
```
현재 GPU를 사용 중일때 GPU 연산을 할 수 있도록 지정해줍니다  
```py
model = model.to(device)
criterion = criterion.to(device)
```
아직 모델은 훈련되지 않은 상태이지만 모델에 입력값을 넣어 출력의 크기를 확인해볼까요? 여기서 넣는 입력값은 앞에서 꺼내두었던 천번째 배치입니다  
```py
prediction = model(batch.text)
```
예측값의 크기는 다음과 같습니다 
 
```py
prediction.shape
``` 

```
torch.Size([46,64,18])
```
46 × 64 × 18은 각각 (첫번째 배치의 시퀀스 길이 × 배치 크기 × 레이블 단어장의 크기)에 해당됩니다 주의할 점은 헌재는
batch_first를 해주지 않아 배치 크기가 맨 앞 차원이 아니라는 점입니다 또한 46은 첫번째 시퀀스 길이일뿐, 다른 배치들은
시퀀스 길이가 다를 수 있습니다  

이제 예측값에 대해서 시퀀스 길이와 배치길이를 모두 펼쳐주는 작업을 해보겠습니다  
```py 
prediction = prediction.view(-1, prediction.shape[-1])
prediction.shape
``` 
``` 
torch.Size([2944, 18]) 
``` 
크기가 (2,944 × 18)이 됩니다 이번에는 첫번째 배치의 레이블 데이터의 크기를 보겠습니다
```py 
batch.udtags.shape 
``` 
``` 
torch.Size([46, 64]) 
```
46 × 64는 (첫번째 배치의 시퀀스 길이 × 배치 크기)에 해당됩니다 이를 펼쳐보겠습니다
```py
batch.udtags.view(-1).shape 
```   
```
torch.Size([2944])
``` 

2,944의 크기를 가지게 됩니다 
