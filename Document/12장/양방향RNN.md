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
