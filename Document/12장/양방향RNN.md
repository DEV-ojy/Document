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

영어에서는 보통 the가 빈도수가 가장 많습니다 토치텍스트는 기본적으로 빈도수가 가장 높은 단어부터 작은 숫자를 부여합니다 물론,<unk>는 0번 <pad>는 1번으로 자동으로 부여되므로 제외입니다 
  
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
