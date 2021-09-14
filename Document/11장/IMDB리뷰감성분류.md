#   IMDB 리뷰 감성 분류하기(IMDB Movie Review Sentiment Analysis)

영화 사이트 IMDB의 리뷰 데이터는 리뷰에 대한 텍스트와 해당 리뷰가 긍정(1)인지 부정(0)인지 표시한 레이블로 구성된 데이터 입니다 


## 1.셋팅 하기 

우선 필요한 도구들을 임포트합니다 

```py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext import data, datasets
import random
```
랜덤 시드를 고정 

```py
SEED = 5
random.seed(SEED)
torch.manual_seed(SEED)
```
하이퍼파라미터들을 변수로 정의합니다 
```py
# 하이퍼파라미터
BATCH_SIZE = 64
lr = 0.001
EPOCHS = 10
```
현 환경에서 GPU를 사용 가능하면 GPU를 사용하고 CPU를 사용 가능하다면 CPU를 사용하도록 설정합니다  
```py
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
print("cpu와 cuda 중 다음 기기로 학습함:", DEVICE)
```
```
cpu와 cuda 중 다음 기기로 학습함: cuda
```

## 2. 토치텍스트를 이용한 전처리

여기서는 앞서 배운 토치텍스트를 사용하여 전처리를 진행합니다

### 1.데이터 로드하기 : torchtext.data
torchtext.data의 Field 클래스를 사용하여 영화 리뷰에 대한 객체 TEXT, 레이블을 위한 객체 LABEL을 생성
```py
TEXT = data.Field(sequential=True, batch_first=True, lower=True)
LABEL = data.Field(sequential=False, batch_first=True)
```
데이터 셋이 순차적인 데이터셋임을 알수 있도록 sequential인자값으로 True를 명시해줍니다 레이블은 단순한 클래스를 나타내는 숫자로
순차적인 데이터가 아니므로 False를 명시합니다 batch_first는 신경망에 입력되는 텐서의 첫번째 차원값이 batch_size가 되도록합니다
그리고 lower변수를 통해 텍스트 데이터 속 모든 영문알파벳이 소문자가 되도록 합니다 


### 2.데이터 로드 및 분할하기 : torchtext.datasets
torchtext.datasets을 통해 IMDB 리뷰 데이터를 다운로드 할수있습니다 데이터를 다운받는 동시에 훈련 데이터와 테스트 데이터를 분할하고, 각각 trainset, testset에 저장합니다

```py
# 전체 데이터를 훈련 데이터와 테스트 데이터를 8:2 비율로 나누기
trainset, testset = datasets.IMDB.splits(TEXT, LABEL)
```

텍스트와 레이블이 제데로 저장되었는지 확인하기 위해서  trainset.fields를 통해 trainset이 포함하는 각 요소를 확인해봅시다
```py
print('trainset의 구성 요소 출력 : ', trainset.fields)
```
```
trainset의 구성 요소 출력 :  {'text': <torchtext.data.field.Field object at 0x7f175ce3fe10>, 'label': <torchtext.data.field.Field object at 0x7f175ce3fe80>}
```
리뷰 데이터가 저장되어져 있는 text 필드와 레이블이 저장되어져 있는 label 필드가 존재합니다. testset.fields도 출력해봅니다
```py
print('testset의 구성 요소 출력 : ', testset.fields)
```
```
testset의 구성 요소 출력 :  {'text': <torchtext.data.field.Field object at 0x7f175ce3fe10>, 'label': <torchtext.data.field.Field object at 0x7f175ce3fe80>}
```
첫번째 룬련 샘플과 해당 샘플에 대한 레이블을 출력해보겠습니다 
```py
print(vars(trainset[0]))
```
```
{'text': ['if', 'you', 'like', 'jamie', 'foxx,(alvin', 'sanders),"date', 'from', 'hell",\'01,', 'you', 'will', 'love', 'his', 'acting', 'as', 'a', 'guy', 'who', 'never', 'gets', 'an', 'even', 'break', 'in', 'life', 'and', 'winds', 'up', 'messing', 'around', 'with', 'shrimp,', '(jumbo', 'size)', 'and', 'at', 'the', 'same', 'time', 'lots', 'of', 'gold', 'bars.', 'alvin', 'sanders', 'has', 'plenty', 'of', 'fbi', 'eyes', 'watching', 'him', 'and', 'winds', 'up', 'getting', 'hit', 'by', 'a', 'brick', 'in', 'the', 'jaw,', 'and', 'david', 'morse,(edgar', 'clenteen),', '"hack"', "'02", 'tv', 'series,', 'decides', 'to', 'zero', 'in', 'on', 'poor', 'alvin', 'and', 'use', 'him', 'as', 'a', 'so', 'called', 'fish', 'hook', 'to', 'attract', 'the', 'criminals.', 'there', 'is', 'lots', 'of', 'laughs,', 'drama,', 'cold', 'blood', 'killings', 'and', 'excellent', 'film', 'locations', 'and', 'plenty', 'of', 'expensive', 'cars', 'being', 'sent', 'to', 'the', 'junk', 'yard.', 'jamie', 'foxx', 'and', 'david', 'morse', 'were', 'outstanding', 'actors', 'in', 'this', 'film', 'and', 'it', 'was', 'great', 'entertainment', 'through', 'out', 'the', 'entire', 'picture.'],
'label': 'pos'}
```

'text' : []에서 대괄호 안에 위치한 단어들이 첫번째 IMDB 리뷰에 해당되면 'label' : []에서 대괄호 안의 단어가 첫번째 IMDB리뷰의 ㅣ레이블에 해당됩니다 여기서 'pos'는 positivie의 줄인말로 긍정을 의미합니다 

### 3. 단어 집합 만들기

이제 단어 집합을 만들어줍니다 단어 집합이란 중복을 제거한 총 단어들의 집합을 의미합니다 
```py
TEXT.build_vocab(trainset, min_freq=5) # 단어 집합 생성
LABEL.build_vocab(trainset)
```
위에서 min_freq는 학습 데이터에서 최소 5번 이상 등장한 단어만을 단어 집합에 추가하겠다는 의미입니다 이때 학습 데이터에서 5번 미만으로 등장한 단어는 Unkown이라는 의미에서 '<unk>'라는 토큰으로 대체됩니다 
  
단어 집합의 크기와 클래스의 개수를 변수에 저장하고 출력해봅니다 단어 집합의 크기란 결국 중복을 제거한 총 단어의 개수입니다  
```py
vocab_size = len(TEXT.vocab)
n_classes = 2
print('단어 집합의 크기 : {}'.format(vocab_size))
print('클래스의 개수 : {}'.format(n_classes))
```
```
단어 집합의 크기 : 46159
클래스의 개수 : 2
```
  
stoi로 단어와 각 단어의 정수 인덱스가 저장되어져 있는 딕셔너리 객체에 접근할 수 있습니다
```py
print(TEXT.vocab.stoi)
```
```
defaultdict(<function _default_unk_index at 0x7fb279f3cc80>, {'<unk>': 0, '<pad>': 1, 'the': 2, 'a': 3, 'and': 4, 'of': 5
... 중략 ...
'zoe,': 46150, 'zombies"': 46151, 'zombies)': 46152, 'zombified': 46153, 'zone.<br': 46154, 'zoolander': 46155, 'zwick': 46156, '{the': 46157, 'émigré': 46158})
```
unknown을 의미하는 <unk>라는 단어가 0번 인덱스로 부여되어져 있고, 그 외에도 수많은 단어들이 고유한 정수 인덱스가 부여된 것을 볼 수 있습니다 
