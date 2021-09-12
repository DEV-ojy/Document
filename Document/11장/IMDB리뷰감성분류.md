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
