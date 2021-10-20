# 어텐션을 이용한 텍스트 요약 (Text Summarization with Attention mechanism)

텍스트 요약은 상대적으로 큰 원문을 핵심 내용만 간추려서 상대적으로 작은 요약문으로 변환하는 것을 말합니다 읽는 사람이 시간을 단축해서 내용을 빠르게 이해할 수 있다는 점에서 글을 많이 쓰는 사람들에게는 꼭 필요한 능력중 하나 일 것입니다 그런데 만약 기계가 이를 자동으로 해줄 수만 있다면 얼마나 좋을까요? 

텍스트를 요약해주는 방법중 seq2seq를 이용하여 구현해 보겠습니다 그리고 어텐션 메커니즘을 적용해봅시다 

## 1. 텍스트 요약(Text Summarization)

텍스트 요약은 크게 추출적 요약(extractive summarization)과 추상적 요약(abstractive summarization)으로 나뉩니다

### 1) 추출적 요약(extractive summarization)

추출적 요약은 원문에서 중요한 핵심 문장 또는 단어구를 몇 개 뽑아서 이들로 구성된 요약문을 만드는 방법입니다 그렇기 때문에 추출적 요약의 결과로 나온 요약문의 문장이나 단어구들은 전부 원문에 있는 문장들입니다 추출적 요약의 대표적인 알고리즘으로는 머신 러닝 알고리즘인 텍스트랭크가 있습니다 아래의 링크에서 텍스트랭크로 구현된 세 줄 요약기를 시험해볼 수 있습니다 

링크 : https://summariz3.herokuapp.com/

위 링크로 이동하여 인테넷 뉴스나 가지고 있는 글을 복사 붙여넣기하여 결과를 살펴볼수있습니다 세 개의 문장은 전부 원문에 존재하던 문장들입니다 

이방법의 단점이라면 이미 존재 하는 문장이나 단어구로만 구성하는 모델이므로 모델의 언어표현 능력이 제한된다는 점입니다 

그렇다면 마치 사람처럼 원문에 없던 단어나 문장을 사용하면서 핵심만 간추려서 표현하는 요약 방법은 무엇일까요 

### 2) 추상적 요약 (abstractive summarization)

추상적 요약은 원문에 없던 문장이라도 핵심 문맥을 반영한 새로운 문장을 생성해서 원문을 요약하는 방법입니다 마치 사람이 요약하는 것 같은 방식인데, 당연히 추출적 요약보다는 난이도가 높습니다 

이 방법은 주로 인공 신경망을 사용하며 대표적인 모델로 seq2seq가 있습니다 단점이라면 seq2seq와 같은 인공 신경망들은 기본적으로 지도 학습이라는 점입니다 다시 말해 추상적 요약을 인공 신경망으로 훈련하기 위해서는 '원문'뿐만 아니라 '실제 요약문'이라는 레이블 데이터가 있어야 합니다 

그렇기 때문에 데이터를 구성하는 것 자체가 하나의 부담입니다 


## 2.아마존 리뷰 데이터에 대한 이해 

데이터는 아마존 리뷰 데이터입니다 아래의 링크에서 데이터를 다운로드 합니다 

링크 : https://www.kaggle.com/snap/amazon-fine-food-reviews


우선 필요한 도구들을 임포트 합니다 

```py
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from bs4 import BeautifulSoup 
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.preprocessing.sequence import pad_sequences
import urllib.request
np.random.seed(seed=0)
```

### 1)데이터 로드하기 

Reviews.csv 파일을 불러와 데이터 프레임에 저장하겠습니다 이 데이터는 실제로는 약 56만개의 샘플을 가지고 있습니다 하지만 여기서는 간단히 10만개의 샘플만 사용하겠습니다 이는 pd.read_csv의 nrows의 인자로 10만이라는 숫자를 적어주면 됩니다 

```py
# Reviews.csv 파일을 data라는 이름의 데이터프레임에 저장. 단, 10만개의 행(rows)으로 제한.
data = pd.read_csv("Reviews.csv 파일의 경로", nrows = 100000)
print('전체 리뷰 개수 :',(len(data)))
```
```
전체 리뷰 개수 : 100000
```

전체 리뷰 개수가 10만개인 것을 확인했습니다 5개의 샘플만 출력해봅시다 
```py
data.head()
```
```py
지면의 한계로 생략
```
5개의 샘플을 출력해보면 'Id', 'ProductId', 'UserId', 'ProfileName', 'HelpfulnessNumerator', 'HelpfulnessDenominator', 'Score', 'Time', 'Summary', 'Text'이라는 10개의 열이 존재함을 알 수 있습니다

그런데 사실 이 중 필요한 열은 'Text'열과 'Summary'열 뿐입니다

Text열과 Summary열만을 분리하고, 다른 열들은 데이터에서 제외시켜서 재저장합니다
그리고 5개의 샘플을 출력합니다

```py
data = data[['Text','Summary']]
data.head()
```

Text열과 Summary열만 저장된 것을 확인할 수 있습니다 Text열이 원문이고, Summary열이 Text열에 대한 요약입니다 다시 말해 모델은 Text(원문)으로부터 Summary(요약)을 예측하도록 훈련됩니다 
랜덤으로 샘플 몇 가지를 더 출력해봅시다

```py
data.sample(10)
```

여기서는 data.sample(10)를 한 번만 실행했지만 지속적으로 몇 차례 더 실행하면서 샘플의 구조를 확인해보세요 원문은 꽤 긴 반면에, Summary에는 3~4개의 단어만으로 구성된 경우도 많아보입니다

