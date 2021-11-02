# 트랜스포머를 이용한 한국어 챗봇 (Transformer Chatbot)1

트랜스포머 코드를 사용하여 일상 대회 챗봇을 구현해보겠습니다 

## 1.데이터 로드하기

```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import urllib.request
import time
import tensorflow_datasets as tfds
import tensorflow as tf
```
```py
urllib.request.urlretrieve("https://raw.githubusercontent.com/songys/Chatbot_data/master/ChatbotData.csv", filename="ChatBotData.csv")
train_data = pd.read_csv('ChatBotData.csv')
```
데이터는 질문(Q)과 대답(A)의 쌍으로 이루어진 데이터 입니다 
```py
print('챗봇 샘플의 개수 :',len(train_data))
```
```
챗봇 샘플의 개수 : 11823
```
총 샘플의 개수는 11,823개입니다 불필요한 Null 값이 있는지 확인해봅시다ㅣ 
```py
print(train_data.insull().sum())
```
```
Q        0
A        0
label    0
dtype: int64
```

Null 값은 별도로 존재하지 않습니다 이번에는 토큰화를 위해 형태소 분석기를 사용하지 않고 다른 방법인 학습 기반의 토크나이저를 사용할 것입니다 그래서 원 데이터에서 `? . !`와 같은 구두점을 미리 처리해두어야 합니다 

구두점을 단순히 제거할수도 있겠지만 여기서는 구두점 앞에 공백을 추가하여 다른 문자들과 구분해주겠습니다 

### 질문
```py
questions = []
for sentence in train_data['Q']:
    # 구두점에 대해서 띄어쓰기
    # ex) 12시 땡! -> 12시 땡 !
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = sentence.strip()
    questions.append(sentence)
```

### 대답
```py
questions = []
for sentence in train_data['A']:
    # 구두점에 대해서 띄어쓰기
    # ex) 12시 땡! -> 12시 땡 !
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = sentence.strip()
    questions.append(sentence)
``` 
