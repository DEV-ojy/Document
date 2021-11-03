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
## 2.단어 집합 생성

서브워드텍스트인코더를 사용해봅시다 자주 사용되는 서브워드 단위로 토큰을 분리하는 토크나이저로 학습 데이터로부터 학습하여 서브워드로 구성된 단어 집합을 생성합니다 

```py
# 서브워드텍스트인코더를 사용하여 질문, 답변 데이터로부터 단어 집합(Vocabulary) 생성
tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
    questions + answers, target_vocab_size=2**13)
```
단어집합이 생성되었습니다 그런데 인코더-디코더 모델 계열에는 디코더의 입력으로 사용할 시작을 의미하는 시작 토큰 SOS 종료 토큰 EOS가 존재합니다 해당 토큰들도 단어 집합에 포함시킬 필요가 있으므로 이 두 토큰에 정수를 부여해줍니다 

```py
# 시작 토큰과 종료 토큰에 대한 정수 부여.
START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]

# 시작 토큰과 종료 토큰을 고려하여 단어 집합의 크기를 + 2
VOCAB_SIZE = tokenizer.vocab_size + 2
```
시작 토큰과 종료 토큰을 추가해주었으나 단어 집합의 크기도 +2를 해줍니다

```py
print('시작 토큰 번호 :',START_TOKEN)
print('종료 토큰 번호 :',END_TOKEN)
print('단어 집합의 크기 :',VOCAB_SIZE)
```
```
시작 토큰 번호 : [8178]
종료 토큰 번호 : [8179]
단어 집합의 크기 : 8180
```

패딩에 사용될 0번 토큰부터 마지막 토큰인 8,179번 토큰까지의 개수를 카운트하면 단어 집합의 크기는 8,180개입니다

## 3. 정수 인코딩과 패딩 

단어집합을 생성한 후에는 서브워드텍스트코더의 토크나이저로 정수 인코딩을 진행할 수 있습니다
이는 토크나이저의 .encode()를 사용하여 가능합니다 우선 임의로 선택한 20번 질문 샘플, 즉 questions[20]을 가지고 정수 인코딩을 진행해봅시다 

```py
# 서브워드텍스트인코더 토크나이저의 .encode()를 사용하여 텍스트 시퀀스를 정수 시퀀스로 변환.
print('임의의 질문 샘플을 정수 인코딩 : {}'.format(tokenizer.encode(questions[20])))
```
```
임의의 질문 샘플을 정수 인코딩 : [5766, 611, 3509, 141, 685, 3747, 849]
```
임의의 질문 문장이 정수 시퀀스로 변환되었습니다 반대로 정수 인코딩 된 결과는 다시 decode()를 사용하여 기존의 텍스트 시퀀스로 복원할 수 있습니다 20번 질문 샘플을 가지고 정수 인코딩하고 다시 이를 디코딩하는 과정은 다음과 같습니다
```py
# 서브워드텍스트인코더 토크나이저의 .encode()와 .decode() 테스트해보기
# 임의의 입력 문장을 sample_string에 저장
sample_string = questions[20]

# encode() : 텍스트 시퀀스 --> 정수 시퀀스
tokenized_string = tokenizer.encode(sample_string)
print ('정수 인코딩 후의 문장 {}'.format(tokenized_string))

# decode() : 정수 시퀀스 --> 텍스트 시퀀스
original_string = tokenizer.decode(tokenized_string)
print ('기존 문장: {}'.format(original_string))
```
```
정수 인코딩 후의 문장 [5766, 611, 3509, 141, 685, 3747, 849]
기존 문장: 가스비 비싼데 감기 걸리겠어
```

정수 인코딩 된 문장을 .decode()을 하면 자동으로 서브워드들까지 다시 붙여서 기존 단어로 복원해줍니다 가령 정수 인코딩 문장을 보면 정수가 7개인데 기존 문장의 띄어쓰기 단위인 어절은 4개밖에 존재하지 않습니다 이는 '가스비'나 '비싼데'라는 한 어절이 정수 인코딩 후에는 두 개 이상의 정수일 수 있다는 겁니다 각 정수가 어떤 서브워드로 맵핑되는지 출력해봅시다 

```py
# 각 정수는 각 단어와 어떻게 mapping되는지 병렬로 출력
# 서브워드텍스트인코더는 의미있는 단위의 서브워드로 토크나이징한다. 띄어쓰기 단위 X 형태소 분석 단위 X
for ts in tokenized_string:
  print ('{} ----> {}'.format(ts, tokenizer.decode([ts])))
```
```
5766 ----> 가스
611 ----> 비 
3509 ----> 비싼
141 ----> 데 
685 ----> 감기 
3747 ----> 걸리
849 ----> 겠어
```

샘플 1개를 가지고 정수 인코딩과 디코딩을 수행해보았습니다 이번에는 전체 데이터에 대해서 정수 인코딩과 패딩을 진행합니다 이를 위한 함수로 tokenize_and_filter()를 만듭니다 여기서는 임의로 패딩의 길이는 40으로 정했습니다 

```py
# 최대 길이를 40으로 정의
MAX_LENGTH = 40

# 토큰화 / 정수 인코딩 / 시작 토큰과 종료 토큰 추가 / 패딩
def tokenize_and_filter(inputs, outputs):
  tokenized_inputs, tokenized_outputs = [], []

  for (sentence1, sentence2) in zip(inputs, outputs):
    # encode(토큰화 + 정수 인코딩), 시작 토큰과 종료 토큰 추가
    sentence1 = START_TOKEN + tokenizer.encode(sentence1) + END_TOKEN
    sentence2 = START_TOKEN + tokenizer.encode(sentence2) + END_TOKEN

    tokenized_inputs.append(sentence1)
    tokenized_outputs.append(sentence2)

  # 패딩
  tokenized_inputs = tf.keras.preprocessing.sequence.pad_sequences(
      tokenized_inputs, maxlen=MAX_LENGTH, padding='post')
  tokenized_outputs = tf.keras.preprocessing.sequence.pad_sequences(
      tokenized_outputs, maxlen=MAX_LENGTH, padding='post')

  return tokenized_inputs, tokenized_outputs

```
```
questions, answers = tokenize_and_filter(questions, answers)
```

