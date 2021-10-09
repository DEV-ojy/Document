# RNN을 이용한 텍스트 생성(Text Generation using RNN)

이번에는 다 대 일(many-to-one)구조의 RNN을 사용하여 문맥을 반영해서 텍스트를 생성하는 모델을 만들어보겠습니다

## 1.RNN을 이용하여 텍스트 생성하기 
예를 들어서 `경마장에 있는 말이 뛰고 있다`와`그의 말이 법이다`와 `가는 말이 고와야 오는 말이 곱다`
라는 세 가지 문장이 있다고 해봅시다 모델이 문맥을 학습할 수 있도록 전체 문장의 앞의 단어들을 전부 고려하여 학습하도록 데이터를 재구성한다면 아래와 같이 총 11개의 샘플이 구성됩니다 

![image](https://user-images.githubusercontent.com/80239748/136223470-458f24e2-9be7-4239-8180-8ecc8105b0c7.png)

### 1) 데이터에 대한 이해와 전처리 

```py
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from tensorflow.keras.utils import to_categorical
```

우선 예제로 언급한 3개의 한국어 문장을 저장합니다
```py
text="""경마장에 있는 말이 뛰고 있다\n
그의 말이 법이다\n
가는 말이 고와야 오는 말이 곱다\n"""
```
단어 집합을 생성하고 크기를 확인해보겠습니다 
```py
t = Tokenizer()
t.fit_on_texts([text])
vocab_size = len(t.word_index) + 1
# 케라스 토크나이저의 정수 인코딩은 인덱스가 1부터 시작하지만,
# 케라스 원-핫 인코딩에서 배열의 인덱스가 0부터 시작하기 때문에
# 배열의 크기를 실제 단어 집합의 크기보다 +1로 생성해야하므로 미리 +1 선언 
print('단어 집합의 크기 : %d' % vocab_size)
```
```
단어 집합의 크기 : 12
```
각 단어와 단어에 부여된 정수 인덱스를 출력해보겠습니다 

```py
print(t.word_index)
```
```
{'말이': 1, '경마장에': 2, '있는': 3, '뛰고': 4, '있다': 5, '그의': 6, '법이다': 7, '가는': 8, '고와야': 9, '오는': 10, '곱다': 11}
```
훈련 데이터를 만들어보겠습니다 
```py
sequences = list()
for line in text.split('\n'): # Wn을 기준으로 문장 토큰화
    encoded = t.texts_to_sequences([line])[0]
    for i in range(1, len(encoded)):
        sequence = encoded[:i+1]
        sequences.append(sequence)

print('학습에 사용할 샘플의 개수: %d' % len(sequences))
```
```
학습에 사용할 샘플의 개수: 11
```
```py
max_len=max(len(l) for l in sequences) # 모든 샘플에서 길이가 가장 긴 샘플의 길이 출력
print('샘플의 최대 길이 : {}'.format(max_len))
```
```
샘플의 최대 길이 : 6
```
전체 훈련 데이터에서 가장 긴 샘플의 길이가 6임을 확인했습니다 

이제 **전체 샘플의 길이를 6으로 패딩합니다**
```py
sequences = pad_sequences(sequences, maxlen=max_len, padding='pre')
```

pad_sequences()는 모든 샘플에 대해서 0을 사용하여 길이를 맞춰줍니다 maxlen의 값으로 6을 주면 모든 샘플의 길이를 6으로 맞춰주며, padding의 인자로 'pre'를 주면 길이가 6보다 짧은 샘플의 앞에 0으로 채웁니다

```py
print(sequences)
```
```
[[ 0  0  0  0  2  3]
 [ 0  0  0  2  3  1]
 [ 0  0  2  3  1  4]
 [ 0  2  3  1  4  5]
 [ 0  0  0  0  6  1]
 [ 0  0  0  6  1  7]
 [ 0  0  0  0  8  1]
 [ 0  0  0  8  1  9]
 [ 0  0  8  1  9 10]
 [ 0  8  1  9 10  1]
 [ 8  1  9 10  1 11]]
```

길이가 6보다 짧은 모든 샘플의 앞에 0을 채워서 모든 샘플의 길이를 6으로 맞춰줍니다 이제 각 샘플의 마지막 단어를 레이블로 분리합니다 레이블의 분리는 Numpy를 이용해서 가능합니다 

```py
sequences = np.array(sequences)
X = sequences[:,:-1]
y = sequences[:,-1]
# 리스트의 마지막 값을 제외하고 저장한 것은 X
# 리스트의 마지막 값만 저장한 것은 y. 이는 레이블에 해당됨.

```

분리된 x와y에 대해서 출력해보면 다음과 같습니다 
```py
print(x)
```
```
[[ 0  0  0  0  2]
 [ 0  0  0  2  3]
 [ 0  0  2  3  1]
 [ 0  2  3  1  4]
 [ 0  0  0  0  6]
 [ 0  0  0  6  1]
 [ 0  0  0  0  8]
 [ 0  0  0  8  1]
 [ 0  0  8  1  9]
 [ 0  8  1  9 10]
 [ 8  1  9 10  1]]
```
```py
print(y) # 모든 샘플에 대한 레이블 출력
```
```
[ 3  1  4  5  1  7  1  9 10  1 11]
```

레이블이 분리되었습니다 이제 RNN 모델에 훈려녀 데이터를 훈련 시키기 전에 레이블에 대해서 원-핫 인코딩을 수행합니다 
```py
y = to_categorical(y, num_classes=vocab_size)

print(y)
```
```
[[0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0.] # 3에 대한 원-핫 벡터
 [0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.] # 1에 대한 원-핫 벡터
 [0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0.] # 4에 대한 원-핫 벡터
 [0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0.] # 5에 대한 원-핫 벡터
 [0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.] # 1에 대한 원-핫 벡터
 [0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0.] # 7에 대한 원-핫 벡터
 [0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.] # 1에 대한 원-핫 벡터
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0.] # 9에 대한 원-핫 벡터
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0.] # 10에 대한 원-핫 벡터
 [0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.] # 1에 대한 원-핫 벡터
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]] # 11에 대한 원-핫 벡터
```

정상적으로 원-핫 인코딩이 수행된 것을 볼 수 있습니다 

### 2) 모델 설계하기 

```py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM
```
```py
model = Sequential()
model.add(Embedding(vocab_size, 10, input_length=max_len-1))
# y데이터를 분리하였으므로 이제 X데이터의 길이는 기존 데이터의 길이 - 1
model.add(LSTM(128))
model.add(Dense(vocab_size, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=200, verbose=2)
```

각 단어의 임베딩 벡터는 10차원을 가지고, 128의 은닉 상태 크기를 가지는 LSTM을 사용합니다 문장을 생성하는 함수 sentence_generation을 만들어서 문장을 생성해봅시다

```py
def sentence_generation(model, t, current_word, n): # 모델, 토크나이저, 현재 단어, 반복할 횟수
    init_word = current_word # 처음 들어온 단어도 마지막에 같이 출력하기위해 저장
    sentence = ''
    for _ in range(n): # n번 반복
        encoded = t.texts_to_sequences([current_word])[0] # 현재 단어에 대한 정수 인코딩
        encoded = pad_sequences([encoded], maxlen=23, padding='pre') # 데이터에 대한 패딩
        result = model.predict_classes(encoded, verbose=0)
    # 입력한 X(현재 단어)에 대해서 y를 예측하고 y(예측한 단어)를 result에 저장.
        for word, index in t.word_index.items(): 
            if index == result: # 만약 예측한 단어와 인덱스와 동일한 단어가 있다면
                break # 해당 단어가 예측 단어이므로 break
        current_word = current_word + ' '  + word # 현재 단어 + ' ' + 예측 단어를 현재 단어로 변경
        sentence = sentence + ' ' + word # 예측 단어를 문장에 저장
    # for문이므로 이 행동을 다시 반복
    sentence = init_word + sentence
    return sentence
```
```py
print(sentence_generation(model, t, 'i', 10))
# 임의의 단어 'i'에 대해서 10개의 단어를 추가 생성
```
```
i disapprove of school vouchers can i still apply for them
```
```py
print(sentence_generation(model, t, 'how', 10))
# 임의의 단어 'how'에 대해서 10개의 단어를 추가 생성
```
```
how to make facebook more accountable will so your neighbor chasing
```
