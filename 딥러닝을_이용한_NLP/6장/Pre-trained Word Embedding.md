# 사전 훈련된 워드 임베딩(Pre-trained Word Embedding)

이번 챕터에서는 케라스의 임베딩 층(embedding layer)과 사전 훈련된 워드 임베딩(pre-trained word embedding)을 가져와서 사용하는 것을 비교해봅니다

자연어 처리를 하려고 할 때 갖고 있는 훈련 데이터의 단어들을 임베딩 층(embedding layer)을 구현하여 임베딩 벡터로 학습하는 경우가 있습니다 케라스에서는 이를 Embedding()이라는 도구를 사용하여 구현합니다.

그런데 위키피디아 등과 같은 방대한 코퍼스를 가지고 Word2vec, FastText, GloVe 등을 통해서 이미 미리 훈련된 임베딩 벡터를 불러오는 방법을 사용하는 경우도 있습니다

이는 현재 갖고 있는 훈련 데이터를 임베딩 층으로 처음부터 학습을 하는 방법과는 대조됩니다


## 1. 케라스 임베딩 층(Keras Embedding layer)

케라스는 훈련 데이터의 단어들에 대해 워드 임베딩을 수행하는 도구 Embedding()을 제공합니다 Embedding()은 인공 신경망 구조 관점에서 임베딩 층(embedding layer)을 구현합니다

### 1) 임베딩 층은 룩업 테이블이다

임베딩 층의 입력으로 사용하기 위해서 입력 시퀀스의 각 단어들은 모두 정수 인코딩이 되어있어야 합니다

**어떤 단어 → 단어에 부여된 고유한 정수값 → 임베딩 층 통과 → 밀집 벡터**

임베딩 층은 입력 정수에 대해 밀집 벡터(dense vector)로 맵핑하고 이 밀집 벡터는 인공 신경망의 학습 과정에서 가중치가 학습되는 것과 같은 방식으로 훈련됩니다 

훈련 과정에서 단어는 모델이 풀고자하는 작업에 맞는 값으로 업데이트 됩니다 그리고 이 밀집 벡터를 임베딩 벡터라고 부릅니다

정수를 밀집 벡터 또는 임베딩 벡터로 맵핑한다는 것은 어떤 의미일까요 특정 단어와 맵핑되는 정수를 인덱스로 가지는 테이블로부터 임베딩 벡터 값을 가져오는 룩업 테이블이라고 볼 수 있습니다

그리고 이 테이블은 단어 집합의 크기만큼의 행을 가지므로 모든 단어는 고유한 임베딩 벡터를 가집니다

![image](https://user-images.githubusercontent.com/80239748/141775046-6e3bd2d9-07cf-49ee-a228-0a0cfa6a458f.png)

위의 그림은 단어 great이 정수 인코딩 된 후 테이블로부터 해당 인덱스에 위치한 임베딩 벡터를 꺼내오는 모습을 보여줍니다 위의 그림에서는 임베딩 벡터의 차원이 4로 설정되어져 있습니다

그리고 단어 great은 정수 인코딩 과정에서 1,918의 정수로 인코딩이 되었고 그에 따라 단어 집합의 크기만큼의 행을 가지는 테이블에서 인덱스 1,918번에 위치한 행을 단어 great의 임베딩 벡터로 사용합니다

이 임베딩 벡터는 모델의 입력이 되고, 역전파 과정에서 단어 great의 임베딩 벡터값이 학습됩니다

룩업 테이블의 개념을 이론적으로 우선 접하고, 처음 케라스를 배울 때 어떤 분들은 임베딩 층의 입력이 원-핫 벡터가 아니어도 동작한다는 점에 헷갈려 합니다 

케라스는 단어를 정수 인덱스로 바꾸고 원-핫 벡터로 한번 더 바꾸고나서 임베딩 층의 입력으로 사용하는 것이 아니라, 단어를 정수 인덱스로만 바꾼채로 임베딩 층의 입력으로 사용해도 룩업 테이블 된 결과인 임베딩 벡터를 리턴합니다

케라스의 임베딩 층 구현 코드를 봅시다

```
# 아래의 각 인자는 저자가 임의로 선정한 숫자들이며 의미있는 선정 기준이 아님.
v = Embedding(20000, 128, input_length=500)
# vocab_size = 20000
# output_dim = 128
# input_length = 500
```

임베딩 층은 다음과 같은 세 개의 인자를 받습니

**vocab_size** : 텍스트 데이터의 전체 단어 집합의 크기입니다.
**output_dim** : 워드 임베딩 후의 임베딩 벡터의 차원입니다.
**input_length** : 입력 시퀀스의 길이입니다. 만약 갖고있는 각 샘플의 길이가 500개의 단어로 구성되어있다면 이 값은 500이 됩니다

Embedding()은 (number of samples, input_length)인 2D 정수 텐서를 입력받습니다  이 때 각 sample은 정수 인코딩이 된 결과로, 정수의 시퀀스입니다 

Embedding()은 워드 임베딩 작업을 수행하고 (number of samples, input_length, embedding word dimentionality)인 3D 실수 텐서를 리턴합니다

케라스의 임베딩 층(embedding layer)을 사용하는 간단한 실습을 진행해보겠습니다

### 2) 임베딩 층 사용하기 

RNN 챕터에서 사용했었던 임베딩 층을 복습해보겠습니다 문장의 긍,부정을 판단하는 감성 분류 모델을 만들어봅시다 

```py
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
```
```py
sentences = ['nice great best amazing', 'stop lies', 'pitiful nerd', 'excellent work', 'supreme quality', 'bad', 'highly respectable']
y_train = [1, 0, 0, 1, 1, 0, 1]
```

문장과 레이블 데이터를 만들었습니다 긍정은 1 부정은 0인 레이블입니다 

```py
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)
vocab_size = len(tokenizer.word_index) + 1
print(vocab_size)
```
```
16
```
케라스의 Tokenizer()를 사용하여 토큰화 시킨다 
```py
X_encoded = tokenizer.texts_to_sequences(sentences)
print(X_encoded)
```
```
[[1, 2, 3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13], [14, 15]]
```
각 문장에 대해서 정수 인코딩을 수행합니다 
```py
max_len = max(len(l) for l in X_encoded)
print(max_len)
```
```
4
```
문장 중에서 가장 길이가 긴 문장의 길이는 4입니다 
```py
X_train = pad_sequences(X_encoded, maxlen=max_len, padding='post')
y_train = np.array(y_train)
print(X_train)
```
```
[[ 1  2  3  4]
 [ 5  6  0  0]
 [ 7  8  0  0]
 [ 9 10  0  0]
 [11 12  0  0]
 [13  0  0  0]
 [14 15  0  0]]
```
모든 문장을 패딩하여 길이를 4로 만들어줍니다 훈련 데이터에 대한 전처리가 끝났습니다
모델을 설계합니다 
```py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten

embedding_dim = 4

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
```
출력층에 1개의 뉴런에 활성화 함수로는 시그모이드 함수를 사용하여 이진 분류를 수행합니다
```py
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model.fit(X_train, y_train, epochs=100, verbose=2)
```

## 2. 사전 훈련된 워드 임베딩(Pre-Trained Word Embedding) 사용하기

임베딩 벡터를 얻기 위해서 케라스의 Embedding()을 사용하기도 하지만, 때로는 이미 훈련되어져 있는 워드 임베딩을 불러서 이를 임베딩 벡터로 사용하기도 합니다 훈련 데이터가 적은 상황이라면 모델에 케라스의 Embedding()을 사용하는 것보다 다른 텍스트 데이터로 사전 훈련되어 있는 임베딩 벡터를 불러오는 것이 나은 선택일 수 있습니다

훈련 데이터가 적다면 케라스의 Embedding()으로 해당 문제에 충분히 특화된 임베딩 벡터를 만들어내는 것이 쉽지 않습니다 차라리 해당 문제에 특화된 임베딩 벡터를 만드는 것이 어렵다면, 해당 문제에 특화된 것은 아니지만 보다 일반적이고 보다 많은 훈련 데이터로 이미 Word2Vec이나 GloVe 등으로 학습되어져 있는 임베딩 벡터들을 사용하는 것이 성능의 개선을 가져올 수 있습니다

사전 훈련된 GloVe와 Word2Vec 임베딩을 사용해서 모델을 훈련시키는 실습을 진행해봅시다

GloVe 다운로드 링크 : http://nlp.stanford.edu/data/glove.6B.zip
Word2Vec 다운로드 링크 : https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM

훈련 데이터는 앞서 사용했던 데이터에 동일한 전처리가 수행된 상태라고 가정하겠습니다

```py
print(X_train)
```
```
[[ 1  2  3  4]
 [ 5  6  0  0]
 [ 7  8  0  0]
 [ 9 10  0  0]
 [11 12  0  0]
 [13  0  0  0]
 [14 15  0  0]]
```
```py
print(y_train)
```
```
[1, 0, 0, 1, 1, 0, 1]
```