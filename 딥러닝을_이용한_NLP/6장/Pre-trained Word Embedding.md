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
### 1) 사전 훈련된 GloVe 사용하기

임베딩 층을 설계하기 위한 과정부터 달라집니다 우선 다운로드 받은 파일 glove.6B.zip의 압축을 풀면 그 안에 4개의 파일이 있는데 여기서 사용할 파일은 glove.6B.100d.txt 파일입니다

```py
from urllib.request import urlretrieve, urlopen
import gzip
import zipfile
```
```py
urlretrieve("http://nlp.stanford.edu/data/glove.6B.zip", filename="glove.6B.zip")
zf = zipfile.ZipFile('glove.6B.zip')
zf.extractall() 
zf.close()
```

glove.6B.100d.txt에 있는 모든 임베딩 벡터들을 불러와보겠습니다 형식은 파이썬의 Dictionary 구조를 사용합니다

```py
embedding_dict = dict()

f = open('glove.6B.100d.txt', encoding="utf8")

for line in f:
    word_vector = line.split()
    word = word_vector[0]

    # 100개의 값을 가지는 array로 변환
    word_vector_arr = np.asarray(word_vector[1:], dtype='float32')
    embedding_dict[word] = word_vector_arr
f.close()

print('%s개의 Embedding vector가 있습니다.' % len(embedding_dict))
```
```
400000개의 Embedding vector가 있습니다.
```
임의의 단어 'respectable'에 대해서 임베딩 벡터를 출력해봅니다
```py
print(embedding_dict['respectable'])
print(len(embedding_dict['respectable']))
```
```
[-0.049773   0.19903    0.10585 ... 중략 ... -0.032502   0.38025  ]
100
```
벡터값이 출력되며 길이는 100인 것을 확인할 수 있습니다 단어 집합 크기의 행과 100개의 열을 가지는 행렬 생성합니다 이 행렬의 값은 전부 0으로 채웁니다 이 행렬에 사전 훈련된 임베딩 값을 넣어줄 것입니다
```py
embedding_matrix = np.zeros((vocab_size, 100))
np.shape(embedding_matrix)
```
```
(16, 100)
```
```py
print(tokenizer.word_index.items())
```
```
dict_items([('nice', 1), ('great', 2), ('best', 3), ('amazing', 4), ('stop', 5), ('lies', 6), 
('pitiful', 7), ('nerd', 8), ('excellent', 9), ('work', 10), ('supreme', 11), ('quality', 12), 
('bad', 13), ('highly', 14), ('respectable', 15)])
```

단어 `great`의 인덱스는 2입니다 

```py
tokenizer.word_index['great']
```
```
2
```
사전 훈련된 GloVe에서 `great`의 벡터값을 확인합니다 

```py
print(embedding_dict['great'])
```
```
[-0.013786   0.38216    0.53236    0.15261   -0.29694   -0.20558
.. 중략 ...
 -0.69183   -1.0426     0.28855    0.63056  ]
```
이제 훈련 데이터의 단어 집합의 모든 단어에 대해서 사전 훈련된 GloVe의 임베딩 벡터들은 맵핑한 후에 `great`의 벡터값이 잘 들어갔는데 확인해봅시다 

```py
for word, index in tokenizer.word_index.items():
    # 단어와 맵핑되는 사전 훈련된 임베딩 벡터값
    vector_value = embedding_dict.get(word)
    if vector_value is not None:
        embedding_matrix[index] = vector_value
```
embedding_matrix의 인덱스 2에서의 값을 확인합니다

```py
tokenizer.word_index['great']
```
```
2
```
사전 훈련된 GloVe에서 `great`의 벡터값을 확인합니다 
```py
print(embedding_dict['great'])
```
```
[-0.013786   0.38216    0.53236    0.15261   -0.29694   -0.20558
.. 중략 ...
 -0.69183   -1.0426     0.28855    0.63056  ]
```
이제 훈련 데이터의 단어 집합의 모든 단어에 대해서 사전훈련된 GloVe의 임베딩 벡터들을 맵핑한 후에 `great`의 벡터값이 잘 들어갔는지 확인해보겠습니다 
```py
for word, index in tokenizer.word_index.items():
    # 단어와 맵핑되는 사전 훈련된 임베딩 벡터값
    vector_value = embedding_dict.get(word)
    if vector_value is not None:
        embedding_matrix[index] = vector_value
```
embedding_matrix의 인덱스 2에서의 값을 확인합니다 
```py
embedding_matrix[2]
```
```
array([-0.013786  ,  0.38216001,  0.53236002,  0.15261   , -0.29694   ,
        ... 중략 ...
       -0.39346001, -0.69182998, -1.04260004,  0.28854999,  0.63055998])
```
이전에 확인한 사전에 훈련된 GloVe에서의 `great`의 벡터값과 일치합니다 이제 Embedding layer에 우리가 만든 매트릭스를 초기값으로 설정해줍니다 헌재 실습에서 사전 훈련된 워드 임베딩을 100차원의 값인 것으로 사용하고 있기
때문에 임베딩층의 output_dim의 인자값으로 100을 주어야합니다 

그리고 사전 훈련된 워드 임베딩을 그래도 사용할 것이므로 별도로 더 이상 훈련을 하지 않는다는 옵션을 줍니다 이는 trainable = False로 선택할 수 있습니다 
```py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten

model = Sequential()
e = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=max_len, trainable=False)
model.add(e)
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model.fit(X_train, y_train, epochs=100, verbose=2)
```

### 2) 사전 훈련된 Word2Vec 사용하기

```py
import gensim
```
```py
urlretrieve("https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz", \
                           filename="GoogleNews-vectors-negative300.bin.gz")
word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)
```
구글의 사전 훈련된 Word2Vec 모델을 로드하여  word2vec_model에 저장합니
```py
print(word2vec_model.vectors.shape) # 모델의 크기 확인
```
```
(3000000, 300)
```
300의 차원을 가진 Word2Vec 벡터가 3,000,000개 있습니다
```py
# 단어 집합 크기의 행과 300개의 열을 가지는 행렬 생성. 값은 전부 0으로 채워진다.
embedding_matrix = np.zeros((vocab_size, 300))
np.shape(embedding_matrix)
```
```
(16, 300)
```
모든 값이 0으로 채워진 임베딩 행렬을 만들어줍니다 이번 문제의 단어는 총 16개개이므로 , 16x300의 크기를 가진 행렬을 만듭니다 
```py
def get_vector(word):
    if word in word2vec_model:
        return word2vec_model[word]
    else:
        return None
```
word2vec_model에서 특정 단어를 입력하면 해당 단어의 임베딩 벡터를 리턴받을텐데, 만약 word2vec_model에 특정 단어의 임베딩 벡터가 없다면 None을 리턴하도록 합니다

```py
for word, index in tokenizer.word_index.items():
    # 단어와 맵핑되는 사전 훈련된 임베딩 벡터값
    vector_value = get_vector(word)
    if vector_value is not None:
        embedding_matrix[index] = vector_value
```
단어집합으로부터 단어를 1개씩 호출하여 word2vec_model에 해당 단어의 임베딩 벡터값이 존재하는지 확인합니다 만약 None이 아니라면 존재한다는 의미이므로 임베딩 행렬에 해당 단어의 인덱스 위치의 행에 임베딩의 값을 지정합니다 

이렇게 되면 현재 풀고자하는 문제의 16개의 단어와 맵핑되는 임베딩 행렬이 완성됩니다 

제대로 맵핑이 됐는지 기존 word2vec_model에 저장되어 있던 단어 `nice`의 임베딩 벡터값을 확인해봅시다
```py
print(word2vec_model['nice'])
```
```
[ 0.15820312  0.10595703 -0.18945312  0.38671875  0.08349609 -0.26757812
  0.08349609  0.11328125 -0.10400391  0.17871094 -0.12353516 -0.22265625
  ... 중략 ...
 -0.16894531 -0.08642578 -0.08544922  0.18945312 -0.14648438  0.13476562
 -0.04077148  0.03271484  0.08935547 -0.26757812  0.00836182 -0.21386719]
```
이 단어 `nice`는 헌재 단어 집합에서 몇 번 인덱스를 가지는지 확인해보겠습니다 
```py
print('단어 nice의 정수 인덱스 :', tokenizer.word_index['nice'])
```
```
단어 nice의 정수 인덱스 : 1
```
1의 값을 가지므로 embedding_matirx의 1번 인덱스에는 단어 'nice'의 임베딩 벡터값이 있어야합니다 
```py
print(embedding_matrix[1])
```
```
[ 0.15820312  0.10595703 -0.18945312  0.38671875  0.08349609 -0.26757812
  0.08349609  0.11328125 -0.10400391  0.17871094 -0.12353516 -0.22265625
  ... 중략 ...
 -0.16894531 -0.08642578 -0.08544922  0.18945312 -0.14648438  0.13476562
 -0.04077148  0.03271484  0.08935547 -0.26757812  0.00836182 -0.21386719]
```
값이 word2vec_model에서 확인했던 것과 동일한 것을 확인할 수 있습니다 
이제 Embedding에 사전 훈련된 embedding_matrix를 입력으로 넣어주고 모델을 학습시켜보겠습니다
```py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten, Input

model = Sequential()
model.add(Input(shape=(max_len,), dtype='int32'))
e = Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=max_len, trainable=False)
model.add(e)
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model.fit(X_train, y_train, epochs=100, verbose=2)
```