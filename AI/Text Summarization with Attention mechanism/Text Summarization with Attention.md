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

### 2) 데이터 정제학 

데이터에 중복 샘플이 있는지 확인

```py
print('Text 열에서 중복을 배제한 유일한 샘플의 수 :', data['Text'].nunique())
print('Summary 열에서 중복을 배제한 유일한 샘플의 수 :', data['Summary'].nunique())
```
```
Text 열에서 중복을 배제한 유일한 샘플의 수 : 88426
Summary 열에서 중복을 배제한 유일한 샘플의 수 : 72348
```
전체 데이터는 10만개의 샘플이 존재하지만, 실제로는 꽤 많은 원문이 중복되는 중복을 배제한 유일한 원문의 개수는 88,426개입니다 중복 샘플이 무려 약 1,200개나 있다는 의미입니다 

Summary는 중복이 더 많지만, 원문은 다르더라도 짧은 문장인 요약은 내용이 겹칠 수 있음을 가정하고 일단 두겠습니다 Summary의 길이 분포는 뒤에서 확인하겠습니다

```py
# text 열에서 중복인 내용이 있다면 중복 제거
data.drop_duplicates(subset=['Text'], inplace=True)
print("전체 샘플수 :", len(data))
```
```
전체 샘플수 : 88426
```
중복을 제거하여 88,426개의 샘플만 존재합니다 이제 Null 샘플이 존재하는지 확인해봅시다 
```py
print(data.isnull().sum())
```
```
Text       0
Summary    1
dtype: int64
```
Summary에서 1개의 Null 샘플이 남아있습니다. 이를 제거해줍니다
```py
# Null 값을 가진 샘플 제거
data.dropna(axis=0, inplace=True)
print('전체 샘플수 :',(len(data)))
```
```
전체 샘플수 : 88425
```
이제 남은 샘플 수는 88,425개입니다 지금까지는 불필요한 샘플의 수를 줄이기 위한 정제 과정이었습니다 이제 샘플 내부를 전처리해야 합니다 단어 정규화와 불용어 제거를 위해 각각의 참고 자료가 필요합니다

동일한 의미를 가졌지만 스펠링이 다른 단어들을 정규화하기 위한 사전을 만듭니다 이 사전은 아래의 링크를 참고하여 만들어진 사전입니다

링크 : https://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python

```py
# 전처리 함수 내 사용
contractions = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is", "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are", "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have", "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have"}
```
NLTK의 불용어를 저장하고 개수를 확인해봅시다
```py
# NLTK의 불용어
stop_words = set(stopwords.words('english'))
print('불용어 개수 :', len(stop_words))
print(stop_words)
```
```
불용어 개수 : 179
{'this', "doesn't", 'until', 'as', ... 중략 ... ,'whom', 'here', 'ma', "it's", 'am', 'your'}
```

전처리 함수를 설계합니다 

```py
# 전처리 함수
def preprocess_sentence(sentence, remove_stopwords = True):
    sentence = sentence.lower() # 텍스트 소문자화
    sentence = BeautifulSoup(sentence, "lxml").text # <br />, <a href = ...> 등의 html 태그 제거
    sentence = re.sub(r'\([^)]*\)', '', sentence) # 괄호로 닫힌 문자열  제거 Ex) my husband (and myself) for => my husband for
    sentence = re.sub('"','', sentence) # 쌍따옴표 " 제거
    sentence = ' '.join([contractions[t] if t in contractions else t for t in sentence.split(" ")]) # 약어 정규화
    sentence = re.sub(r"'s\b","",sentence) # 소유격 제거. Ex) roland's -> roland
    sentence = re.sub("[^a-zA-Z]", " ", sentence) # 영어 외 문자(숫자, 특수문자 등) 공백으로 변환
    sentence = re.sub('[m]{2,}', 'mm', sentence) # m이 3개 이상이면 2개로 변경. Ex) ummmmmmm yeah -> umm yeah

    # 불용어 제거 (Text)
    if remove_stopwords:
        tokens = ' '.join(word for word in sentence.split() if not word in stop_words if len(word) > 1)
    # 불용어 미제거 (Summary)
    else:
        tokens = ' '.join(word for word in sentence.split() if len(word) > 1)
    return tokens
```
여기서는 Text 열에서는 불용어를 제거하고, Summary 열에서는 불용어를 제거하지 않기로 결정했습니다 Summary를 입력으로 할 때는 두번째 인자를 0으로 줘서 불용어를 제거하지 않는 버전을 실행하겠습니다 

임의의 Text 문장과 Summary 문장을 만들어 전처리 함수를 통한 전처리 후의 결과를 확인해보겠습니다

```py
temp_text = 'Everything I bought was great, infact I ordered twice and the third ordered was<br />for my mother and father.'
temp_summary = 'Great way to start (or finish) the day!!!'
print(preprocess_sentence(temp_text))
print(preprocess_sentence(temp_summary, 0))
```
```
everything bought great infact ordered twice third ordered wasfor mother father
great way to start the day
```

우선 Text 열에 대해서 전처리를 수행하겠습니다 전처리 후에는 5개의 전처리 된 샘플을 출력합니다 

```py
# Text 열 전처리
clean_text = []
for s in data['Text']:
    clean_text.append(preprocess_sentence(s))
clean_text[:5]
```
```
['bought several vitality canned dog food products found good quality product looks like stew processed meat smells better labrador finicky appreciates product better',
 'product arrived labeled jumbo salted peanuts peanuts actually small sized unsalted sure error vendor intended represent product jumbo',
 'confection around centuries light pillowy citrus gelatin nuts case filberts cut tiny squares liberally coated powdered sugar tiny mouthful heaven chewy flavorful highly recommend yummy treat familiar story lewis lion witch wardrobe treat seduces edmund selling brother sisters witch',
 'looking secret ingredient robitussin believe found got addition root beer extract ordered made cherry soda flavor medicinal',
 'great taffy great price wide assortment yummy taffy delivery quick taffy lover deal']
```

이제 Summary열에 대해서 전처리를 수행하겠습니다 전처리 후에는 5개의 전처리 된 샘플을 출력합니다

```py
# Summary 열 전처리
clean_summary = []
for s in data['Summary']:
    clean_summary.append(preprocess_sentence(s, 0))
clean_summary[:5]
```
```
['good quality dog food',
 'not as advertised',
 'delight says it all',
 'cough medicine',
 'great taffy']
```
전처리 후의 결과를 다시 데이터프레임에 저장합니다 
```py
data['Text'] = clean_text
data['Summary'] = clean_summary
```

혹시 전처리 과정에서 빈 값이 생겼다면 Null 값으로 변경한 후에 Null 값을 가진 샘플이 생겼는지 확인합니다

```py
# 길이가 공백인 샘플은 NULL 값으로 변환
data.replace('', np.nan, inplace=True)
print(data.isnull().sum())
```
```
Text        0
Summary    70
dtype: int64
```
Summary 열에서 70개의 샘플이 Null 값을 가집니다 이 샘플들을 제거해주고, 전체 샘플수를 확인합니다

```py
data.dropna(axis = 0, inplace = True)
print('전체 샘플수 :',(len(data)))
```
```
전체 샘플수 : 88355
```

이제 Text 열과 Summary 열에 대해서 길이 분포를 확인해보겠습니다

```py
# 길이 분포 출력
text_len = [len(s.split()) for s in data['Text']]
summary_len = [len(s.split()) for s in data['Summary']]

print('텍스트의 최소 길이 : {}'.format(np.min(text_len)))
print('텍스트의 최대 길이 : {}'.format(np.max(text_len)))
print('텍스트의 평균 길이 : {}'.format(np.mean(text_len)))
print('요약의 최소 길이 : {}'.format(np.min(summary_len)))
print('요약의 최대 길이 : {}'.format(np.max(summary_len)))
print('요약의 평균 길이 : {}'.format(np.mean(summary_len)))

plt.subplot(1,2,1)
plt.boxplot(summary_len)
plt.title('Summary')
plt.subplot(1,2,2)
plt.boxplot(text_len)
plt.title('Text')
plt.tight_layout()
plt.show()

plt.title('Summary')
plt.hist(summary_len, bins=40)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()

plt.title('Text')
plt.hist(text_len, bins=40)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()
```

```
텍스트의 최소 길이 : 2
텍스트의 최대 길이 : 1235
텍스트의 평균 길이 : 38.792428272310566
요약의 최소 길이 : 1
요약의 최대 길이 : 28
요약의 평균 길이 : 4.010729443721352
```

![image](https://user-images.githubusercontent.com/80239748/138558103-b8d92304-b3d3-4981-84e7-1a70a0d0caa8.png)

![image](https://user-images.githubusercontent.com/80239748/138558112-cf84009f-1044-4dc5-bbcd-4b11d1140269.png)


원문 텍스트는 대체적으로 100이하의 길이를 가집니다 또한 평균 길이는 38입니다 
요약의 경우에는 대체적으로 15이하의 길이를 가지며 평균 길이는 4입니다 여기서 패딩의 길이를 정하겠습니다 평균 길이보다는 크게 잡아 각각 50과 8로 결정합니다 


```py
text_max_len = 50 
summary_max_len = 8
```

50과 8이라는 이 두 길이가 얼마나 많은 샘플들의 길이보다 큰지 확인해보겠습니다

```py
def below_threshold_len(max_len, nested_list):
  cnt = 0
  for s in nested_list:
    if(len(s.split()) <= max_len):
        cnt = cnt + 1
  print('전체 샘플 중 길이가 %s 이하인 샘플의 비율: %s'%(max_len, (cnt / len(nested_list))))
```

우선 Text열에 대해서 확인해봅시다 

```py
below_threshold_len(text_max_len, data['Text'])
```
```
전체 샘플 중 길이가 50 이하인 샘플의 비율: 0.7745119121724859
```

Text 열은 길이가 50 이하인 비율이 77%입니다 약 23%의 샘플이 길이 50보다 큽니다 
Summary열에 대해서 확인해봅시다 

```py
below_threshold_len(summary_max_len, data['Summary'])
```
```
전체 샘플 중 길이가 8 이하인 샘플의 비율: 0.9424593967517402
```
Summary 열은 길이가 8 이하인 경우 94%입니다 여기서는 정해준 최대 길이보다 큰 샘플들은 제거하겠습니다 

```py
data = data[data['Text'].apply(lambda x: len(x.split()) <= text_max_len)]
data = data[data['Summary'].apply(lambda x: len(x.split()) <= summary_max_len)]
print('전체 샘플수 :',(len(data)))
```
```
전체 샘플수 : 65818
```

seq2seq 훈련을 위해서는 디코더의 입력과 레이블에 시작 토큰과 종료 토큰을 추가할 필요가 있습니다 시작 토큰은 'sostoken' 종료 토큰은 'eostoken'이라 명명하고 앞뒤로 추가하겠습니다 

```py
# 요약 데이터에는 시작 토큰과 종료 토큰을 추가한다.
data['decoder_input'] = data['Summary'].apply(lambda x : 'sostoken '+ x)
data['decoder_target'] = data['Summary'].apply(lambda x : x + ' eostoken')
data.head()
```

인코더의 입력, 디코더의 입력과 레이블을 각각 저장해줍니다 

```py
encoder_input = np.array(data['Text'])
decoder_input = np.array(data['decoder_input'])
decoder_target = np.array(data['decoder_target'])
```

### 3) 데이터의 분리 

훈련 데이터와 테스트 데이터를 분리해보겠습니다 우선 순서가 섞인 정수 시퀀스를 만들어줍니다 

```py
indices = np.arange(encoder_input.shape[0])
np.random.shuffle(indices)
print(indices)
```
```
[29546 43316 24839 ... 45891 42613 43567]
```
이 정수 시퀀스 순서를 데이터의 샘플 순서로 정의해주면 샘플의 순서는 섞이게 됩니다 

```py
encoder_input = encoder_input[indices]
decoder_input = decoder_input[indices]
decoder_target = decoder_target[indices]
```
이제 섞인 데이터를 8:2의 비율로 훈련 데이터와 테스트 데이터로 분리해주겠습니다 
```py
n_of_val = int(len(encoder_input)*0.2)
print('테스트 데이터의 수 :',n_of_val)
```
```
테스트 데이터의 수 : 13163
```
테스트 데이터는 전체 데이터에서 20%에 해당하는 13,163개를 사용하겠습니다 
```py
encoder_input_train = encoder_input[:-n_of_val]
decoder_input_train = decoder_input[:-n_of_val]
decoder_target_train = decoder_target[:-n_of_val]

encoder_input_test = encoder_input[-n_of_val:]
decoder_input_test = decoder_input[-n_of_val:]
decoder_target_test = decoder_target[-n_of_val:]
```
```py
print('훈련 데이터의 개수 :', len(encoder_input_train))
print('훈련 레이블의 개수 :',len(decoder_input_train))
print('테스트 데이터의 개수 :',len(encoder_input_test))
print('테스트 레이블의 개수 :',len(decoder_input_test))
```
```
훈련 데이터의 개수 : 52655
훈련 레이블의 개수 : 52655
테스트 데이터의 개수 : 13163
테스트 레이블의 개수 : 13163
```
### 4) 정수 인코딩 

이제 기계가 텍스트를 숫자로 처리할 수 있도록 훈련 데이터와 테스트 데이터에 정수 인코딩을 수행해야 합니다 훈련 데이터에 대해서 단어 집합(vocaburary)을 만들어봅시다 

```py
src_tokenizer = Tokenizer()
src_tokenizer.fit_on_texts(encoder_input_train)
```

이제 단어 집합이 생성되는 동시에 각 단어에 고유한 정수가 부여되었습니다 이는 src_tokenizer.word_index에 저장되어져 있습니다 

여기서는 빈도수가 낮은 단어들은 자연어 처리에서 배제하고자 합니다 등장 빈도수가 7회 미만인 단어들이 이 데이터에서 얼만큼의 비중을 차지하는지 확인해봅시다

```py
threshold = 7
total_cnt = len(src_tokenizer.word_index) # 단어의 수
rare_cnt = 0 # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
total_freq = 0 # 훈련 데이터의 전체 단어 빈도수 총 합
rare_freq = 0 # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

# 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
for key, value in src_tokenizer.word_counts.items():
    total_freq = total_freq + value

    # 단어의 등장 빈도수가 threshold보다 작으면
    if(value < threshold):
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + value

print('단어 집합(vocabulary)의 크기 :',total_cnt)
print('등장 빈도가 %s번 이하인 희귀 단어의 수: %s'%(threshold - 1, rare_cnt))
print('단어 집합에서 희귀 단어를 제외시킬 경우의 단어 집합의 크기 %s'%(total_cnt - rare_cnt))
print("단어 집합에서 희귀 단어의 비율:", (rare_cnt / total_cnt)*100)
print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq)*100)
```
```
단어 집합(vocabulary)의 크기 : 32031
등장 빈도가 6번 이하인 희귀 단어의 수: 23779
단어 집합에서 희귀 단어를 제외시킬 경우의 단어 집합의 크기 8252
단어 집합에서 희귀 단어의 비율: 74.23745746308263
전체 등장 빈도에서 희귀 단어 등장 빈도 비율: 3.393443023084609
```
등장 빈도가 threshold 값인 7회 미만 즉, 6회 이하인 단어들은 단어 집합에서 무려 70% 이상을 차지합니다 하지만, 실제로 훈련 데이터에서 등장 빈도로 차지하는 비중은 상대적으로 적은 수치인 3.39%밖에 되지 않습니다 

여기서는 등장 빈도가 6회 이하인 단어들은 정수 인코딩 과정에서 배제시키고자 합니다 위에서 이를 제외한 단어 집합의 크기를 8,233으로 계산했는데, 저자는 깔끔한 값을 선호하여 이와 비슷한 값으로 단어 집합의 크기를 8000으로 제한하겠습니다

```
src_vocab = 8000
src_tokenizer = Tokenizer(num_words = src_vocab) 
src_tokenizer.fit_on_texts(encoder_input_train)

# 텍스트 시퀀스를 정수 시퀀스로 변환
encoder_input_train = src_tokenizer.texts_to_sequences(encoder_input_train) 
encoder_input_test = src_tokenizer.texts_to_sequences(encoder_input_test)
```

이제 레이블에 해당하는 요약 데이터에 대해서도 수행하겠습니다.
```py
tar_tokenizer = Tokenizer()
tar_tokenizer.fit_on_texts(decoder_input_train)
```
이제 단어 집합이 생성되는 동시에 각 단어에 고유한 정수가 부여되었습니다 이는 tar_tokenizer.word_index에 저장되어져 있습니다 

등장 빈도수가 6회 미만인 단어들이 이 데이터에서 얼만큼의 비중을 차지하는지 확인해봅시다
```py
threshold = 6
total_cnt = len(tar_tokenizer.word_index) # 단어의 수
rare_cnt = 0 # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
total_freq = 0 # 훈련 데이터의 전체 단어 빈도수 총 합
rare_freq = 0 # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

# 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
for key, value in tar_tokenizer.word_counts.items():
    total_freq = total_freq + value

    # 단어의 등장 빈도수가 threshold보다 작으면
    if(value < threshold):
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + value

print('단어 집합(vocabulary)의 크기 :',total_cnt)
print('등장 빈도가 %s번 이하인 희귀 단어의 수: %s'%(threshold - 1, rare_cnt))
print('단어 집합에서 희귀 단어를 제외시킬 경우의 단어 집합의 크기 %s'%(total_cnt - rare_cnt))
print("단어 집합에서 희귀 단어의 비율:", (rare_cnt / total_cnt)*100)
print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq)*100)
```
```
단어 집합(vocabulary)의 크기 : 10510
등장 빈도가 5번 이하인 희귀 단어의 수: 8128
단어 집합에서 희귀 단어를 제외시킬 경우의 단어 집합의 크기 2382
단어 집합에서 희귀 단어의 비율: 77.33587059942911
전체 등장 빈도에서 희귀 단어 등장 빈도 비율: 5.896286343062141
```
등장 빈도가 5회 이하인 단어들은 단어 집합에서 약 77%를 차지합니다 하지만, 실제로 훈련 데이터에서 등장 빈도로 차지하는 비중은 상대적으로 매우 적은 수치인 5.89%밖에 되지 않습니다

이 단어들은 정수 인코딩 과정에서 배제시키겠습니다

```
tar_vocab = 2000
tar_tokenizer = Tokenizer(num_words = tar_vocab) 
tar_tokenizer.fit_on_texts(decoder_input_train)
tar_tokenizer.fit_on_texts(decoder_target_train)
```
```
# 텍스트 시퀀스를 정수 시퀀스로 변환
decoder_input_train = tar_tokenizer.texts_to_sequences(decoder_input_train) 
decoder_target_train = tar_tokenizer.texts_to_sequences(decoder_target_train)
decoder_input_test = tar_tokenizer.texts_to_sequences(decoder_input_test)
decoder_target_test = tar_tokenizer.texts_to_sequences(decoder_target_test)
```

정수 인코딩이 정상 진행되었는지 훈련 데이터에 대해서 5개의 샘플을 출력해봅시다.
```py
print(decoder_input_train[:5])
```
```
[[1, 687], [1, 53, 21, 182, 1162, 240], [1, 6, 480, 113, 278, 181], [1, 15, 108, 215], [1, 54, 178, 21]]
```
```py
print(decoder_target_train[:5])
```
```
[[687, 2], [53, 21, 182, 1162, 240, 2], [6, 480, 113, 278, 181, 2], [15, 108, 215, 2], [54, 178, 21, 2]]
```

### 5) 빈 샘플 제거 

전체 데이터에서 빈도수가 낮은 단어가 삭제되었다는 것은 빈도수가 낮은 단어만으로 구성되었던 샘플들은 이제 빈 샘플이 되었다는것을 의미합니다 

요약문에서 길이가 0이 된 샘플들의 인덱스를 받아옵시다 주의할 점은 요약문인 decoder_input에는 sostoken 또는 decoder_target에는 eostoken이 추가된 상태이고, 이 두 토큰은 모든 샘플에서 등장하므로 빈도수가 샘플수와 동일하여 단어 집합 제한에도 삭제 되지 않습니다

그래서 이제 길이가 0이 된 요약문의 실질적 길이는 1입니다 decoder_input에는 sostoken, decoder_target에는 eostoken만 남았을 것이기 때문입니다

```py
drop_train = [index for index, sentence in enumerate(decoder_input_train) if len(sentence) == 1]
drop_test = [index for index, sentence in enumerate(decoder_input_test) if len(sentence) == 1]
```

훈련 데이터와 데스트 데이터에 대해서 요약문의 길이가 1인 경우 인덱스를 각각 drop_train과 drop_test에 저장하였습니다 이 샘플들을 모두 삭제하고자 합니다 

```py
print('삭제할 훈련 데이터의 개수 :',len(drop_train))
print('삭제할 테스트 데이터의 개수 :',len(drop_test))
```
```
삭제할 훈련 데이터의 개수 : 1235
삭제할 테스트 데이터의 개수 : 337
```

삭제 후의 개수는 다음과 같습니다 

```py
encoder_input_train = np.delete(encoder_input_train, drop_train, axis=0)
decoder_input_train = np.delete(decoder_input_train, drop_train, axis=0)
decoder_target_train = np.delete(decoder_target_train, drop_train, axis=0)

encoder_input_test = np.delete(encoder_input_test, drop_test, axis=0)
decoder_input_test = np.delete(decoder_input_test, drop_test, axis=0)
decoder_target_test = np.delete(decoder_target_test, drop_test, axis=0)

print('훈련 데이터의 개수 :', len(encoder_input_train))
print('훈련 레이블의 개수 :',len(decoder_input_train))
print('테스트 데이터의 개수 :',len(encoder_input_test))
print('테스트 레이블의 개수 :',len(decoder_input_test))
```
```
훈련 데이터의 개수 : 51420
훈련 레이블의 개수 : 51420
테스트 데이터의 개수 : 12826
테스트 레이블의 개수 : 12826
```





