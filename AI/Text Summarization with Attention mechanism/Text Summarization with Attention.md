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

