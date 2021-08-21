# 영어/한국어 Word2Vec 훈련시키기

영어와 한국어 훈련 데이터에 대해서 Word2Vec을 학습해보겠습니다 gensim 패키지에서  Word2Vec은 이미 구현되어져 있으므로, 
별도로 Word2Vec을 구현할 필요없이 손쉽게 훈련시킬 수 있습니다

## 1. 영어 Word2Vec 만들기
이번에는 영어 데이터를 다운로드 받아 직접 Word2Vec 작업을 진행해보도록 하겠습니다 파이썬의 gensim패키지에는 Word2Vec을 지원하고
있어 gensim 패키지를 이용하면 손쉽게 단어를 임베딩 벡터로 변환시킬 수 있습니다 영어로 된 코퍼스를 다운받아 전처리를 수행하고
전처리한 데이터를 바탕으로 Word2Vec 작업을 진행하겠습니다

우선 필요한 도구를 임포트합니다

```py
import nltk
nltk.download('punkt')

import urllib.request
import zipfile
from lxml import etree
import re
from nltk.tokenize import word_tokenize, sent_tokenize
```

### 1) 훈련 데이터 이해하기
링크 : https://wit3.fbk.eu/get.php?path=XML_releases/xml/ted_en-20160408.zip&filename=ted_en-20160408.zip

위 zip파일의 압출을 풀면  ted_en-20160408.xml이라는 파일이 있습니다 
여기서는 파이썬 코드를 통해 자동으로 xml파일을 다운로드 받겠습니다
```py
# 데이터 다운로드
urllib.request.urlretrieve("https://raw.githubusercontent.com/GaoleMeng/RNN-and-FFNN-textClassification/master/ted_
en-20160408.xml", filename="ted_en-20160408.xml")
```
아래의 해당 파일은 xml 문법으로 작성되어 있어 자연어를 얻기 위해서는 전처리가 필요하다고 합니다 
얻자고한느 실직적 데이터는 영어문장으로만 구성된 내용을 담고 있는  <content>와 </content> 사이의 내용입니다
전처리 작업을 통해 xml문법들은 제고하고 해당 데이터만 가져와야합니다 
```
<file id="1">
  <head>
<url>http://www.ted.com/talks/knut_haanaes_two_reasons_companies_fail_and_how_to_avoid_them</url>
       <pagesize>72832</pagesize>
... xml 문법 중략 ...
<content>
Here are two reasons companies fail: they only do more of the same, or they only do what's new.
To me the real, real solution to quality growth is figuring out the balance between two activities:
... content 내용 중략 ...
To me, the irony about the Facit story is hearing about the Facit engineers, who had bought cheap, small electronic calculators in Japan that they used to double-check their calculators.
(Laughter)
... content 내용 중략 ...
(Applause)
</content>
</file>
<file id="2">
    <head>
<url>http://www.ted.com/talks/lisa_nip_how_humans_could_evolve_to_survive_in_space<url>
... 이하 중략 ...
```

### 3)훈련 데이터 전처리하기
위 데이터를 위한 전처리 코드는 아래와 같습니다 
```py
targetXML=open('ted_en-20160408.xml', 'r', encoding='UTF8')
# 저자의 경우 윈도우 바탕화면에서 작업하여서 'C:\Users\USER\Desktop\ted_en-20160408.xml'이 해당 파일의 경로.  
target_text = etree.parse(targetXML)
parse_text = '\n'.join(target_text.xpath('//content/text()'))
# xml 파일로부터 <content>와 </content> 사이의 내용만 가져온다.

content_text = re.sub(r'\([^)]*\)', '', parse_text)
# 정규 표현식의 sub 모듈을 통해 content 중간에 등장하는 (Audio), (Laughter) 등의 배경음 부분을 제거.
# 해당 코드는 괄호로 구성된 내용을 제거.

sent_text = sent_tokenize(content_text)
# 입력 코퍼스에 대해서 NLTK를 이용하여 문장 토큰화를 수행.

normalized_text = []
for string in sent_text:
     tokens = re.sub(r"[^a-z0-9]+", " ", string.lower())
     normalized_text.append(tokens)
# 각 문장에 대해서 구두점을 제거하고, 대문자를 소문자로 변환.

result = []
result = [word_tokenize(sentence) for sentence in normalized_text]
# 각 문장에 대해서 NLTK를 이용하여 단어 토큰화를 수행.

print('총 샘플의 개수 : {}'.format(len(result)))
```
```
총 샘플의 개수 : 273424
```
```py
for line in result[:3]: # 샘플 3개만 출력
    print(line)
```
```
['here', 'are', 'two', 'reasons', 'companies', 'fail', 'they', 'only', 'do', 'more', 'of', 'the', 'same', 'or', 'they', 'only', 'do', 'what', 's', 'new']
['to', 'me', 'the', 'real', 'real', 'solution', 'to', 'quality', 'growth', 'is', 'figuring', 'out', 'the', 'balance', 'between', 'two', 'activities', 'exploration', 'and', 'exploitation']
['both', 'are', 'necessary', 'but', 'it', 'can', 'be', 'too', 'much', 'of', 'a', 'good', 'thing']
```

### 3) Word2Vec 훈련시키기
```py
from gensim.models import Word2Vec, KeyedVectors
model = Word2Vec(sentences=result, size=100, window=5, min_count=5, workers=4, sg=0)
```
여기서 Word2Vec의 하이퍼파라미터값은 다음과 같습니다
size = 워드 벡터의 특징 값. 즉, 임베딩 된 벡터의 차원
window = 컨텍스트 윈도우 크기
min_count = 단어 최소 빈도 수 제한 (빈도가 적은 단어들은 학습하지 않는다.)
workers = 학습을 위한 프로세스 수
sg = 0은 CBOW, 1은 Skip-gram



```py
model_result = model.wv.most_similar("man")
print(model_result)
```
```
[('woman', 0.842622697353363),('guy', 0.8178728818893433), ('boy', 0.7774451375007629), ('lady', 0.7767927646636963), 
('girl', 0.7583760023117065), ('gentleman', 0.7437191009521484), ('soldier', 0.7413754463195801), 
('poet', 0.7060446739196777), ('kid', 0.6925194263458252), ('friend', 0.6572611331939697)]
```

### 4) Word2Vec 모델 저장하고 로드하기
공들여 학습한 모델을 언제든 나중에 다시 사용할 수 있도록 컴퓨터 파일로 저장하고 
```py
model.wv.save_word2vec_format('./eng_w2v') # 모델 저장
loaded_model = KeyedVectors.load_word2vec_format("eng_w2v") # 모델 로드

model_result = loaded_model.most_similar("man")
print(model_result)
```
```
[('woman', 0.842622697353363), ('guy', 0.8178728818893433), ('boy', 0.7774451375007629), 
('lady', 0.7767927646636963), ('girl', 0.7583760023117065), ('gentleman', 0.7437191009521484), 
('soldier', 0.7413754463195801), ('poet', 0.7060446739196777), ('kid', 0.6925194263458252), ('friend', 0.6572611331939697)]
```
