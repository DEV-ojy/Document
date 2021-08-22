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
urllib.request.urlretrieve("https://raw.githubusercontent.com/GaoleMeng/RNN-and-FFNN-textClassification/
master/ted_en-20160408.xml", filename="ted_en-20160408.xml")
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
To me, the irony about the Facit story is hearing about the Facit engineers,
who had bought cheap, small electronic calculators in Japan that they used 
to double-check their calculators.
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

## 2.한국어 Word2Vec 만들기

이번에는 위키피디아 한국어 덤프 파일을 다운받아서 한국어로 Word2Vec을 직접 진행해보도록 하겠습니다


### 1) 위키피디아 한국어 덤프 파일 다운로드

https://dumps.wikimedia.org/kowiki/latest/

위 링크에는 많은 위키피디아 덤프 파일들이 존재합니다 그중에서 사용할 데이터는 kowiki-latest-pages-articles.xml.bz2 파일입니다 

### 2) 위키피디아 익스트랙터 다운로드

해당 파일을 모두 다운로드 받았다면 위키피디아 덤프 파일을 텍스트 형식으로 변환시켜주는 오픈소스인 '위키피디아 익스트랙터'를 사용할 것입니다 git clone 명령어를 통해 다운로드 받을 수 있습니다

```py
git clone "https://github.com/attardi/wikiextractor.git"  
```

### 3) 위키피디아 한국어 덤프 파일 변환
위키피디아 익스트랙터와 위키피디아 한국어 덤프 파일을 동일한 디렉토리 경로에 두고 아래 명령어를 실행하면 위키피디아 덤프파일 텍스트 파일로 변환됩니다 

```py
python WikiExtractor.py kowiki-latest-pages-articles.xml.bz2 
```

### 4) 훈련 데이터 만들기

우선 AA 디렉토리 안의 모든 파일인 wiki00 ~ wiki90에 대해서 wikiAA.txt로 통합해보도록 하겠습니다. 프롬프트에서 아래의 커맨드를 수행합니다 (윈도우 환경 기준)

```py
copy AA디렉토리의 경로\wiki* wikiAA.txt

copy AB디렉토리의 경로\wiki* wikiAB.txt
copy AC디렉토리의 경로\wiki* wikiAC.txt
copy AD디렉토리의 경로\wiki* wikiAD.txt
copy AE디렉토리의 경로\wiki* wikiAE.txt
copy AF디렉토리의 경로\wiki* wikiAF.txt

copy 현재 디렉토리의 경로\wikiA* wiki_data.txt
```

### 5) 훈련 데이터 전처리 하기

```py
f = open('wiki_data.txt 파일의 경로', encoding="utf8")
# 예를 들어 위도우 바탕화면에서 작업한 저자의 경우
# f = open(r'C:\Users\USER\Desktop\wiki_data.txt', encoding="utf8")
```

우선 파일을 불러왔습니다 

이제 본격적으로 Word2Vec을 위한 학습 데이터를 만들어보겠습니다
```py
from konlpy.tag import Okt  
okt=Okt()
fread = open('wiki_data.txt 파일의 경로', encoding="utf8")
# 파일을 다시 처음부터 읽음.
n=0
result = []

while True:
    line = fread.readline() #한 줄씩 읽음.
    if not line: break # 모두 읽으면 while문 종료.
    n=n+1
    if n%5000==0: # 5,000의 배수로 While문이 실행될 때마다 몇 번째 While문 실행인지 출력.
        print("%d번째 While문."%n)
    tokenlist = okt.pos(line, stem=True, norm=True) # 단어 토큰화
    temp=[]
    for word in tokenlist:
        if word[1] in ["Noun"]: # 명사일 때만
            temp.append((word[0])) # 해당 단어를 저장함

    if temp: # 만약 이번에 읽은 데이터에 명사가 존재할 경우에만
      result.append(temp) # 결과에 저장
fread.close()
```

```py
print('총 샘플의 개수 : {}'.format(len(result))
```

```
총 샘플의 개수 : 2466209
```

### 6) Word2Vec 훈련시키기

학습을 다했다면 이제 임의의 입력 단어로부터 유사한 단어들을 구해봅시다

```py
from gensim.models import Word2Vec
model = Word2Vec(result, size=100, window=5, min_count=5, workers=4, sg=0)

model_result1=model.wv.most_similar("대한민국")
print(model_result1)
```

```
[('한국', 0.6331368088722229), ('우리나라', 0.5405941009521484), ('조선민주주의인민공화국', 0.5400398969650269), ('정보통신부', 0.49965575337409973), ('고용노동부', 0.49638330936431885), ('경남', 0.47878748178482056), ('국내', 0.4761977791786194), ('국무총리실', 0.46891751885414124), ('공공기관', 0.46730121970176697), ('관세청', 0.46708711981773376)]
```


```py
model_result2=model.wv.most_similar("어벤져스")
print(model_result2)
```

```
[('스파이더맨', 0.8560965657234192), ('아이언맨', 0.8376990556716919), ('데어데블', 0.7797115445137024), ('인크레더블', 0.7791407108306885), ('스타트렉', 0.7752881050109863), ('엑스맨', 0.7738450765609741), ('슈퍼맨', 0.7715340852737427), ('어벤저스', 0.7453964948654175), ('슈퍼히어로', 0.7452991008758545), ('다크나이트', 0.7413955926895142)]
```

```py
model_result3=model.wv.most_similar("반도체")
print(model_result3)
```

```
[('전자부품', 0.7620885372161865), ('실리콘', 0.7574189901351929), ('집적회로', 0.7497618198394775), ('웨이퍼', 0.7465146780014038), ('태양전지', 0.735146164894104), ('트랜지스터', 0.7293091416358948), ('팹리스', 0.7275552749633789), ('디램', 0.722482442855835), ('전자제품', 0.712360143661499), ('그래핀', 0.7025551795959473)]
```
