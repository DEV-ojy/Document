# 01. 자연어 처리 전처리 이해하기

자연어 처리는 일반적으로 토큰화, 단어 집합 생성, 정수 인코딩, 패딩, 벡터화의 과정을 거칩니다 그래서
이번챕터에선 이러한 전반적인 과정에 대해서 이해합니다 

## 1. 토큰화(Tokenization)

주어진 텍스트를 **단어 또는 문자 단위로** 자르는것을 토큰화라고 합니다 
영어의 경우 토큰화를 사용하는 도구는 대표적으로  spaCy와 NLTK가 있습니다 물론 
파이썬 기본 함수인 split으로 토큰화를 할 수도 있습니다

우선 영어에 대해서 토큰화 실습을 해봅시다 

```
en_text = "A Dog Run back corner near spare bedrooms"
```

### 1. spaCy 사용하기

```py
import spacy
spacy_en = spacy.load('en')

def tokenize(en_text):
    return [tok.text for tok in spacy_en.tokenizer(en_text)]
    
print(tokenize(en_text))
```
```
['A', 'Dog', 'Run', 'back', 'corner', 'near', 'spare', 'bedrooms']
```

### 2. NLTK 사용하기

```py
!pip install nltk

import nltk
nltk.download('punkt')

from nltk.tokenize import word_tokenize
print(word_tokenize(en_text))
```
```
['A', 'Dog', 'Run', 'back', 'corner', 'near', 'spare', 'bedrooms']
```
### 3. 띄어쓰기로 토큰화

```py
print(en_text.split())
```
```
['A', 'Dog', 'Run', 'back', 'corner', 'near', 'spare', 'bedrooms']
```

사실 영어의 경우에는 띄어쓰기 단위로 토큰화를 해도 단어들 간 구분이 꽤나 명확한편이기 때문에 
토큰화 작업이 수월하지만한국어의 경우 토큰화 작업이 훨씬 까다롭습니다 한국어는 조사,접사 등으로 
단순 띄어쓰기 단위로 나누면 같은 단어가
다른 단어로 인식되어서 단어 집합의 크기가 불필요하게 커지기 때문입니다 

### 4. 한국어 띄어쓰기 토큰화

```py
kor_text = "사과의 놀라운 효능이라는 글을 봤어. 그래서 오늘 사과를 먹으려고 했는데 사과가 썩어서 슈퍼에 가서 
사과랑 오렌지 사왔어"
print(kor_text.split())
```

```
['사과의', '놀라운', '효능이라는', '글을', '봤어.', '그래서',
'오늘','사과를', '먹으려고', '했는데', '사과가', '썩어서', '슈퍼에', 
'가서', '사과랑', '오렌지', '사왔어']
```

위의 예제에서는 '사과'란 단어가 총 4번 등장했는데 모두 '의', '를', '가', '랑' 등이
붙어있어 이를 제거해주지 않으면 기계는 전부 다른 단어로 인식하게 됩니

### 5. 형태소 토큰화
위와 같은 상황을 방지하기 위해서 한국어는 보편적으로 '형태소 분석기'로 토큰화를 합니다
여기서는 형태소 분석기 중에서 mecab을 사용해도록하겠습니다 아래의 커맨드로 colab에서 mecab을 설치해줍니다 

```py
!git clone https://github.com/SOMJANG/Mecab-ko-for-Google-Colab.git
%cd Mecab-ko-for-Google-Colab
!bash install_mecab-ko_on_colab190912.sh
```
```py
from konlpy.tag import Mecab
tokenizer = Mecab()
print(tokenizer.morphs(kor_text))
```
```
['사과', '의', '놀라운', '효능', '이', '라는', '글', '을', '봤', '어', '.',
'그래서', '오늘', '사과', '를', '먹', '으려고', '했', '는데', '사과', '가', '썩', '어서', 
'슈퍼', '에', '가', '서', '사과', '랑', '오렌지', '사', '왔', '어']
```

### 6. 문자 토큰화

```py
print(list(en_text))
```

```
['A', ' ', 'D', 'o', 'g', ' ', 'R', 'u', 'n', ' ', 'b', 'a', 'c', 'k', '
', 'c', 'o', 'r', 'n', 'e', 'r', ' ', 'n', 'e', 'a', 'r', ' ', 's', 'p', 'a', 'r', 'e', '
', 'b', 'e', 'd', 'r', 'o', 'o', 'm', 's']
```
