# 시퀀스-투-시퀀스(Sequence-to-Sequence, seq2seq)

시퀀스-투-시퀀스는 입력된 시퀀스로부터 다른 도메인의 시퀀스를 출력하는 다양한 분야에서 사용되는 모델입니다 예를 들어 챗봇과 기계번역이 그러한 대표적인 예인데 입력 시퀀스와 출력 시퀀스를 각각 질문과 대답으로 구성하면 챗봇으로 만들 수 있고, 입력 시퀀스와 출력 시퀀스를 각각 입력 문장과 번역문장으로 만들면 번역기로 만들 수 있습니다 
그 외에도 내용 요약,STT등에서 쓰일수 있습니다 

이번에는 기계번역을 예제로 시퀀스-투-시퀀스를 설명해보겠습니다 

## 1. 시퀀스-투-시퀀스(Sequence-to-Sequence)

seq2seq는 번역기에서 대표적으로 사용되는 모델입니다 앞으로의 설명 방식은 내부가 보이지 않는 커다란 블랙 박스에서 점차적으로 확대해가는 방식으로 설명합니다 

고로 여기서 설명하는 내용의 대부분은 RNN 챕터에서 언급한 내용들입니다 단지 이것을 가지고 어떻게 조립했느냐에 따라서 seq2seq라는 구조가 만들어집니다

![image](https://user-images.githubusercontent.com/80239748/144242713-ca80a2de-372f-4848-8fdb-16a3073ea63f.png)

위의 그림은 seq2seq 모델로 만들어진 번역기가 'I am a student'라는 영어 문장을 입력받아서, 'je suis étudiant'라는 프랑스 문장을 출력하는 모습을 보여줍니다

그렇다면, seq2seq 모델 내부의 모습은 어떻게 구성되었을까요

![image](https://user-images.githubusercontent.com/80239748/144242827-4ff4271e-39b6-4aa8-8ee9-ebb2960eef61.png)

seq2seq는 크게 두 개로 구성된 아키텍처로 구성되는데, 바로 인코더와 디코더입니다 

인코더는 입력 문장의 모든 단어들을 순차적으로 입력받은 뒤에 마지막에 이 모든 단어 정보들을 압축해서 하나의 벡터로 만드는데, 이를 컨텍스트 벡터(context vector)라고 합니다

입력 문장의 정보가 하나의 컨텍스트 벡터로 모두 압축되면 인코더는 컨텍스트 벡터를 디코더로 전송합니다 디코더는 컨텍스트 벡터를 받아서 번역된 단어를 한 개씩 순차적으로 출력합니다

![image](https://user-images.githubusercontent.com/80239748/144242951-f94951cd-c252-45e8-9ed8-2099f48592e9.png)

컨텍스트 벡터에 대해서는 뒤에서 다시 언급하겠습니다 위의 그림에서는 컨텍스트 벡터를 4의 사이즈로 표현하였지만, 실제 현업에서 사용되는 seq2seq 모델에서는 보통 수백 이상의 차원을 갖고있습니다 

이제 인코더와 디코더의 내부를 좀 더 확대해보겠습니다.

![image](https://user-images.githubusercontent.com/80239748/144243111-70344edc-881b-41ee-a011-2324c9948913.png)

인코더 아키텍처와 디코더 아키텍처의 내부는 사실 두 개의 RNN 아키텍처 입니다 입력 문장을 받는 RNN 셀을 인코더라고 하고, 출력 문장을 출력하는 RNN 셀을 디코더라고 합니다

이번 챕터에서는 인코더의 RNN 셀을 주황색으로, 디코더의 RNN 셀을 초록색으로 표현합니다
물론, 성능 문제로 인해 실제로는 바닐라 RNN이 아니라 LSTM 셀 또는 GRU 셀들로 구성됩니다

우선 인코더를 자세히보면, 입력 문장은 단어 토큰화를 통해서 단어 단위로 쪼개지고 단어 토큰 각각은 RNN 셀의 각 시점의 입력이 됩니다

인코더 RNN 셀은 모든 단어를 입력받은 뒤에 인코더 RNN 셀의 마지막 시점의 은닉 상태를 디코더 RNN 셀로 넘겨주는데 이를 컨텍스트 벡터라고 합니다

텍스트 벡터는 디코더 RNN 셀의 첫번째 은닉 상태로 사용됩니다

디코더는 기본적으로 RNNLM(RNN Language Model)입니다 그래서 RNNLM의 개념을 기억하고 있다면 좀 더 이해하기 쉽지만, 기억하지 못 하더라도 다시 처음부터 설명할 것이므로 상관없습니다

디코더는 초기 입력으로 문장의 시작을 의미하는 심볼 <sos>가 들어갑니다 디코더는 <sos>가 입력되면, 다음에 등장할 확률이 높은 단어를 예측합니다 

첫번째 시점(time step)의 디코더 RNN 셀은 다음에 등장할 단어로 je를 예측하였습니다 첫번째 시점의 디코더 RNN 셀은 예측된 단어 je를 다음 시점의 RNN 셀의 입력으로 입력합니다 

그리고 두번째 시점의 디코더 RNN 셀은 입력된 단어 je로부터 다시 다음에 올 단어인 suis를 예측하고, 또 다시 이것을 다음 시점의 RNN 셀의 입력으로 보냅니다 디코더는 이런 식으로 기본적으로 다음에 올 단어를 예측하고, 그 예측한 단어를 다음 시점의 RNN 셀의 입력으로 넣는 행위를 반복합니다

이 행위는 문장의 끝을 의미하는 심볼인 <eos>가 다음 단어로 예측될 때까지 반복됩니다. 지금 설명하는 것은 테스트 과정 동안의 이야기입니다

seq2seq는 훈련 과정과 테스트 과정(또는 실제 번역기를 사람이 쓸 때)의 작동 방식이 조금 다릅니다 

훈련 과정에서는 디코더에게 인코더가 보낸 컨텍스트 벡터와 실제 정답인 상황인 <sos> je suis étudiant를 입력 받았을 때, je suis étudiant <eos>가 나와야 된다고 정답을 알려주면서 훈련합니다

반면 테스트 과정에서는 앞서 설명한 과정과 같이 디코더는 오직 컨텍스트 벡터와 <sos>만을 입력으로 받은 후에 다음에 올 단어를 예측하고, 그 단어를 다음 시점의 RNN 셀의 입력으로 넣는 행위를 반복합니다

즉, 앞서 설명한 과정과 위의 그림은 테스트 과정에 해당됩니다 이번에는 입, 출력에 쓰이는 단어 토큰들이 있는 부분을 좀 더 확대해보겠습니다.

![image](https://user-images.githubusercontent.com/80239748/144712429-52a72c15-3b1b-4cc1-a49d-80df29f98382.png)

기계는 텍스트보다 숫자를 잘 처리합니다 그리고 자연어 처리에서 텍스트를 벡터로 바꾸는 방법으로 워드 임베딩이 사용됩니다 
즉, seq2seq에서 사용되는 모든 단어들은 워드 임베딩을 통해 임베딩 벡터로서 표현된 임베딩 벡터입니다 위 그림은 모든 단어에 대해서 임베딩 과정을 거치게 하는 단계인 임베딩 층(embedding layer)의 모습을 보여줍니다

![image](https://user-images.githubusercontent.com/80239748/144712617-ee22ca19-fe59-48f6-8fd4-1423c9289f87.png)

예를 들어 I, am, a, student라는 단어들에 대한 임베딩 벡터는 위와 같은 모습을 가집니다 여기서는 그림으로 표현하고자 사이즈를 4로 하였지만, 보통 실제 임베딩 벡터는 수백 개의 차원을 가질 수 있습니다

이제 RNN 셀에 대해서 확대해보겠습니다

![image](https://user-images.githubusercontent.com/80239748/144712717-8f01d0a1-4eef-4293-83e7-8e3fb16c3c9d.png)

현재 시점(time step)을 t라고 할 때, RNN 셀은 t-1에서의 은닉 상태와 t에서의 입력 벡터를 입력으로 받고, t에서의 은닉 상태를 만듭니다

이때 t에서의 은닉 상태는 바로 위에 또 다른 은닉층이나 출력층이 존재할 경우에는 위의 층으로 보내거나, 필요없으면 값을 무시할 수 있습니다 

그리고 RNN 셀은 다음 시점에 해당하는 t+1의 RNN 셀의 입력으로 현재 t에서의 은닉 상태를 입력으로 보냅니다

RNN 챕터에서도 언급했지만, 이런 구조에서 현재 시점 t에서의 은닉 상태는 과거 시점의 동일한 RNN 셀에서의 모든 은닉 상태의 값들의 영향을 누적해서 받아온 값이라고 할 수 있습니다 

그렇기 때문에 앞서 우리가 언급했던 컨텍스트 벡터는 사실 인코더에서의 마지막 RNN 셀의 은닉 상태값을 말하는 것이며, 이는 입력 문장의 모든 단어 토큰들의 정보를 요약해서 담고있다고 할 수 있습니다

디코더는 인코더의 마지막 RNN 셀의 은닉 상태인 컨텍스트 벡터를 첫번째 은닉 상태의 값으로 사용합니다 

디코더의 첫번째 RNN 셀은 이 첫번째 은닉 상태의 값과, 현재 t에서의 입력값인 <sos>로부터, 다음에 등장할 단어를 예측합니다 
그리고 이 예측된 단어는 다음 시점인 t+1 RNN에서의 입력값이 되고, 이 t+1에서의 RNN 또한 이 입력값과 t에서의 은닉 상태로부터 t+1에서의 출력 벡터 

즉, 또 다시 다음에 등장할 단어를 예측하게 될 것입니다

이제 디코더가 다음에 등장할 단어를 예츨하는 부분을 확대해보도록 하겠습니다 

![image](https://user-images.githubusercontent.com/80239748/145803033-3f3b8bcc-0a04-4df8-91ee-38105caeffaa.png)

출력 단어로 나올 수 있는 단어들은 다양한 단어들이 있습니다 seq2seq 모델은 선택될 수 있는 모든 단어들로부터 하나의 단어를 골라서 예측해야합니다 

이를 예측하기 위해서 쓸 수 있는 함수로는 뭐가 있을까요 바로 `소프트맥스 함수`입니다

디코더에서 각 시점의 RNN 셀에서 출력 벡터가 나오면, 해당 벡터는 소프트맥스 함수를 통해 출력 시퀀스의 각 단어별 활률값을 반환하고 디코더는 출력 단어를 결정합니다 

지금까지 가장 기본적인 seq2seq에 대해서 배워보았습니다 

사실 seq2seq는 어떻게 구현하느냐에 따라서 충분히 더 복잡해질 수 있습니다 
컨텍스트 벡터를 디코더의 초기 은닉 상태로만 사용할 수도 있고, 거기서 더 나아가 컨텍스트 벡터를 디코더가 단어를 예측하는 매 시점마다 하나의 입력으로 사용할 수도 있으며 

거기서 더 나아가면 어텐션 메커니즘이라는 방법을 통해 지금 알고있는 컨텍스트 벡터보다 더욱 문맥을 반영할 수 있는 컨텍스트 벡터를 구하여 매 시점마다 하나의 입력으로 사용할 수도 있습니다

## 2. 글자 레벨 기계 번역기(Character-Level Neural Machine Translation) 구현하기

이제 seq2seq를 이용해서 기계 번역기를 만들어보도록 하겠습니다  
```
시작하기에 앞서 참고하면 좋은 게시물을 소개합니다. 인터넷에 케라스로 seq2seq를 구현하는 많은 유사 예제들이 나와있지만 대부분은 케라스 개발자 프랑수아 숄레의 블로그의 유명 게시물인 'sequence-to-sequence 10분만에 이해하기'가 원본입니다. 이번 실습 또한 해당 게시물의 예제에 많이 영향받았습니다
```

    해당 게시물 링크 : https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html

실제 성능이 좋은 기계 번역기를 구현하려면 정말 방대한 데이터가 필요합니다  
여기서는 방금 배운 seq2seq를 실습해보는 수준에서 아주 간단한 기계 번역기를 구축해보도록 하겠습니다

기계 번역기를 훈련시키기 위해서는 훈련 데이터로 병렬 코퍼스(parallel corpus)가 필요합니다 
병렬 코퍼스란, 두 개 이상의 언어가 병렬적으로 구성된 코퍼스를 의미합니다

    다운로드 링크 : http://www.manythings.org/anki


### 1) 병렬 코퍼스 데이터에 대한 이해와 전처리

우선 병령 코퍼스 데이터에 대한 이해를 해보겠습니다 병령 데이터라고 하면 앞서 수행한  태깅 작업의 데이터를 생각할 수 있지만, 앞서 수행한 태깅 작업의 병렬 데이터와 seq2seq가 사용하는 병렬 데이터는 성격이 조금 다릅니다

**태깅 작업의 병렬 데이터는 쌍이 되는 모든 데이터가 길이가 같았지만 여기서는 쌍이 된다고 해서 길이가 같지않습니다**

|한국어|->|영어|
|---|---|---|
|나는 학생이다|->|I am a student|

`나는 학생이다`라는 토큰의 개수가 2인 문장을 넣었을 때 `I am a student`라는 토큰의 개수가 4인 문장이 나오는 것과 같은 이치입니다 

seq2seq는 기본적으로 입력 시퀀스와 출력 시퀀스의 길이가 다를 수 있다고 가정합니다 
지금은 기본적으로 입력 시퀀스와 출력 시퀀스의 길이가 다를 수 있다고 가정합니다 

지금은기계 번역기가 예제지만 seq2seq의 또 다른 유명한 예제 중 하나인 챗봇을 만든다고 가정해보면, 대답의 길이가 질문의 길이와 항상 똑같아야 한다고하면 그 또한 이상합니다

    Watch me.           Regardez-moi !

여기서 사용할 fra.txt 데이터는 위와 같이 왼쪽의 영어 문장과 오른쪽의 프랑스어 문장 사이에 탭으로 구분되는 구조가 하나의 샘플입니다 그리고 이와 같은 형식의 약 16만개의 병령 문장 샘플을 포함하고 있습니다 

해당 데이터를 읽고 전처리를 진행해보겠습니다 
```py
import os
import shutil
import zipfile

import pandas as pd
import tensorflow as tf
import urllib3
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
```
```py
http = urllib3.PoolManager()
url ='http://www.manythings.org/anki/fra-eng.zip'
filename = 'fra-eng.zip'
path = os.getcwd()
zipfilename = os.path.join(path, filename)
with http.request('GET', url, preload_content=False) as r, open(zipfilename, 'wb') as out_file:       
    shutil.copyfileobj(r, out_file)

with zipfile.ZipFile(zipfilename, 'r') as zip_ref:
    zip_ref.extractall(path)
```
```py
lines = pd.read_csv('fra.txt', names=['src', 'tar', 'lic'], sep='\t')
del lines['lic']
print('전체 샘플의 개수 :',len(lines))
```

    전체 샘플의 개수 : 191954

전체 샘플의 개수는 총 약 19만2천개입니다 

```py
lines = lines.loc[:, 'src':'tar']
lines = lines[0:60000] # 6만개만 저장
lines.sample(10)
```
번역 문장에 해당되는 프랑스어 데이터는 앞서 배웠듯이 시작을 의미하는 심볼 <sos>과 종료를 의미하는 심볼 <eos>을 넣어주어야 합니다 
여기서는 <sos>와<eos> 대신 '\t'를 시작 심볼, '\n'을 종료 심볼로 간주하여 추가합니다 

```py
lines.tar = lines.tar.apply(lambda x : '\t '+ x + ' \n')
lines.sample(10)
```

```py
# 글자 집합 구축
src_vocab = set()
for line in lines.src: # 1줄씩 읽음
    for char in line: # 1개의 글자씩 읽음
        src_vocab.add(char)

tar_vocab = set()
for line in lines.tar:
    for char in line:
        tar_vocab.add(char)
```
글자 집합의 크기를 보겠습니다 

```py
src_vocab_size = len(src_vocab)+1
tar_vocab_size = len(tar_vocab)+1
print('source 문장의 char 집합 :',src_vocab_size)
print('target 문장의 char 집합 :',tar_vocab_size)
```
```
source 문장의 char 집합 : 79
target 문장의 char 집합 : 105
```

##### 2021.12.16