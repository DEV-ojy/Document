# 패스트텍스트(FastText)

단어를 벡터로 만드는 또 다른 방법으로는 페이스북에서 개발한 FastText가 있습니다 Word2Vec 이후에 나온 것이기 때문에, 메커니즘 자체는 Word2Vec의 확장이라고 볼 수 있습니다

Word2Vec와 FastText와의 가장 큰 차이점이라면 Word2Vec는 단어를 쪼개질 수 없는 단위로 생각한다면, FastText는 하나의 단어 안에도 여러 단어들이 존재하는 것으로 간주합니다

즉 내부 단어(subword)를 고려하여 학습합니다

## 1. 내부 단어(subword)의 학습

FastText에서는 각 단어는 글자 단위 n-gram의 구성으로 취급합니다  n을 몇으로 결정하는지에 따라서 단어들이 얼마나 분리되는지 결정됩니다

예를 들어서 n을 3으로 잡은 트라이그램(tri-gram)의 경우, apple은 app, ppl, ple로 분리하고 이들을 벡터로 만듭니다 더 정확히는 시작과 끝을 의미하는 <, >를 도입하여 아래의 5개 내부 단어(subword) 토큰을 벡터로 만듭니다

```py
# n = 3인 경우
<ap, app, ppl, ple, le> 
```
그리고 여기에 추가적으로 하나를 더 벡터화하는데, 기존 단어에 <, 와 >를 붙인 토큰입니다
```py
# 특별 토큰
<apple>
```
다시 말해 n = 3인 경우, FastText는 단어 apple에 대해서 다음의 6개의 토큰을 벡터화하는 것입니다
```py
# n = 3인 경우
<ap, app, ppl, ple, le>, <apple>
```
그런데 실제 사용할 때는 n의 최소값과 최대값으로 범위를 설정할 수 있는데, 기본값으로는 각각 3과 6으로 설정되어져 있습니다 

다시 말해 최소값 = 3, 최대값 = 6인 경우라면, 단어 apple에 대해서 FastText는 아래 내부 단어들을 벡터화합니다
```py
# n = 3 ~ 6인 경우
<ap, app, ppl, ppl, le>, <app, appl, pple, ple>, <appl, pple>, ..., <apple>
```
여기서 내부 단어들을 벡터화한다는 의미는 저 단어들에 대해서 Word2Vec을 수행한다는 의미입니다

위와 같이 내부 단어들의 벡터값을 얻었다면, 단어 apple의 벡터값은 저 위 벡터값들의 총 합으로 구성합니다
```
apple = <ap + app + ppl + ppl + le> + <app + appl + pple + ple> + <appl + pple> + , ..., +<apple>
```

그리고 이런 방법은 Word2Vec에서는 얻을 수 없었던 강점을 가집니다

## 2. 모르는 단어(Out Of Vocabulary, OOV)에 대한 대응

FastText의 인공 신경망을 학습한 후에는 데이터 셋의 모든 단어의 각 n-gram에 대해서 워드 임베딩이 됩니다 

이렇게 되면 장점은 데이터 셋만 충분한다면 위와 같은 내부 단어(Subword)를 통해 모르는 단어(Out Of Vocabulary, OOV)에 대해서도 다른 단어와의 유사도를 계산할 수 있다는 점입니다

가령, FastText에서 birthplace(출생지)란 단어를 학습하지 않은 상태라고 해봅시다
하지만 다른 단어에서 birth와 place라는 내부 단어가 있었다면, FastText는 birthplace의 벡터를 얻을 수 있습니다

이는 모르는 단어에 제대로 대처할 수 없는 Word2Vec, GloVe와는 다른 점입니다

## 3. 단어 집합 내 빈도 수가 적었던 단어(Rare Word)에 대한 대응

Word2Vec의 경우에는 등장 빈도 수가 적은 단어(rare word)에 대해서는 임베딩의 정확도가 높지 않다는 단점이 있었습니다 참고할 수 있는 경우의 수가 적다보니 정확하게 임베딩이 되지 않는 경우입니다

하지만 FastText의 경우, 만약 단어가 희귀 단어라도, 그 단어의 n-gram이 다른 단어의 n-gram과 겹치는 경우라면, Word2Vec과 비교하여 비교적 높은 임베딩 벡터값을 얻습니다

FastText가 노이즈가 많은 코퍼스에서 강점을 가진 것 또한 이와 같은 이유입니다 모든 훈련 코퍼스에 오타(Typo)나 맞춤법이 틀린 단어가 없으면 이상적이겠지만, 실제 많은 비정형 데이터에는 오타가 섞여있습니다

그리고 오타가 섞인 단어는 당연히 등장 빈도수가 매우 적으므로 일종의 희귀 단어가 됩니다

즉, Word2Vec에서는 오타가 섞인 단어는 임베딩이 제대로 되지 않지만 FastText는 이에 대해서도 일정 수준의 성능을 보입니다

예를 들어 단어 apple과 오타로 p를 한 번 더 입력한 appple의 경우에는 실제로 많은 개수의 동일한 n-gram을 가질 것입니다

## 4. 실습으로 비교하는 Word2Vec Vs. FastText

간단한 실습을 통해 Word2Vec와 FastText의 차이를 비교해보도록 하겠습니다 

### 1) Word2Vec

우선 이전 챕터의 전처리 코드와 Word2Vec 학습 코드를 그대로 수행했음을 가정하겠습니다

입력 단어에 대해서 유사한 단어를 찾아내는 코드에 이번에는 electrofishing이라는 단어를 넣어보겠습니다

```py
model.wv.most_similar("electrofishing")
```
해당 코드는 작동하지 않고 이런 에러를 발생시킵니다
```
KeyError: "word 'electrofishing' not in vocabulary"
```
에러 메시지는 단어 집합(Vocabulary)에 electrofishing이 존재하지 않는다고 합니다 이처럼 Word2Vec는 학습 데이터에 존재하지 않는 단어

즉, 모르는 단어에 대해서는 임베딩 벡터가 존재하지 않기 때문에 단어의 유사도를 계산할 수 없습니다
### 2) FastText

이번에는 전처리 코드는 그대로 사용하고, Word2Vec 학습 코드만 FastText 학습 코드로 변경하여 실행해봅시다

```py
from gensim.models import FastText
model = FastText(result, size=100, window=5, min_count=5, workers=4, sg=1)
```

학습이 진행되었다면, 이제 electrofishing에 대해서 유사 단어를 찾아보도록 하겠습니다

```py
model.wv.most_similar("electrofishing")
```
```
[('electrolux', 0.7934642434120178), ('electrolyte', 0.78279709815979), ('electro', 0.779127836227417), ('electric', 0.7753111720085144), ('airbus', 0.7648627758026123), ('fukushima', 0.7612422704696655), ('electrochemical', 0.7611693143844604), ('gastric', 0.7483425140380859), ('electroshock', 0.7477173805236816), ('overfishing', 0.7435552477836609)]
```

Word2Vec는 학습하지 않은 단어에 대해서 유사한 단어를 찾아내지 못 했지만, FastText는 유사한 단어를 계산해서 출력하고 있음을 볼 수 있습니다
