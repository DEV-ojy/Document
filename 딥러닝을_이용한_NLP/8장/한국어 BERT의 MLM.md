# 한국어 BERT의 마스크드 언어 모델(Masked Language Model) 실습

* 모든 BERT 실습은 Colab에서 진행한다고 가정합니다

사전 학습된 한국어 BERT를 이용하여 마스크드 언어 모델을 실습해봅시다 이번 실습을 위해서만이 아니라 앞으로 사전 학습된 BERT를 사용할 때는 transformers라는 패키지를 자주 사용하게 됩니다

    pip install transformers

## 1. 마스크드 언어 모델과 토크나이저

transformers 패키지를 사용하여 모델과 토크나이저를 로드합니다 BERT는 이미 누군가가 학습해둔 모델을 사용하는 것이므로 우리가 사용하는 모델과 토크나이저는 항상 맵핑 관계여야 합니다

예를 들어서 A라는 이름의 BERT를 사용하는데, B라는 이름의 BERT의 토크나이저를 사용하면 모델은 텍스트를 제대로 이해할 수 없습니다

    A라는 BERT의 토크나이저는 '사과'라는 단어를 36번으로 정수 인코딩하는 반면에, B라는 BERT의 토크나이저는 '사과'라는 단어를 42번으로 정수 인코딩하는 등 단어와 맵핑되는 정수 정보 자체가 다르기 때문입니다

klue/bert-base는 대표적인 한국어 BERT입니다. klue/bert-base의 마스크드 언어 모델과 klue/bert-base의 토크나이저를 로드해봅시다
```py
from transformers import TFBertForMaskedLM
from transformers import AutoTokenizer
```
TFBertForMaskedLM.from_pretrained('BERT 모델 이름')을 넣으면 [MASK]라고 되어있는 단어를 맞추기 위한 마스크드 언어 모델링을 위한 구조로 BERT를 로드합니다 다시 말해서 BERT를 마스크드 언어 모델 형태로 로드합니다

from_pt=True는 해당 모델이 기존에는 텐서플로우가 아니라 파이토치로 학습된 모델이었지만 이를 텐서플로우에서 사용하겠다라는 의미입니다

AutoTokenizer.from_pretrained('모델 이름')을 넣으면 해당 모델이 학습되었을 당시에 사용되었던 토크나이저를 로드합니다

```py
model = TFBertForMaskedLM.from_pretrained('klue/bert-base', from_pt=True)
tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
```

## 2. BERT의 입력

    '축구는 정말 재미있는 [MASK]다'
라는 임의의 문장이 있다고 가정해봅시다 이를 마스크드 언어 모델의 입력으로 넣으면, 마스크드 언어 모델은 [MASK]의 위치에 해당하는 단어를 예측합니다 마스크드 언어 모델의 예측 결과를 보기위해서 klue/bert-base의 토크나이저를 사용하여 해당 문장을 정수 인코딩해봅시다

```py
inputs = tokenizer('축구는 정말 재미있는 [MASK]다.', return_tensors='tf')
```
토크나이저로 변환된 결과에서 input_ids를 통해 정수 인코딩 결과를 확인할 수 있습니다

```py
print(inputs['input_ids'])
```
```
tf.Tensor([[   2 4713 2259 3944 6001 2259    4  809   18    3]], shape=(1, 10), dtype=int32)
```
토크나이저로 변환된 결과에서 token_type_ids를 통해서 문장을 구분하는 세그먼트 인코딩 결과를 확인할 수 있습니다
```py
print(inputs['token_type_ids'])
```
```
tf.Tensor([[0 0 0 0 0 0 0 0 0 0]], shape=(1, 10), dtype=int32)
```
현재의 입력은 문장이 두 개가 아니라 한 개이므로 여기서는 문장 길이만큼의 0 시퀀스를 얻습니다
약 문장이 두 개였다면 두번째 문장이 시작되는 구간부터는 1의 시퀀스가 나오게 되지만, 여기서는 해당되지 않습니다

토크나이저로 변환된 결과에서 attention_mask를 통해서 실제 단어와 패딩 토큰을 구분하는 용도인 어텐션 마스크를 확인할 수 있습니다

```py
print(inputs['attention_mask'])
```
```
tf.Tensor([[1 1 1 1 1 1 1 1 1 1]], shape=(1, 10), dtype=int32)
```
현재의 입력에서는 패딩이 없으므로 여기서는 문장 길이만큼의 1 시퀀스를 얻습니다 만약 뒤에 패딩이 있었다면 패딩이 시작되는 구간부터는 0의 시퀀스가 나오게 되지만, 여기서는 해당되지 않습니다

## 3. [MASK] 토큰 예측하기

FillMaskPipeline은 모델과 토크나이저를 지정하면 손쉽게 마스크드 언어 모델의 예측 결과를 정리해서 보여줍니다 FillMaskPipeline에 우선 앞서 불러온 모델과 토크나이저를 지정해줍니다 

```py
from transformers import FillMaskPipeline
pip = FillMaskPipeline(model=model, tokenizer=tokenizer)
```
이제 입력 문장으로부터 [MASK]의 위치에 들어갈 수 있는 상위 5개의 후보 단어들을 출력해봅시다
```
pip('축구는 정말 재미있는 [MASK]다.')
```
```py
[{'score': 0.8963505625724792,
  'sequence': '축구는 정말 재미있는 스포츠 다.',
  'token': 4559,
  'token_str': '스포츠'},
 {'score': 0.02595764957368374,
  'sequence': '축구는 정말 재미있는 거 다.',
  'token': 568,
  'token_str': '거'},
 {'score': 0.010033931583166122,
  'sequence': '축구는 정말 재미있는 경기 다.',
  'token': 3682,
  'token_str': '경기'},
 {'score': 0.007924391888082027,
  'sequence': '축구는 정말 재미있는 축구 다.',
  'token': 4713,
  'token_str': '축구'},
 {'score': 0.00784421805292368,
  'sequence': '축구는 정말 재미있는 놀이 다.',
  'token': 5845,
  'token_str': '놀이'}]
```
```
pip('어벤져스는 정말 재미있는 [MASK]다.')
```
```py
[{'score': 0.8382411599159241,
  'sequence': '어벤져스는 정말 재미있는 영화 다.',
  'token': 3771,
  'token_str': '영화'},
 {'score': 0.028275618329644203,
  'sequence': '어벤져스는 정말 재미있는 거 다.',
  'token': 568,
  'token_str': '거'},
 {'score': 0.017189407721161842,
  'sequence': '어벤져스는 정말 재미있는 드라마 다.',
  'token': 4665,
  'token_str': '드라마'},
 {'score': 0.014989694580435753,
  'sequence': '어벤져스는 정말 재미있는 이야기 다.',
  'token': 3758,
  'token_str': '이야기'},
 {'score': 0.009382619522511959,
  'sequence': '어벤져스는 정말 재미있는 장소 다.',
  'token': 4938,
  'token_str': '장소'}]
```

