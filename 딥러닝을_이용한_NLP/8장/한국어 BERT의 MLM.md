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
















