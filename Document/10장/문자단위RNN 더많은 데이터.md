# 문자 단위 RNN(Char RNN) - 더 많은 데이터

앞서 한 것보다 더 많은 데이터 문자 단위 RNN을 구현합니다 


## 1.문자 단위 RNN(Char RNN)

먼저 필요한 도구들을 임포트 합니다 

```py
import torch
import torch.nn as nn
import torch.optim as optim
```

다음과 같이 임의의 샘플을 만듭니다 

### 1. 훈련 데이터 전처리하기

```py
sentence = ("if you want to build a ship, don't drum up people together to "
            "collect wood and don't assign them tasks and work, but rather "
            "teach them to long for the endless immensity of the sea.")
```

문자 집합을 생성하고, 각 문자에 고유한 정수를 부여합니다 

```py
char_set = list(set(sentence)) # 중복을 제거한 문자 집합 생성
char_dic = {c: i for i, c in enumerate(char_set)} # 각 문자에 정수 인코딩

print(char_dic) # 공백도 여기서는 하나의 원소
```
```
{'k': 0, 'o': 1, 'r': 2, 'a': 3, 'f': 4, 'b': 5, 'g': 6, 'w': 7, ',': 8, ' ': 9, 'h': 10, 'l': 11, "'": 
12, 'e': 13, '.': 14, 'd': 15, 's': 16, 'y': 17, 'u': 18, 't': 19, 'n': 20, 'i': 21, 'm': 22, 'c': 23, 'p': 24}
```

각 문자에 정수가 부여됐다면 총 25개의 문자가 존재합니다 문자 집합의 크기를 확인해봅시다 
```py
dic_size = len(char_dic)
print('문자 집합의 크기 : {}'.format(dic_size))
```
```
문자 집합의 크기 : 25
```

문자 집합의 크기는 25이며 입력을 원-핫벡터로 사용할 것이므로 이는 매 시점마다 들어갈 입력의 크기이기도 합니다 이제 하이퍼파라미터를 설정
hidden_size(은닉 상태의 크기)를 입력의 크기와 동일하게 줬는데, 이는 사용자의 선택으로 다른 값을 줘도 무방

그리고 sequence_length라는 변수를 선언했는데 우리가 앞서 만든 샘플을 10개 단위로 끊어서 샘플을 만들예정이기 때문입니다 
```py
# 하이퍼파라미터 설정
hidden_size = dic_size
sequence_length = 10  # 임의 숫자 지정
learning_rate = 0.1
```
