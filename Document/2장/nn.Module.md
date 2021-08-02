# nn.Module로 구현하는 선형 회귀 

파이토치에서 이미 구현되어져 제공되고 있는 함수들을 불러오는 것으로 더 쉽게 선형 회귀 모델을 구현해보겟습니다

예를 들어 파이토치에서는 선형 회귀 모델이 nn.Linear()라고 함수로, 또 평균 제곱오차가 nn.functional.mse_loss()라는 함수로 구현되어져 있습니다
아래는 이번 실습에서 사용할 두 함수의 사용 예제를 간단히 보여줍니다 

```py
import torch.nn as nn
model = nn.Linear(input_dim, output_dim)

import torch.nn.functional as F
cost = F.mse_loss(prediction, y_train)
```

## 1.단순 선형 회귀 구현하기

우선 필요한 도구들을 임포트합니다 

```py
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)
```

```py
#데이터
x_train = torch.FloatTensor([[1],[2],[3]])
y_train = torch.FloatTensor([[2],[4],[6]])
```

데이터를 정의하였으니 이제 선형 회귀 모델을 구현할 차례입니다 
nn.Linear()는 입력의 차원,출력의 차원을 인수로 받습니다

```py
# 모델을 선언 및 초기화. 단순 선형 회귀이므로 input_dim=1, output_dim=1.
model = nn.Linear(1,1)
```

위 torch.nn.Linear 인자로 1,1을 사용하였습니다 하나의 입력 x에 대해서 하나의 출력y을 가지므로, 입력 차원과 출력 차원 
모두 1을 인수로 사용하였습니다 model에는 가중치 W와 편향 b가 저장되어져 있습니다 
이 값은 model.parameters()라는 함수를 사용하여 불러올 수 있는데, 한 번 출력해보겠습니다 

```py 
print(list(model.parameters()))
```

2개의 값이 출력되는데 첫번째 값이 W고, 두번째 값이 b에 해당됩니다 두 값 모두 현재는 랜덤 초기화가 되어져 있습니다
그리고 두 값 모두 학습의 대상이므로 requires_grad = True가 되어져 있는 것을 볼 수 있습니다 

이제 옵티마이저를 정의합니다 model.parameters() 사용하여 W와 b를 전달합니다 
학습률(learning rate)은 0.01로 정합니다 

```py
# optimizer 설정. 경사 하강법 SGD를 사용하고 learning rate를 의미하는 lr은 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=0.01) 

# 전체 훈련 데이터에 대해 경사 하강법을 2,000회 반복
nb_epochs = 2000
for epoch in range(nb_epochs+1):

    # H(x) 계산
    prediction = model(x_train)

    # cost 계산
    cost = F.mse_loss(prediction, y_train) # <== 파이토치에서 제공하는 평균 제곱 오차 함수

    # cost로 H(x) 개선하는 부분
    # gradient를 0으로 초기화
    optimizer.zero_grad()
    # 비용 함수를 미분하여 gradient 계산
    cost.backward() # backward 연산
    # W와 b를 업데이트
    optimizer.step()

    if epoch % 100 == 0:
    # 100번마다 로그 출력
      print('Epoch {:4d}/{} Cost: {:.6f}'.format(
          epoch, nb_epochs, cost.item()
      ))
```

학습이 완료되었습니다 Cost의 값이 매우 작습니다 W와 b의 값도 최적화가 되었는지 확인해봅시다 
x에 임의의 값 4를 넣어 모델이 예측하는 y의 값을 확인해보겠습니다

```py
#임의의 입력 4를 선언 
new_var = torch.FloatTensor([[4.0]])
#입력한 값 4에 대해서 예측값 y를 리턴받아서 pred_y에 저장
pred_y = model(new_var) #forward 연산
# y=2x 이므로 입력이 4라면 y가 8에 가까운 값이 나와야 제대로 학습이 된 것
print("훈련 후 입력이 4일 때의 예측값 :", pred_y) 
```
 
사실 이 문제의 정답은 y=2x가 정답이므로 y값이 8에 가까우면 W와 b의 값이 어느정도 최적화 된 것으로 볼 수 있습니다
실제로 예측된 y값은 7.9989로 8에 매우 가깝습니다 

이제 학습 후의 W와 b의 값을 출력해보겠습니다 

```py
print(list(model.parameters()))
```

W의 값이 2에 가깝고,b의 값이 0에 가까운 것을 볼 수 있습니다 


## 2.다중 선형 회귀 구현하기 

이제 nn.Linear()와 nn.functional.mse_loss()로 다중 선형 회귀를 구현해봅시다 사실 코드 자체는 달라지는 건 거의 없지만
nn.Linear()의 인자값과 학습률(learning rate)만 조절해주었습니다 

```py
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)
```

이제 데이터를 선언해줍니다 여기서는 3개의 x로부터 하나의 y를 예측하는 문제입니다 

```py
# 데이터
x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])
```

```py
#모델을 선언 및 초기화. 다중 선형 회귀이므로 input_dim=3,output_dim=1.
model = nn.Linear(3,1)
```



















