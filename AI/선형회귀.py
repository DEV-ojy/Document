# 실습을 위해 파이토치의 도구들을 임포트하는 기본셋팅을 진행
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 현재 실습하고 있는 파이썬 코드를 재실행해도 다음에도 같은 결과가 나오도록 랜덤 시드 (random seed)를 줍니다
torch.manual_seed(1)

# 변수 선언 
x_train = torch.FloatTensor([[1],[2],[3]])
y_train = torch.FloatTensor([[2],[4],[6]])

# 변수의 크기를 출력
print(x_train)
print(x_train.shape)

print(y_train)
print(y_train.shape)

# 가중치와 편향의 초기화
# 가중치 W를 0으로 초기화하고 학습을 통해 값이 변경되는 변수임을 명시

W = torch.zeros(1,requires_grad=True)
#가중치 W를 출력
print(W)

"
가중치 W가 0으로 초기화되어있으므로 0이 출력된 것을 확인할 수 있습니다 위에서 requires_grad=True가
인자로 주어진것을 확인할 수 있습니다 이는 이 변수는 학스을 통해 계속 값이 변경되는 변수음을 의미
"

# 마찬가지로 편향 b도 0으로 초기화 하고 학습을 통해 값이 변경되는 변수임을 명시한다
b = torch.zeros(1,requires_grad=True)
print(b)

# 가설 세우기
hypothesis = x_train * W+b
print(hypothesis)

# 비용 함수 선언하기

#앞서 배운 torch.mean으로 평균을 구한다
cost = torch.mean((hypothesis - y_train)** 2)
print(cost)

# 경사 하강법 구현하기
optimizer = optim.SGD([W,b], lr=0.01)

"

optimizer.zero_grad()를 실행하므로서 미분을 통해 얻은 기울기를 0으로 초기화한다 
기울기를 초기화해야만 새로운 가중치 편향에 대해서 새로운 기울기를 구할 수 있다 
그 다음 cost.backward() 함수를 호출하면 가중치 W와 편향 b에 대한 기울기가 계산 된다 
그 다음 경사 하강법 최적화 함수 optimizer의 .step() 함수를 호출하여 인수로 들어갔던 
W와 b에서 리턴되는 변수들의 기울기에 확습률 0.01을 곱하여 뻬줌으로서 업데이트한다

"
# gradient를 0으로 초기화
optimizer.zero_grad()

# 비용 함수를 미분하여 gradient 계산
cost.backward()

#W와 b를 업데이트
optimizer.step()

# 전체 코드
#데이터
x_train = torch.FloatTensor([[1],[2],[3]])
y_train = torch.FloatTensor([[2],[4],[6]])

#모델 초기화
W = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# optimizer 설정
optimizer = optim.SGD([W, b], lr=0.01)

nb_epochs = 1999 # 원하는만큼 경사 하강법을 반복

for epoch in range(nb_epochs + 1):

    # H(x) 계산
    hypothesis = x_train * W + b

    # cost 계산
    cost = torch.mean((hypothesis - y_train) ** 2)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 100번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} W: {:.3f}, b: {:.3f} Cost: {:.6f}'.format(
            epoch, nb_epochs, W.item(), b.item(), cost.item()
        ))
        
"
에포크는 전체 훈련 데이터가 학습에 한 번 사용된 주기를 말한다 이 실습의 경우 2,000번 수행
최종 훈련 결과를 보면 최적의 기울기 W는 2에 가깝고,b는 0에 가까운 것을 볼 수 있다 
현재 훈련 데이터가 x_train은 [[1],[2],[3]]이고 y_train은 [[2],[4],[6]]인 것을 감안하면 
실제 정답은 W가 2이고,b가 0인 H(x) =2x 이므로 거의 정답을 찾은 셈입니다
"

# optimizer.zero_grad()가 필요한 이유 
# -> 계속해서 미분값인 2가 누적되는 것을 볼수있다 그렇게때문에 optimizer.zero_grad()를 통해
#    계속 0으로 초기화시켜줘야한다

import torch
w = torch.tensor(2.0, requires_grad=True)

nb_epochs = 20
for epoch in range(nb_epochs + 1):

  z = 2*w

  z.backward()
  print('수식을 w로 미분한 값 : {}'.format(w.grad))

# torch.manual_seed()를 하는 이유

"
torch.manual_seed()를 사용한 프로그램의 결과는 다른 컴퓨터에서 실행시켜도 동일한 결과를 얻을 수 있습니다 
그 이유는 torch.manual_seed()는 난수 발생순서와 값을 동ㅇ리하게 보장해준다는 특징때문입니다 
우선 랜덤 시드가 3일때 두번 난수를 발생시켜보고 다른 랜덤 시드를 사용한 후에 다시 랜덤 시드를 3을 사용한다면 
난수 발생값이 동일하게 나오는지 보겠습니다
"
import torch

torch.manual_seed(3)
print('랜덤 시드가 3일 때')
for i in range(1,3):
  print(torch.randn(1))

torch.manual_seed(5)
print('랜덤 시드가 5일 때')
for i in range(1,3):
  print(torch.rand(1))

torch.manual_seed(3)
print('랜덤 시드가 다시 3일 때')
for i in range(1,3):
  print(torch.rand(1))

 
"
텐서에는 requires_grad라는 속성이 있습니다 
이것을 True로 설정하면 자동 미분 기능이 적용됩니다 선형회귀부터 신경망과 같은 복잡한 구조에서 
피라미터들이 모두 이 기능이 적용됩니다 requires_grad = True가 적용된 텐서에 연산을 하면, 
계산 그래프가 생성되면 backward 함수를 호출하면 그래프로부터 자동으로 미분이 계산된다
"
