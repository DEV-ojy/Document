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




