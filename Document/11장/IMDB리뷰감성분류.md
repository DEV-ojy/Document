#   IMDB 리뷰 감성 분류하기(IMDB Movie Review Sentiment Analysis)

영화 사이트 IMDB의 리뷰 데이터는 리뷰에 대한 텍스트와 해당 리뷰가 긍정(1)인지 부정(0)인지 표시한 레이블로 구성된 데이터 입니다 


## 1.셋팅 하기 

우선 필요한 도구들을 임포트합니다 

```py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext import data, datasets
import random
```
랜덤 시드를 고정 

```py
SEED = 5
random.seed(SEED)
torch.manual_seed(SEED)
```
하이퍼파라미터들을 변수로 정의합니다 
```py
# 하이퍼파라미터
BATCH_SIZE = 64
lr = 0.001
EPOCHS = 10
```
현 환경에서 GPU를 사용 가능하면 GPU를 사용하고 CPU를 사용 가능하다면 CPU를 사용하도록 설정합니다  
```py
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
print("cpu와 cuda 중 다음 기기로 학습함:", DEVICE)
```
```
cpu와 cuda 중 다음 기기로 학습함: cuda
```

## 2. 토치텍스트를 이용한 전처리

여기서는 앞서 배운 토치텍스트를 사용하여 전처리를 진행합니다

### 1.데이터 로드하기 : torchtext.data
torchtext.data의 Field 클래스를 사용하여 영화 리뷰에 대한 객체 TEXT, 레이블을 위한 객체 LABEL을 생성
```py
TEXT = data.Field(sequential=True, batch_first=True, lower=True)
LABEL = data.Field(sequential=False, batch_first=True)
```
데이터 셋이 순차적인 데이터셋임을 알수 있도록 sequential인자값으로 True를 명시해줍니다 레이블은 단순한 클래스를 나타내는 숫자로
순차적인 데이터가 아니므로 False를 명시합니다 batch_first는 신경망에 입력되는 텐서의 첫번째 차원값이 batch_size가 되도록합니다
그리고 lower변수를 통해 텍스트 데이터 속 모든 영문알파벳이 소문자가 되도록 합니다 


### 2.데이터 로드 및 분할하기 : torchtext.datasets
torchtext.datasets을 통해 IMDB 리뷰 데이터를 다운로드 할수있습니다 데이터를 다운받는 동시에 훈련 데이터와 테스트 데이터를 분할하고, 각각 trainset, testset에 저장합니다

```py
# 전체 데이터를 훈련 데이터와 테스트 데이터를 8:2 비율로 나누기
trainset, testset = datasets.IMDB.splits(TEXT, LABEL)
```

텍스트와 레이블이 제데로 저장되었는지 확인하기 위해서  trainset.fields를 통해 trainset이 포함하는 각 요소를 확인해봅시다
```py
print('trainset의 구성 요소 출력 : ', trainset.fields)
```
```
trainset의 구성 요소 출력 :  {'text': <torchtext.data.field.Field object at 0x7f175ce3fe10>, 'label': <torchtext.data.field.Field object at 0x7f175ce3fe80>}
```
리뷰 데이터가 저장되어져 있는 text 필드와 레이블이 저장되어져 있는 label 필드가 존재합니다. testset.fields도 출력해봅니다
```py
print('testset의 구성 요소 출력 : ', testset.fields)
```
```
testset의 구성 요소 출력 :  {'text': <torchtext.data.field.Field object at 0x7f175ce3fe10>, 'label': <torchtext.data.field.Field object at 0x7f175ce3fe80>}
```
첫번째 룬련 샘플과 해당 샘플에 대한 레이블을 출력해보겠습니다 
```py
print(vars(trainset[0]))
```
```
{'text': ['if', 'you', 'like', 'jamie', 'foxx,(alvin', 'sanders),"date', 'from', 'hell",\'01,', 'you', 'will', 'love', 'his', 'acting', 'as', 'a', 'guy', 'who', 'never', 'gets', 'an', 'even', 'break', 'in', 'life', 'and', 'winds', 'up', 'messing', 'around', 'with', 'shrimp,', '(jumbo', 'size)', 'and', 'at', 'the', 'same', 'time', 'lots', 'of', 'gold', 'bars.', 'alvin', 'sanders', 'has', 'plenty', 'of', 'fbi', 'eyes', 'watching', 'him', 'and', 'winds', 'up', 'getting', 'hit', 'by', 'a', 'brick', 'in', 'the', 'jaw,', 'and', 'david', 'morse,(edgar', 'clenteen),', '"hack"', "'02", 'tv', 'series,', 'decides', 'to', 'zero', 'in', 'on', 'poor', 'alvin', 'and', 'use', 'him', 'as', 'a', 'so', 'called', 'fish', 'hook', 'to', 'attract', 'the', 'criminals.', 'there', 'is', 'lots', 'of', 'laughs,', 'drama,', 'cold', 'blood', 'killings', 'and', 'excellent', 'film', 'locations', 'and', 'plenty', 'of', 'expensive', 'cars', 'being', 'sent', 'to', 'the', 'junk', 'yard.', 'jamie', 'foxx', 'and', 'david', 'morse', 'were', 'outstanding', 'actors', 'in', 'this', 'film', 'and', 'it', 'was', 'great', 'entertainment', 'through', 'out', 'the', 'entire', 'picture.'],
'label': 'pos'}
```

'text' : []에서 대괄호 안에 위치한 단어들이 첫번째 IMDB 리뷰에 해당되면 'label' : []에서 대괄호 안의 단어가 첫번째 IMDB리뷰의 ㅣ레이블에 해당됩니다 여기서 'pos'는 positivie의 줄인말로 긍정을 의미합니다 

### 3. 단어 집합 만들기

이제 단어 집합을 만들어줍니다 단어 집합이란 중복을 제거한 총 단어들의 집합을 의미합니다 
```py
TEXT.build_vocab(trainset, min_freq=5) # 단어 집합 생성
LABEL.build_vocab(trainset)
```
위에서 min_freq는 학습 데이터에서 최소 5번 이상 등장한 단어만을 단어 집합에 추가하겠다는 의미입니다 이때 학습 데이터에서 5번 미만으로 등장한 단어는 Unkown이라는 의미에서 '<unk>'라는 토큰으로 대체됩니다 
  
단어 집합의 크기와 클래스의 개수를 변수에 저장하고 출력해봅니다 단어 집합의 크기란 결국 중복을 제거한 총 단어의 개수입니다  
```py
vocab_size = len(TEXT.vocab)
n_classes = 2
print('단어 집합의 크기 : {}'.format(vocab_size))
print('클래스의 개수 : {}'.format(n_classes))
```
```
단어 집합의 크기 : 46159
클래스의 개수 : 2
```
  
stoi로 단어와 각 단어의 정수 인덱스가 저장되어져 있는 딕셔너리 객체에 접근할 수 있습니다
```py
print(TEXT.vocab.stoi)
```
  
```
defaultdict(<function _default_unk_index at 0x7fb279f3cc80>, {'<unk>': 0, '<pad>': 1, 'the': 2, 'a': 3, 'and': 4, 'of': 5
... 중략 ...
'zoe,': 46150, 'zombies"': 46151, 'zombies)': 46152, 'zombified': 46153, 'zone.<br': 46154, 'zoolander': 46155, 'zwick': 46156, '{the': 46157, 'émigré': 46158})
```
  
unknown을 의미하는 <unk>라는 단어가 0번 인덱스로 부여되어져 있고, 그 외에도 수많은 단어들이 고유한 정수 인덱스가 부여된 것을 볼 수 있습니다 

### 4.데이터 로더 만들기 
  
훈련 데이터와 테스트 데이터는 분리하였지만, 이제 검증 데이터를 분리할 차례입니다 훈련 데이터를 다시 8:2로 분리하여 검증
데이터를 만들겠습니다 검증 데이터는 valset이라는 변수에 저장합니다 
  
```py
trainset, valset = trainset.split(split_ratio=0.8)
```

정리하면 훈련 데이터는 trainset, 테스트 데이터는 testeset, 검증 데이터는 valset에 저장되었습니다 
  
토치텍스트는 모든 텍스트를 배치 처리하는 것을 지원하고, 단어를 인덱스 번호로 대체하는 Bucketlterator를 제공합니다 
Bucketlterator는 batch_size,device,shuffle 등의 인자를 받습니다 BATCH_SIZE는 앞서 64로 설정했었습니다
```py
train_iter, val_iter, test_iter = data.BucketIterator.splits(
      (trainset, valset, testset), batch_size=BATCH_SIZE,
      shuffle=True, repeat=False)
```
이제 train_iter, val_iter, test_iter에는 샘플과 레이블이 64개 단위 묶음으로 저장됩니다. 64개씩 묶었을 때 총 배치의 개수가 몇 개가 되는지 출력해봅시다.
```py
print('훈련 데이터의 미니 배치의 개수 : {}'.format(len(train_iter)))
print('테스트 데이터의 미니 배치의 개수 : {}'.format(len(test_iter)))
print('검증 데이터의 미니 배치의 개수 : {}'.format(len(val_iter)))  
```
```
훈련 데이터의 미니 배치의 개수 : 313
테스트 데이터의 미니 배치의 개수 : 391
검증 데이터의 미니 배치의 개수 : 79  
```
첫번째 미니 배치의 크기를 확인해보겠습니다 
  
```py
batch = next(iter(train_iter)) # 첫번째 미니배치
print(batch.text.shape)  
```
```
torch.Size([64, 968]) 
``` 
  
첫번째 미니 배치의 크기는 64 x 968임을 확인할 수 있습니다 현재 fix_length를 정해주지 않았으므로 미니 배치 간 샘플들의 길이는 전부 상이합니다 가령, 두번째 미니 배치의 크기를 확인하면 또 길이가 다름을 확인할 수 있습니다 
```py
batch = next(iter(train_iter)) # 두번째 미니배치
print(batch.text.shape)  
```  
```
torch.Size([64, 873])
```  
  
두 개의 미니배치를 꺼내서 크기를 확인하였으므로 이미 꺼낸 두 개의 미니배치를 다시 담기위해 재로드해줍니다
```
train_iter, val_iter, test_iter = data.BucketIterator.splits(
        (trainset, valset, testset), batch_size=BATCH_SIZE,
        shuffle=True, repeat=False)
```
 
## 3.RNN 모델구현하기  
  
```py
  class GRU(nn.Module):
    def __init__(self, n_layers, hidden_dim, n_vocab, embed_dim, n_classes, dropout_p=0.2):
        super(GRU, self).__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.embed = nn.Embedding(n_vocab, embed_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.gru = nn.GRU(embed_dim, self.hidden_dim,
                          num_layers=self.n_layers,
                          batch_first=True)
        self.out = nn.Linear(self.hidden_dim, n_classes)

    def forward(self, x):
        x = self.embed(x)
        h_0 = self._init_state(batch_size=x.size(0)) # 첫번째 히든 스테이트를 0벡터로 초기화
        x, _ = self.gru(x, h_0)  # GRU의 리턴값은 (배치 크기, 시퀀스 길이, 은닉 상태의 크기)
        h_t = x[:,-1,:] # (배치 크기, 은닉 상태의 크기)의 텐서로 크기가 변경됨. 즉, 마지막 time-step의 은닉 상태만 가져온다.
        self.dropout(h_t)
        logit = self.out(h_t)  # (배치 크기, 은닉 상태의 크기) -> (배치 크기, 출력층의 크기)
        return logit

    def _init_state(self, batch_size=1):
        weight = next(self.parameters()).data
        return weight.new(self.n_layers, batch_size, self.hidden_dim).zero_()
```  

모델을 설계합니다
  
```py
model = GRU(1, 256, vocab_size, 128, n_classes, 0.5).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
```
모델 훈련 함수를 만듭니다.
```py
def train(model, optimizer, train_iter):
    model.train()
    for b, batch in enumerate(train_iter):
        x, y = batch.text.to(DEVICE), batch.label.to(DEVICE)
        y.data.sub_(1)  # 레이블 값을 0과 1로 변환
        optimizer.zero_grad()

        logit = model(x)
        loss = F.cross_entropy(logit, y)
        loss.backward()
        optimizer.step()  
```
모델 평가 함수를 만듭니다.
```py
def evaluate(model, val_iter):
    """evaluate model"""
    model.eval()
    corrects, total_loss = 0, 0
    for batch in val_iter:
        x, y = batch.text.to(DEVICE), batch.label.to(DEVICE)
        y.data.sub_(1) # 레이블 값을 0과 1로 변환
        logit = model(x)
        loss = F.cross_entropy(logit, y, reduction='sum')
        total_loss += loss.item()
        corrects += (logit.max(1)[1].view(y.size()).data == y.data).sum()
    size = len(val_iter.dataset)
    avg_loss = total_loss / size
    avg_accuracy = 100.0 * corrects / size
    return avg_loss, avg_accuracy  
```
모델을 훈련시킵니다     

```py
 best_val_loss = None
for e in range(1, EPOCHS+1):
    train(model, optimizer, train_iter)
    val_loss, val_accuracy = evaluate(model, val_iter)

    print("[Epoch: %d] val loss : %5.2f | val accuracy : %5.2f" % (e, val_loss, val_accuracy))

    # 검증 오차가 가장 적은 최적의 모델을 저장
    if not best_val_loss or val_loss < best_val_loss:
        if not os.path.isdir("snapshot"):
            os.makedirs("snapshot")
        torch.save(model.state_dict(), './snapshot/txtclassification.pt')
        best_val_loss = val_loss 
```
```
[Epoch: 1] val loss :  0.69 | val accuracy : 49.82
[Epoch: 2] val loss :  0.69 | val accuracy : 50.82
[Epoch: 3] val loss :  0.69 | val accuracy : 53.60
[Epoch: 4] val loss :  0.71 | val accuracy : 50.22
[Epoch: 5] val loss :  0.48 | val accuracy : 79.34
[Epoch: 6] val loss :  0.33 | val accuracy : 85.80
[Epoch: 7] val loss :  0.33 | val accuracy : 86.48
[Epoch: 8] val loss :  0.31 | val accuracy : 87.04
[Epoch: 9] val loss :  0.34 | val accuracy : 86.92
[Epoch: 10] val loss :  0.37 | val accuracy : 87.24  
``` 
