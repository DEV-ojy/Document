# ELMo(Embeddings from Language Model)

ELMo는 2018년에 제안된 새로운 워드 임베딩 방법론입니다 ELMo는 Embeddings from Language Model의 약자이며 해석하면 `언어 모델로 하는 임베딩` 입니다 

ELMo의 가장 큰 특징은 사전 훈련된 언어 모델(Pre-trained language model)을 사용한다는 
점입니다 이는 ELMo의 이름에 LM이 들어간 이유 입니다 

## 1. ELMo(Embeddings from Language Model)

Bank라는 단어를 생각해봅시다 Bank Account(은행 계좌)와 River Bank(강둑)에서의 Bank는 전혀 다른 의미를 가지는데 Word2Vec이나 GloVe 등으로 표현된 임베딩 벡터들은 이를 제대로 반영하지 못한다는 단점이 있습니다 

그렇다면 같은 표기의 단어라도 문맥에 따라서 다르게 워드 임베딩을 할 수 있으며 자연어 처리의 성능이 더 올라갈 것입니다 단어를 임베딩하기 전에 전체 문장을 고려해서 임베딩을 하겠다는 것입니다 그래서 탄생한 것이 **문맥을 반영한 워드 임베딩(Contextualized Word Embeddin)** 입니다

## 2. biLM(Bidirectional Language Model)의 사전 훈련

우선 다음단어를 예측하는 작업인 언어 모델링을 상기해봅시다 아래의 그림은 은닉층이 2개인 일반적인 단방향 RNN 언어 모델의 언어 모델링을 보여줍니다 

![image](https://user-images.githubusercontent.com/80239748/140600893-2a48ba8d-c0f4-4de1-9301-dd4d044005e6.png)

RNN 언어 모델은 문장으로부터 단어 단위로 입력을 받는데, RNN 내부의 은닉 상태 ht는 시점이 지날수록 점점 업데이트되갑니다 이는 결과적으로 ht의 값이 문장의 문낵 정보를 점차적으로 반영한다고 말할 수 있습니다 

지금 설명하는 내용은 새로운 개념이 아니라 RNN의 기본 개념입니다 그런데 ELMo는 위의 그림의 순방향 RNN뿐만 아니라 위의 그림과는 반대 방향으로 문장을 스캔하는 역방향 RNN 또한 활용합니다 ELMo는 양쪽 방향의 언어 모델을 둘 다 활용한다고하여 이 언어 모델을 biLM(Bidirectional Language Model)이라고 합니다 

ELMo에서 말하는 biLM은 기본적으로 다층 구조(Multi-layer)를 전제로 합니다 은닉층이 최소 2개 이상이라는 의미입니다 아래의 그림은 은닉층이 2개인 순방향 언어 모델과 역방향 언어 모델의 모습을 보여줍니다 

![image](https://user-images.githubusercontent.com/80239748/140601074-95e1fc35-2794-423b-81ff-0248ea118884.png)

이때 biLM의 입력이 되는 워드 임베딩 방법으로는 이 책에서는 다루지 않은 char CNN이라는 방법을 사용합니다 이 임베딩 방법은 글자 단위로 계산되는데, 이렇게 하면 마치 서브단어의 정보를 참고하는 것처럼 문맥과 상관없이 dog란 단어와 doggy란 단어의 연광성을 찾아낼 수 있습니다 또한 이 방법은 OOV에도 견고한 다는 장점이 있습니다 

주의할 점은 양방향 RNN과 ELMo에서의 biLM은 다소 다릅니다 양방향 RNN은 순방향 RNN의 은닉 상태와 역방향의 RNN의 은닉 상태를 다음층의 입력으로 보내기 전에 연결시킵니다 biLM의 순방향 언어모델과 역방향 언어모델이 각각의 은닉 상태만을 다음 은닉층으로 보내며 훈련시킨 후에 ELMo표현으로 사용하기 위해서 은닉상태를 연결 시키는 것과는 다릅니다 

## 3. biLM의 활용

biLM이 훈련되었다면, 이제 ELMo가 사전 훈련된 biLM을 통해 입력 문장으로부터 단어를 임베딩하기 위한 과정을 보겠습니다 

![image](https://user-images.githubusercontent.com/80239748/140633286-feee4f36-d102-419b-9bce-696b5f47c321.png)

이 예제에서는 play란 단어가 임베딩이 되고 있다는 가정 하에 ELMo를 설명합니다 
play라는 단어를 임베딩 하기위해서 ELMo는 위의 점선의 삭가형 내부의 각 층의 결과값을 재료로 사용합니다

다시 말해 해당 시점의 BiLM의 각 층의 출력값을 가져옵니다 그리고 순방향 언어 모델과 역방향 언어 모델의 각 층의 출력값을 연결하고 추가 작업을 진행합니다 

![image](https://user-images.githubusercontent.com/80239748/140634241-aeabc2ca-1e34-4bb2-8c60-1a0d15f0ea1e.png)

이렇게 완성된 벡터를 ELMo표현이라고 합니다 지금까지는 ELMo표현을 얻기 위한 과정이고 이제 ELMo를 입력으로 사용하고 수행하고 싶은 텍스트 분류,질의 응답 시스템등의 자연어 처리 작업이 있을 것 입니다 

ELMo 표현은 기존의 임베딩 벡터와 함께 사용할 수 있습니다 우선 텍스트 분류 작업을 위해서 GloVe와 같은 기존의 방법론을 사용한 임베딩 벡터를 준비했다고 합시다 

이때,GloVe를 사용한 임베딩 벡터만 텍스트 분류 작업에 사용하는 것이 아니라 이렇게 준비된 ELMo 표현을 GloVe 임베딩 벡터와 연결해서 입력으로 사용할 수있습니다 

그리고 이때 ELMo 표현을 만드는데 사용되는 사전 훈련된 언어 모델의 가중치는 고정시킵니다 그리고 대신 위에서 사용한 s1,s2,s3와 y는 훈련 과정에서 학습 됩니다 

![image](https://user-images.githubusercontent.com/80239748/140715365-b53b2e24-ec63-41d8-9c48-85b8f5cf6581.png)

위의 그림은 ELMo 표현이 기존의 GloVe 등과 같은 임베딩 벡터와 함께 NLP 태스크의 입력이 되는 것을 보여줍니다