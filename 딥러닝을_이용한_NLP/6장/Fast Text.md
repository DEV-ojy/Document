# 패스트텍스트(FastText)

단어를 벡터로 만드는 또 다른 방법으로는 페이스북에서 개발한 FastText가 있습니다 Word2Vec 이후에 나온 것이기 때문에, 메커니즘 자체는 Word2Vec의 확장이라고 볼 수 있습니다

Word2Vec와 FastText와의 가장 큰 차이점이라면 Word2Vec는 단어를 쪼개질 수 없는 단위로 생각한다면, FastText는 하나의 단어 안에도 여러 단어들이 존재하는 것으로 간주합니다

즉 내부 단어(subword)를 고려하여 학습합니다

