# cnn으로 IMDB 리뷰 분류하기 
from tensorflow.keras import datasets
from tensorflow.keras.preprocessing.sequence import pad_sequences
#우선 IMDB 리뷰 데이터를 받아로기 위한 datasets과 패딩을 위한 pad_sequences를 임포트

vocab_size = 10000
(X_train, y_train), (X_test, y_test) = datasets.imdb.load_data(num_words = vocab_size)
#최대 10,000개의 단어만을 허용하여 데이터를 받아온다

print(X_train[:5])
#각 샘플은 이미 정수 인코딩까지 전처리가 된 상태 

max_len = 200
X_train = pad_sequences(X_train,maxlen=max_len)
X_test = pad_sequences(X_test,maxlen=max_len)
#패딩으로 모든 샘플들의 길이를 200으로 맞춥니다 

print('X_train의 크기(shape) :',X_train.shape)
print('X_test의 크기(shape) :',X_test.shape)

print(y_train[:5])
#이진 분류를 수행할 것이므로 레이블에는 더 이상 전처리를 할 것이 없다

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dropout, Conv1D, GlobalMaxPooling1D, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
#모델을 설계하기위해 필요한 도구를 임포트

model = Sequential()
model.add(Embedding(vocab_size, 256))
model.add(Dropout(0.3))
model.add(Conv1D(256, 3, padding='valid', activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

#1D 합성곱 연산을 수행하되, 커널수는 256, 커널의 크기는 3을 사용합니다. 
#그리고 GlobalMaxPooling1D를 사용하고, 두 개의 밀집층으로 은닉층과 출력층을 설계합니다.
#검증 데이터의 손실(loss)이 증가하면, 과적합 징후이므로 검증 데이터 손실이 3회 증가하면 학습을 중단하는 조기 종료(EarlyStopping)를 사용합니다. 
#또한, ModelCheckpoint를 사용하여 검증 데이터의 정확도가 이전보다 좋아질 경우에만 모델을 저장하도록 합니다.

es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 3)
mc = ModelCheckpoint('best_model.h5', monitor = 'val_acc', mode = 'max', verbose = 1, save_best_only = True)

model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['acc'])
history = model.fit(X_train, y_train, epochs = 20, validation_data = (X_test, y_test), callbacks=[es, mc])

loaded_model = load_model('best_model.h5')
print("\n 테스트 정확도: %.4f" % (loaded_model.evaluate(X_test, y_test)[1]))
