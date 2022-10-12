import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model


# 시퀀스 데이터만 사용하여 학습

def train():
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

#   

    # 라벨명 추출
    with open("classset.txt", 'r', encoding="UTF-8") as f:
        actions = f.read().splitlines()

    data = []
    
    # 시퀀스 데이터 로드하여 하나로 합침
    for i in range(len(actions)):
        data.append(np.load(f'dataset/seq_{i}_{actions[i]}.npy'))
        
    data = np.concatenate(data)

    data.shape

#
    
    # 마지막 값에 라벨값이 저장되어 있으므로 분리
    x_data = data[:, :, :-1]
    labels = data[:, 0, -1]

    print(x_data.shape)
    print(labels.shape)

#
    # one-hot encoding
    y_data = to_categorical(labels, num_classes=len(actions))
    y_data.shape

#
    # training set validation set으로 나눔
    x_data = x_data.astype(np.float32)
    y_data = y_data.astype(np.float32)

    x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.1, random_state=2021)

    print(x_train.shape, y_train.shape)
    print(x_val.shape, y_val.shape)

#
    # 모델 정의
    model = Sequential([
        LSTM(64, activation='relu', input_shape=x_train.shape[1:3]),
        Dense(32, activation='relu'),
        Dense(len(actions), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    model.summary()

#
    # 200번 학습
    history = model.fit(
        x_train,
        y_train,
        # batch_size=12,
        validation_data=(x_val, y_val),
        epochs=200,
        callbacks=[
            ModelCheckpoint('./models/model2_1.0.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto'),
            ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=50, verbose=1, mode='auto')
        ]
    )

#
    # 학습 완료 그래프
    # 초록,파랑은 training이랑 validation accuracy
    # 노랑, 빨강은 loss
    # accuracy가 100%가 된것을 볼 수 있음
    fig, loss_ax = plt.subplots(figsize=(16, 10))
    acc_ax = loss_ax.twinx()

    loss_ax.plot(history.history['loss'], 'y', label='train loss')
    loss_ax.plot(history.history['val_loss'], 'r', label='val loss')
    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')
    loss_ax.legend(loc='upper left')

    acc_ax.plot(history.history['acc'], 'b', label='train acc')
    acc_ax.plot(history.history['val_acc'], 'g', label='val acc')
    acc_ax.set_ylabel('accuracy')
    acc_ax.legend(loc='upper left')

    plt.show()

#

    model = load_model('models/model2_1.0.h5')

    y_pred = model.predict(x_val)

    confusion_matrix(np.argmax(y_val, axis=1), np.argmax(y_pred, axis=1))

