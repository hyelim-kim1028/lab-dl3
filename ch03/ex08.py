"""
MNIST 숫자 손글씨 데이터 신경망 구현
"""
import pickle
import numpy as np
from PIL import Image  #파이썬 이미지 라이브러리
from ch03.ex01 import sigmoid
from ch03.ex05 import softmax
from dataset.mnist import load_mnist


def init_network():
    """가중치 행렬들(W1, W2, W3, b1, b2, b3)을 생성"""
    # 교재의 저자가 만든 가중치 행렬(sample_weight.pkl)을 읽어 옴.
    with open('sample_weight.pkl', mode='rb') as file:
        network = pickle.load(file)
    print(network.keys())
    # W1, W2, W3, b1, b2, b3 shape 확인
    return network

def forward(netwrok, x):
    """
        forward propagation(순방향 전파)
        forward(각 가중치 행렬, x = 이미지 1장)
        파라미터 x: 이미지 한 개의 정보를 가지고 있는 배열(784,) / 사이즈만 있고, 1차원인 배열
        행이 1개이고 원소의 갯수가 784개인 데이터
        """
    # 가중치 행렬(weight matrices)
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    # 첫번쨰 은닉층
    a1 = x.dot(W1) + b1
    # 첫번째 은닉층에서 나오는 출력값 (z1)
    z1 = sigmoid(a1)
    # shorcut: z1 = sigmoid(x.dot(W1) + b1)

    # 두번째 은닉층
    a2 = z1.dot(W2) + b2
    z2 = sigmoid(a2)

    # 출력층
    a3 = z2.dot(W3) + b3
    y = softmax(a3)

    return y


def predict(network, X_test):
    """신경망에서 사용되는 가중치 행렬들과 테스트 데이터를 파라미터로 전달받아서,
    테스트 데이터의 예측값(배열)을 리턴.
    파라미터 X_test : 만개의 테스트 이미지들의 정보를 가지고 있는 배열"""
    # 이미지 1장에는 각각 [x1, x2 .. x784] 의 점들을 가지고 있다
    # 이 점들이 은닉층들을 거치고 나오면 -> 숫자 0~9사이에 놓일 확률들을 가지고 숫자를 맞추는 것
    # predict()은 위의 forward함수들을 이미지의 갯수만큼 반복해주면 됨

    y_pred = []
    for sample in X_test: # 테스트 세트의 각 이미지들에 대해서 반복
        # 이미지를 신경망에 전파(통과) 시켜서 어떤 숫자인지 확률을 계산
        sample_hat = forward(network, sample)
        # print('sample_hat', sample_hat)
        # 가장 큰 확률의 인덱스를 찾음 (우리가 원하는 것은 어떤 숫자인지 맞추는 것 => final output printed result의 중요도: 인덱스 >>>> 확률)
        # 최댓값의 인덱스를 넘겨주는 함수: argmax
        sample_pred = np.argmax(sample_hat) # 1차원 이상일 때: axis를 주어야한다. 하지만, 1차원 배열에서는 안줘도 됨 : there is no axis to be given
        y_pred.append(sample_pred) #예측값을 결과 리스트에 추가
    return np.array(y_pred)

def accuracy(y_true, y_pred):
    """테스트 데이터 레이블(y_true)과 테스트 데이터 예측값(y_pred)을 파라미터로 전달받아서,
    정확도(accuracy) = (정답 개수)/(테스트 데이터 개수) 를 리턴."""
    # result = (y_true == y_pred) # 10,000개 짜리 np.array를 비교하는 것 #list
                                  # 정답과 예측값의 비교 (bool) 결과를 저장한 배열
    # # result -> list of bool dtype
    # print(result[:10])
    # return np.mean(result) # 평균계산은 숫자여야한다 # True = 1, False = 0으로 대체된 후 평균 계산됨
    # (1 + 1 + ... 0 + ...) / 전체 개수
    # above lines are the same as:
    return np.mean(y_true == y_pred)

if __name__ == '__main__':
    # In real life: pre-processing -> 데이터 준비
    # 데이터 준비 (학습 세트, 테스트 세트)
                                                    # 파라미터들 설정
    (X_train, y_train), (X_test, y_test) = load_mnist(normalize = True, # normalize: normalize the data (byte = 8bit = 0 ~255 를 0~1사이의 값으로 바꾼다)
                                                      flatten = True, #차원을 줄여서 1줄로 볼 수 있게 한다
                                                      one_hot_label= False) # one_hot_label: enable to see the categorical data in a numerical dtype # changes the label (i.e. 5 (False) -> [0, 0, 0, 0, 0, 1, 0 , 0 , 0 ,0] (True) )
    print(X_train[0])
    print(y_train[0])

    # 신경망 가중치(와 편향, bias) 행렬들을 생성
    network = init_network()
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    print(f'W1: {W1.shape}, W2: {W2.shape}, W3: {W3.shape}')
    # W1: (784, 50), W2: (50, 100), W3: (100, 10)
    # W1의 입력층이 784개가 있고, [x1, x2 ... x783, x784] # 50개의 neuron이 있다 # 그러므로, 784개의 가중치가 필요
    #
    print(f'b1: {b1.shape}, b2: {b2.shape}, b3:{b3.shape}')

    # 테스트 이미지들의 예측랎
    y_pred = predict(network, X_test)
    print('y_pred: ', y_pred[:10], ', y_shape', y_pred.shape)
    print('y_test', y_test[:10]) # 실제값 # 오답 1개: 5 -> 6 으로 해석

    acc = accuracy(y_test, y_pred)
    print('정확도(accuracy) =', acc)
    # 우리가 for-loop을 만드는 것 보다 numpy에 내제되어 있는 함수의 기능을 사용하는 것이 훨씬 빠르다 (속도)

    # softmax와 argmax가 array(행렬 > 2차 행렬) 계산이 제대로 안된다규,,.?? what is dis

    # 예측이 틀린 첫번쨰 이미지: X_test[8]
    # img = X_test[8] # 그냥 볼 수 없다, the image had been normalized, image needs to be 2-dimensional (widthxheight) but is flattened to be 1-dimensional
    # 파이썬 이미지로 불 수 없다 # 되돌리기!!!!
    img = X_test[8] * 255 # 0~1 > 0~255 -> 역역정규화? (denormalization)
    img = img.reshape(28, 28) #1차원 배열 -> 2차원 배열
    img = Image.fromarray(img) #2차원 NumPy 배열을 이미지로 변환
    img.show()
    # 헷갈리게 생겼넹,,,,,,,,,,,,,-,,-









