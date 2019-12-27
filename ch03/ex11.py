"""
mini-batch에서 사용되는 soft max 함수 만들기
"""
import pickle

import numpy as np

from ch03.ex01 import sigmoid
from dataset.mnist import load_mnist


def softmax(X):
    """
    두가지 가정:
    1) X - 1차원 [x_1, x_2, ... , x_n]
    2) X - 2차원 [[x_11, x_12, ..., x_1n],
                  [x_21, x_22, ..., x_2n]]
    """
    dimension = X.ndim
    if dimension == 1:
        m = np.max(X) #1차원 배열 전체의 max/최댓값를 계산
        X = X - m # 0보다 같거나 작은 숫자로 변환 (이유: exponential의 overflow를 방지)
            # broadcast (m -> scalar value, X is an array)
        y = np.exp(X)/np.sum(np.exp(X)) # np.exp(X) 1개의 값 / 전체의 합
        # 배열의 크기 그대로 나온다: 모든 원소당 값이 나온다 => 리스트
        # 여기서도 broadcast가 일어남
    elif dimension == 2:
        # 이미지별로 최댓값을 찾아줘야한다 (row 뱡향으로 들어가 있음, axis = 1)
        # m = np.max(X, axis =1).reshape(len(X), 1) # len(X): 2차원 리스트 X의 row의 개수
        # X = X - m
        # sum = np.sum(np.exp(X), axis =1).reshape((len(X),1))
        # y = np.exp(X)/ sum # sum 은 모든 원소의 sum 인데 우리가 필요한것 -> sum of each row => axis = 1 => (k,1) (k,)을 계산하는 꼴 => sum()이라는 변수를 새로 지정해서 계산
        # 에러: 2차원에서 행개수는 k개, 열의 개수는 n개이다.

        # (k,n) (k,) 의 행렬은 계산이 안된다
        # X = X - m이 가능해지려면 reshape 을 해줘야
        Xt = X.T  # X의 전치 행렬(transpose)
        m = np.max(Xt, axis=0)
        Xt = Xt - m
        y = np.exp(Xt) / np.sum(np.exp(Xt), axis=0)
        y = y.T
    return y

def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    z1 = sigmoid(x.dot(W1) + b1)  # 첫번째 은닉층 전파(propagation)
    z2 = sigmoid(z1.dot(W2) + b2)  # 두번째 은닉층 전파(propagation)
    y = softmax(z2.dot(W3) + b3)  # 출력층 전파(propagation)
    return y


def mini_batch(network, X, batch_size):
    y_pred = np.array([])  # 예측값들을 저장할 배열
    # batch_size 만큼씩 X의 데이터들을 나눠서 forward propagation(전파)
    for i in range(0, len(X), batch_size):
        X_batch = X[i:(i + batch_size)]
        y_hat = forward(network, X_batch)  # (batch_size, 10) shape의 배열
        predictions = np.argmax(y_hat, axis=1)
        # 각 row에서 최댓값의 인덱스 -> (batch_size,) 배열
        y_pred = np.append(y_pred, predictions)  # 예측값들을 결과 배열에 추가
    return y_pred  # (len(X),) shape의 배열


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


if __name__ == '__main__':
    np.random.seed(2020)
    # 1차원 softmax 테스트
    a = np.random.randint(10, size = 5)
    print(a)
    # softmax = 0~1사이의 숫자로 바꿔주고, 모두 더하면 1이 되고, 크기의 비율을? 적용해주고, 순서를 지켜준다
    print(softmax(a)) # 합해주면 거의 1이 됨

    # 2차원 softmax
    A = np.random.randint(10, size = (2,3))
                    # 10 = 0 부터 10보다 작은 양의 정수들 중에서 (2,3)의 모양을 갖게 랜덤한 숫자들을 생성
    print(A)
    print(softmax(A)) # 행끼리 더해서 1이 되어야한다

    # (Train/Test) 데이터 세트 로드.
    (X_train, y_train), (X_test, y_test) = load_mnist()
    print('X_test shape:', X_test.shape)  # (10000, 784)
    print('y_test shape:', y_test.shape)  # (10000,)
    print(X_test[0])
    print(y_test[0])

    # 신경망 생성 (W1, b1, ...)
    with open('sample_weight.pkl', 'rb') as file:
        network = pickle.load(file)
    print('network:', network.keys())
    print('W1:', network['W1'].shape)
    print('W2:', network['W2'].shape)
    print('W3:', network['W3'].shape)

    batch_size = 100
    y_pred = mini_batch(network, X_test, batch_size)
    print('true[:10]', y_test[:10])
    print('pred[:10]', y_pred[:10])
    print('true[-10:]', y_test[-10:])
    print('pred[-10:]', y_pred[-10:])

    # 정확도(accuracy) 출력
    acc = accuracy(y_test, y_pred)
    print('정확도:', acc)





















