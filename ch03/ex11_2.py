"""
ex12 - HW for 12262019
ex12_1 - teacher's solution for ex12

1) (Train/Test) 데이터 세트 로드,
2) 신경망생성 -> network
3) batch size가 100인 mini-batch 모델을 만들기
4) 정확도 출력
"""
import pickle

from ch03.ex01 import sigmoid
from ch03.ex05 import softmax
from dataset.mnist import load_mnist
import numpy as np

def forward(network,x): # x는 1차원이 아닌 2차원이라고 가정한다
    # 가중치 행렬
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    z1 = sigmoid(x.dot(W1) + b1)    # 첫번째 은닉층 (propagation)
    z2 = sigmoid(z1.dot(W2) + b2)   # 두번째 은닉층 (propagation)
    y = softmax(z2.dot(W3) + b3)    # output layer

    return y
    # 계산된 값만 달라질 뿐, 순서는 유지가 된다

# mini-batch function 만들기
# network (전파 시킬 수 있게끔 가중치를 준다 -> 앞으로 나아가렴,,,)
# y_pred = [예측한 숫자들의 배열]

def mini_batch(network, X, batch_size):
    # X를 신경망 네트워크로 batch_size만큼씩 보내서 예측값 생성
    y_pred = [] # 예측값을 저장할 리스트
    # batch-size 만큼씩 X의 데이터들을 나눠서 forward propagation 시킨다
    # range 만들기:
    for i in range(0, len(X), batch_size): # 0에서 ~까지 100단위로 건너뛴다 # i = 0, 100, 200. 300 ...
        X_batch = X[i:(i+batch_size)]
        y_hat = forward(network, X_batch) # (batch_size,10)의 이차원 배열
        # max값의 index를 찾는다 -> argmax()함수 사용
        predictions = np.argmax(y_hat, axis = 1)  # 행별로 인덱스를 준다 # 각 row에서 최댓값 인덱스
                                                  # (batch_size,) 배열
        y_pred = np.append(y_pred, predictions) #예측값들을 결과 배열에 추가
                            # predictions의 1차원 배열을 y_pred의 1차원 배열에 추가하겠다
    return y_pred #(len(X),) shape의 배열

# 정확도 (accuracy) 출력
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)
    # boolean dtype의 np.array(y_pred)


if __name__ == '__main__':

    # (Train/Test) 데이터 세트 로드,
    (X_train, y_train), (X_test, y_test) = load_mnist(normalize=True,
                                                      flatten=True,
                                                      one_hot_label=False)
    print('X_test.shape =',X_test.shape,
          ', y_test.shape =', y_test.shape)
    # X_test.shape = (10000, 784) , y_test.shape = (10000,)
    # 784 = 28 * 28 = pixel numbers/ width and height of the image
    print(X_test[0])

    # 2) 신경망 생성 1
    with open('sample_weight.pkl', 'rb') as file:
        network = pickle.load(file)
    print('network:', network.keys())
    print('W1:',network['W1'].shape) # (784, 50)
    print('W2:',network['W2'].shape) #(50, 100)
    print('W3:',network['W3'].shape) #(100, 10)

    # mini-batch 함수
    batch_size = 100
    y_pred = mini_batch(network, X_test, batch_size)
    print('true[:10]', y_test[:10])
    print('pred[:10]', y_pred[:10])
    print('true[-10:]', y_test[-10:])
    print('pred[-10:]', y_pred[-10:])

    # 정확도 출력
    acc = accuracy(y_test, y_pred)
    print('정확도:', acc)
    # 달라지는건 없다