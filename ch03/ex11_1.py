"""
HW:
# Load the train & test data set
# 신경망 생성 -> network
# batch_size = 100
# y = pred = mini_batch(network, X_test, batch_size)
# 신경망에 100개씩 100번 테스트하겠다 (y에는 10,000개의 예측값이 있어야한다)
# 데이터는 n번 돌리느냐는 계산 비용의 차이를 많이 돌리지만, 데이터 자체의 dimension이 커지는 것은 계산 비용의 차이가 크지 않다
# 정확도(accuracy) 출력
"""
# 신경망생성 (W1, b1 , ...) -> network
import pickle
import numpy as np

from ch03.ex01 import sigmoid
from ch03.ex05 import softmax
from dataset.mnist import load_mnist


def init_network():
    with open('sample_weight.pkl', mode='rb') as file:
        network = pickle.load(file)
    print(network.keys())
    return network


def forward(network, x):
    """  ex08의 forward에서 마지막에 mini_batch 활성 함수에 넣어줌  """
    # 가중치 행렬 (weight matrices)
    W1, W2, W3 = network['W1'], network['W2'],network['W3']
    b1, b2, b3 = network['b1'], network['b2'],network['b3']

    # 첫번째 은닉층
    a1 = x.dot(W1) + b1
    z1 = sigmoid(a1)

    # 두번째 은닉층
    a2 = z1.dot(W2) + b2
    z2 = sigmoid(a2)

    #output layer
    a3 = z2.dot(W3) + b3
    y = softmax(a3)

    return y

# def minibatches(X, batch_size):
#     batch_starts = [s for s in range(0, len(X), batch_size)]
#     mini = [X[s: s+batch_size] for s in batch_starts]
#     return mini

def predict(network, X_test, batch_size):

    # my solution 1
    # y_pred = []
    # for sample in X_test:
    #     sample_hat = forward(network, sample)
    #     sample_pred = [sample for sample_hat in range(0, len(X_test), batch_size)]
    #     mini = [X_test[s: s + batch_size] for s in sample_pred]
    #     y_pred.append(mini)
    # return y_pred

    # my solution2
    # y_pred = []
    # for i in X_test:  # for i in range(0, len(X_test), batch_size):
    #     X_batch = [X_test[i:i+batch_size]]
    #     sample_hat = forward(network, X_batch) #X_batch is not a scalar index > but I wanted to give an array instead
    #     y_batch = predict(network,X_test)
    #     # sample_pred = np.argmax(sample_hat)
    #     # y_pred.append(sample_pred)
    # return y_pred

    # my solution 3
    # for i in range(0, len(X_test), batch_size):
    #     x_batch = X_test[i:i+batch_size]
    #     y_batch = predict(network, x_batch)
    #     p = np.argmax(y_batch, axis = 1)
    #     accuracy_cnt += np.sum(p == X_test[i:i+batch_size])



if __name__ == '__main__':

    # (Train/Test) 데이터 세트 로드,
    (X_train, y_train), (X_test, y_test) = load_mnist(normalize=True,
                                                  flatten=True,
                                                  one_hot_label=False)

    # 신경망생성
    network = init_network()
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b1'], network['b1']

    # batch_size = 100
# y = pred = mini_batch(network, X_test, batch_size)
# 신경망에 100개씩 100번 테스트하겠다 (y에는 10,000개의 예측값이 있어야한다)
    y_pred = predict(network, X_test, 100)
    print(y_pred[:10])
# 정확도(accuracy) 출력
