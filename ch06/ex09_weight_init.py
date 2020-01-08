"""
6.2 가중치 최깃값: Y = X @ W + b
신경망의 가중치 행렬(W)를 처음에 어떻게 초기화를 하느냐에 따라서
신경망의 학습 성능이 달라질 수 있다.
weight의 초깃값을 모두 0으로 하면 (또는 모두 균일한 값으로 하면) 학습이 이루어지지 않음.
그래서 weight의 초깃값은 정규 분포를 따르는 난수를 랜덤하게 추출해서 만든다
그런데, 정규 분포의 표준 편차에 따라서 학습의 성능이 달라짐

Y = X @ W + b
f(Y) = f(X@W + b)
activation function like ReLU, Sigmoid
"""

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1/(1 + np.exp(-x))

def tanh(x): #hyperbolic tangent
    return np.tanh(x)

def relu(x):
    return np.maximum(x, 0)

if __name__ == '__main__':
    #은닉층(hidden layer)에서 자주 사용하는 3가지 활성화 함수 그래프
    x = np.linspace(-10, 10, 1000)
    y_sig = sigmoid(x)
    y_tanh = tanh(x)
    y_relu = relu(x)
    plt.plot(x, y_sig, label = 'Sigmoid')
    plt.plot(x, y_tanh, label = 'tanh')
    plt.plot(x, y_relu, label = 'relu')
    plt.title('Activation Functions')
    plt.legend()
    plt.ylim((-1.5, 1.5))
    plt.axvline()
    plt.axhline()
    plt.show()


    # std = 1 일 때, std = 0.01일 때 이 세가지 activation functions는 각각 어떻게 처리(?) 하느냐
    # 가상 신경망에서 사용할 테스트 데이터를 생성
    np.random.seed(108)
    x = np.random.randn(1000,100) # 미니배치라고 생각하면 된다 (1000 rows, 100 columns)
                    # 정규분포를 따르도록 만들었다.
                    # 어느 단위를 맞추기 위해서 정규화를 한다 (randn: 평균과 표준편차를 갖는 것으로 데스트 세트 자체가 정규화 되었다)
    # z-score, min-max normalization,,,
    node_num = 100 #은닉층의 노드(뉴런) 개수
    hidden_layer_size = 5 #은닉층의 갯수
    activations = dict() # 데이터가 각 은닉층을 지날 때 마다 출력되는 결과들 저장할 딕셔너리

    # 은닉층에서 사용하는 가중치 행렬 * 5 -> 은닉층의 갯수
    w = np.random.randn(node_num, node_num)
            # 행 갯수는 input data의 컬럼 데이터와 같아야하고 (= node_num), column의 갯수는 은닉층의 노드와 같아야한다
            # a = X dot W
    a = x.dot(w) # 두번째에서는 z가 x가 되어야한다
    z = sigmoid(a) #첫번째 은닉층을 지난 결과값
    activations[0] = z

    for i in range(hidden_layer_size):
        if i == 0:
            w = np.random.randn(node_num, node_num)
            a = x.dot(w)
            z = sigmoid(a)
        else:
            w = np.random.randn(node_num, node_num)
            a = z.dot(w)
            z = sigmoid(a)
        activations[i] = z
        # print('z =', z)

    # 교재 p.204의 히스토그램 그리기

    # x-axis = activations[i]
    # flatten z? because it's a multi-dimensional array?
    # how can I put the range huhuhuuhuhuh wtf

    for i, a in activations.items():
        plt.subplot(1, len(activations), i+1)
        plt.title(str(i+1) + "-layer")
        plt.hist(a.flatten(), 30, range = (0,1))
    plt.show()





