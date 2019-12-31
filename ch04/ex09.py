import pickle

import numpy as np
from dataset.mnist import load_mnist


class TwoLayerNetwork:
    def __init__(self, input_size, hidden_size, output_size,
                 weight_init_std = 0.01):
        """ 입력: input_size d EX) 784(28x28)개 이고, (784, 32)
            첫번째 층 (layer)에는 neuron(뉴런): hidden_size EX) 32개 (32,10)
             출력층 (layer)의 뉴런 개수: output_size EX)10개
            2개의 가중치가 필요하다: W1, W2
             weight_init_std: 표준편차 1을 0.01로 바꾼다
             - 모양은 바뀌지 않지만 숫자는 바뀜"""
        np.random.seed(1231)
        self.params = dict() #weight/bias 행렬들을 저장하는 딕셔너리
        # weight 행렬(W1, W2), bias 행렬(b1, b2)을 난수로 생성
        # x(1,784) @ W2(784, 32) + b2(1,32)
        self.params['W1'] =  weight_init_std * np.random.randn(input_size,hidden_size)
        self.params['b1'] = np.zeros(hidden_size) # we do not know yet what will be in here so we fill the df with 0
        # z1(1,32) @ W2(32,10) + b2(1,10)
        self.params['W2'] =  weight_init_std * np.random.randn(hidden_size,output_size)
        self.params['b2'] = np.zeros(output_size)
        # weight_init_std: the ranges becomes smaller since the bell-like graph is narrower
        # uniform/ 비슷비슷한 results are returned (10개가 균일한 분포/distribution)을 갖게 된다

    def predict(self, x):
        """ data -> hidden layer을 거쳐서 -> output layer을 거쳐서 예측값이 나온다
            In hidden layer = data@W1 + b1 => activation function (i.e. sigmoid, reLu) -> output layer(in classification problem, softmax)
                                            sigmoid(hidden_layer) = z ->                softmax(z@W2+b2) = y """
        a1 = x.dot(self.params['W1']) + self.params['b1']
        z = self.sigmoid(a1)
        a2 = z.dot(self.params['W2']) + self.params['b2']
                    #a2가 배열
        y = self.softmax(a2)
        return y

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
        # 내부적으로는 전부 행렬의 연산이 일어난다

    def softmax(self, x):
        """ softmax = exp(x_k)/ sum i to n [exp(x_i)] """
        dimension = x.ndim
        if dimension == 1:
            max_x = np.max(x)
            x -= max_x #overflow 방지!
            result = np.exp(x) / np.sum(np.exp(x))
        elif dimension == 2:
            xt = x.T # 배열 x의 전치 행렬를 먼저 작성
            max_x = np.max(xt, axis = 0)
            xt -= max_x
            result = np.exp(xt)/np.sum(np.exp(xt), axis = 0)
            result = result.T
        return result
    # 내부적으로 전부 행렬의 연산이 일어난다
    # softmax는 모든 값을 더했을 때의 값은 1이다
    # each row pertains to one image
    # row / axis = 0, column/ axis = 1 -> so we add by column-oriented i

    def accuracy(self, x, y_true):
        """
        x의 예측값과
        :param x: 예측값을 구하고 싶은 데이터; x is a two-dimensional array
        :param y_true: 실제 레이블
        :return: 정확도
        """
        y_pred = self.predict(x)
        predictions = np.argmax(y_pred, axis = 1)
        # argmax => 한 배열에서 최댓값을 갖는 값의 인덱스를 리턴
        true_values = np.argmax(y_true, axis =1)
        print('predictions =', predictions)
        print('true_values =', true_values)
        acc = np.mean(predictions == true_values)
        return acc

    def loss(self, x, y_true):
        y_pred = self.predict(x)
        entropy = self.cross_entropy(y_true, y_pred)
        return entropy

    def cross_entropy(self, y_true, y_pred):
        if y_pred.ndim == 1:
            # 1차원 배열인 경우, 행의 개수가 1인 2차원 배열로 변환
            y_pred = y_pred.reshape((1, y_pred.size)) # 2차원 행렬로 만든다
            y_true = y_true.reshape((1, y_true.size))
        # y_true 값이 one-hot-encoding임을 가정했다
        # y_true 에서 1이 있는 컬럼 위치 (인덱스)를 찾음
        true_values = np.argmax(y_true, axis = 1) # 1차원에서는 axis = 1같은게 없다
                                                  # 2차원 배열로 변환했기 때문에 가능한 것
        n = y_pred.shape[0] #2차원 배열의 shape: (row, column) => row의 갯수
        rows = np.arange(n) #1차원 배열 [0, 1, 2, 3...] # row의 인덱스가 될 값들
        # y_pred의 y_pred[[0, 1, 2], [3, 3, 9]]  => [y_pred[0,3], y_pred[1,3], y_pred[2,3]]
        log_p = np.log(y_pred[rows, true_values])
        # 이 로그값들의 배열을 sum 해준다
        entropy = -np.sum(log_p)/n
        return entropy

    # my solution
        # delta = 1e-7
        # y_pred = self.predict(x)
        # if y_pred.ndim == 1:
        #     ce = -np.sum(y_true * np.log(y_pred + delta))
        # elif y_pred.ndim == 2:
        #     ce = -np.sum(y_true * np.log(y_pred + delta))/len(y_pred)
        # return ce

# entropy 가 클수록 불확실성이 크고, 맞는게 많이 없고, vice-versa

    def gradient(self, x, y_true):
    # 변화율을 구하는 값이 손실함수고, 변화율을 구하고 싶으면 데이터와 실제값이 있어야한다
    # 변화율을 찾는 목적은 => W값의 변환
    # entropy를 최소화 시켜나가는 것이 우리가 할 일
        loss_fn = lambda w: self.loss(x, y_true) # 이 함수를 최소화 시켜나가는 것
        gradients = dict() #W1, b1, W2, b2의 gradient를 저장할 딕셔너리
        gradients['W1'] = self.numerical_gradient(loss_fn, self.params['W1'])
        gradients['b1'] = self.numerical_gradient(loss_fn, self.params['b1'])
        gradients['W2'] = self.numerical_gradient(loss_fn, self.params['W2'])
        gradients['b2'] = self.numerical_gradient(loss_fn, self.params['b2'])
        return gradients

    # other method for def gradients()
    # def gradients(self, x, y_true):
    #     loss_fn = lambda w: self.loss(x, y_true)
    #     gradient = dict()
    #     for key in self.params:
    #         gradient[key] = self.numerical_gradient(loss_fn, self.params[key])

    def numerical_gradient(self, fn, x):
       # ch04 05
       # 앞에서 processing 된 행렬들이 들어오는데, W는 2차원, b는 1차원이므로, 두개 다 가능한 함수여야한다
        h = 1e-4
        gradient = np.zeros_like(x) #와 같은 모양으로 0의 배열을 만든다
        with np.nditer(x, flags = ['c_index', 'multi_index'], op_flags = ['readwrite']) as it:
            while not it.finished:
                i = it.multi_index
                ith_value = it[0] #원본 데이터를 임시 변수에 저장
                it[0] = ith_value + h # 원본 값을 h만큼 중가
                fh1 = fn(x) # f(x) + h
                it[0] = ith_value - h
                fh2 = fn(x) # f(x) - h
                # gradient[i] = (fh1 - fh2)/(2*h) # gradient에 넣어야한다 not in it (not applicable for 2-dimensional index) -> so we added multi_index in the parameter as well as changed i to multi_index
                gradient[i] = (fh1 - fh2) / (2 * h)
                it[0] = ith_value #원본값으로 되돌려준다 / 가중치 행렬의 원소를 원본값으로 복원
                it.iternext()
        return gradient

if __name__ == '__main__':
    # 신경망 생성
    # W1, W2, b1, b2의 shape을 확인

    # 신경망 가중치(와 편향, bias) 행렬들을 생성
    neural_net = TwoLayerNetwork(input_size = 784,
                                 hidden_size = 32,
                                 output_size = 10)
    print(f'W1: {neural_net.params["W1"].shape}, W2: {neural_net.params["W2"].shape}')
    print(f'b1: {neural_net.params["b1"].shape}, b2: {neural_net.params["b2"].shape}')

    # 신경망 클래스의 predict() 메소드 테스트
    #mnist 데이터 세트를 로드
    (X_train, y_train), (X_test, y_test) = load_mnist(one_hot_label=True)

    with open('../ch03/sample_weight.pkl', 'rb') as file:
        network = pickle.load(file)

    #X_train[0]을 신경망에 전파(propagate)를 시켜서 예측값 확인
    y_pred0 = neural_net.predict(X_train[0])
    # 무슨 값을 넘겨야 하는지 잘 모르겠다
    print('y_pred0',y_pred0)
    print('y_true0', y_train[0])

    # X_train[:5]를 신경망에 전파시켜서 예측값 확인
    y_pred1 = neural_net.predict(X_train[:5])
    print('y_pred1', y_pred1)
    print('y_true1', y_train[:5])
    print('cross entropy =', neural_net.loss(X_train[:5], y_train[:5]))

    # accuracy
    acc = neural_net.accuracy(X_train[:100], y_train[:100])
    print('accuracy =', acc)
    print('cross entropy =', neural_net.loss(X_train[:100], y_train[:100]))

    # W는 어떻게 조절할 것인가?
    # W의 변화율대로 변화할 수 있도록 고려,,, ^^ -> gradient descent -> 무엇의? gradient의!
    # bias도 take into consideration
    # 손실의 변화를 찾아서 -> 손실을 줄여줄 수 있는 함수 => entropy 함수 만들기


    # gradients 메소드 테스트
    gradients = neural_net.gradient(X_train[:100], y_train[:100])
    for key in gradients:
        print(key,  np.sum(gradients[key]))

    # gradient: 가중치W를 바꿀 때 사용하는 값
    # 찾은 gradient를 이용해서 weight/bias 행렬들을 업데이트
    lr = 0.1 #학습률(learning rate)
    for key in gradients:
        neural_net.params[key] -= lr * gradients[key] # 한번 바뀐 가중치 행렬

    # 단점: consumes a lot of time (each round takes a minute, but we have to run 600 of them for 100 times duh)

    # How you do the mini-batch
    # epoch = 100
    # for i in range(epoch):
    #     for i in range(10):
    #        gradients = neural_net.gradient(X_train[i * 100:(i+1) *100],
    #                                     y_train[i * 100:(i+1) *100])
    #         for key in gradients:
    #               neural_net.params[key] -= lr * gradients[key]
    # 순서 유지된다 -> epoch가 들어올 때 마다 index를 셔플된다



