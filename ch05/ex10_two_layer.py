"""
오차 역전파를 사용한 2층 신경망 (1 은닉층 + 1 출력층)
"""
from collections import OrderedDict
import numpy as np

from ch04.ex03 import cross_entropy
from ch05.ex05_ReLU import Relu
from ch05.ex07 import Affine
from ch05.ex08_softmax_loss import SoftmaxWithLoss
from dataset.mnist import load_mnist


class TwolayerNetwork:
    def __init__(self, input_size, hidden_size, output_size,
                 weight_init_std = 0.01): # 난수들을 선택할 때 0에 가까운 수들을 더 선택할 것이냐
        """ 신경망의 구조(모양) 결정
            input size: number of variables
            output size: i.e. 숫자 이미지 -> 10개, iris -> 3개
        """
        np.random.seed(106)

        # 가중치/편향 행렬들을 초기화
        # W1, b1, W2, b2 와 같은 값들을 리스트로 만들어두면 나중에 iteration하기 쉽다
        self.params = dict()
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size) # 편향은 모두 0으로 준다
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        # 위의 방법대로라면 저장한 순서대로 반복문에서 출력되는 것이 보장되지 않는다
        # layer 생성/초기화
        self.layers = OrderedDict()
        # 딕셔너리에 데이터가 추가된 순서가 유지되는 딕셔너리
        # 순서가 중요하기 때문에 (Affine - ReLU - Affine - SoftmaxWithLoss => forward그리고 backward는 이 반대 순서대로 진행되어야하기 때문)
        self.layers['affine1']  = Affine(self.params['W1'],
                                         self.params['b1'])
        self.layers['relu'] = Relu()
        self.layers['affine2'] = Affine(self.params['W2'],
                                        self.params['b2'])
        self.last_layer = SoftmaxWithLoss()

        # backward propagation: since it is a propagation with order, we can just reverse the process above

    def predict(self, X):
        """ 입력데이터를 전달을 받아서, 입력받은 데이터가 무엇이 되느냐를 예측하는 함수
            입력값을 입력 받아서, softmaxwithloss가기 직전에 모든 값들의 확률?을 리턴한 값을 사용 (왜냐하면 swl는 CE값 1개만 리턴해준다) (affine2 까지 진행된 값)
            확률로 표현되어 있어도 되고, 아니여도 상관없다 (output 중에서 가장 큰 값이 무엇이냐가 제일 중요; 대소비교)  """

        # How predict works
        # Y1= self.layers['affine1'].forward(X) #첫번째 출력값
        # Y2 = self.layers['relu'].forward(Y1)
        # Y3 = self.layers['affine2'].forward(Y2) # this is the data that we want to process

        # could shorten the code using for-loop
        for layer in self.layers.values():
            X = layer.forward(X)
        return X
        # we can (creer/confiar en) the for-loop because we used ordered dict
        # We can use the code regardless the number of layers

        # how I thought
        # X 가 무슨 값을 돌려줄 것인지 predict
        # X = self.layers['affine2']
        # # affine2의 output column을 빼서 one_hot_label의 자리 비교
        # print('X =', X)

    def loss(self, X, Y_true):
        """ 손실 함수, 끝까지 가기 위한 함수 -> 문제를 끝까지 넘겨서 예측값을 줘야하고 (forward) -> 실제값(정답지)를 가지고 있어서 비교할 수 있게 해주어야한다
            최종 리턴: cross_entropy """

        #SoftmaxWithLoss, 가장 마지막 출력층 전까지의 forward를 먼저 계산
        Y_pred = self.predict(X) # predict: 입력값을 주면 예측값을 주는 함수 -> softmax전까지의 값을 계산
        loss = self.last_layer.forward(Y_pred, Y_true)
        return loss # 만들어진 ce를 리턴


    def accuracy(self, X, Y_true):
        """ 손실의 반대 개념 이라고 생각해도 된다.
            입력 데이터 X와 실제 값(레이블) Y_true가 주어졌을 때,
            예측 값들의 정확도를 계산해서 리턴.
            accuracy = 예측이 실제값과 일차하는 개수/ 전체 입력 데이터 개수
             # X에서 적어도 2개 이상의 값을 넘겼을 때, n개에 대한 예측값
            X, Y_true는 모두 2차원 배열(행렬)라고 가정.
             """
        Y_pred = self.predict(X)
        predictions = np.argmax(Y_pred, axis =1)
        trues = np.argmax(Y_true, axis = 1) #one_hot_encoding에서 1이 어디에 들어가 있느냐를 찾는 것
                                            # (어짜피 0,1밖에 없으니까, 1이 제일 큰 값. 행별로 비교하면 각 행에서 가장 큰 값이 1)
        # predictions = trues
        acc = np.mean(predictions == trues) # true와 false로 이루어진 리스트 => 0,1
        # 1들만 더해짐 (true values) -> 일치하는 갯수/ 전체 갯수
        return acc

    def gradient(self, X, Y_true):
        """ X: 입력 데이터와 Y_true: 실제 데이터가 주어졌을 때,
            모든 레이어에 대해서 forward propagation을 수행한 후,
            오차 역전파 방법을 이용해서 dW1, db1, dW2, db2를 계산하고 리턴
            -> W와 b의 값 수정을 위해서 gradient를 찾는다 """
        gradients = dict()

        # loss 1개만 호출하면 모든 레이어에 대해 forward가 호출이 된다( 마지막 레이어 포함)
        self.loss(X, Y_true) #forward propagation

        # back propagation
        dout = 1
        dout = self.last_layer.backward(dout)
        # 반복문을 사용한다
        layers = list(self.layers.values()) #[Affine1, ReLU, Affine2] # 순서가 있는 딕셔너리 -> 순서대로 값들이 꺼내진다
                                            # 리스트로 만드는 이유: 우리가 원하는 것 -> 반대로 하기 위해서 -> We cannot change the order from a dictionary, but we can from a list
        layers.reverse() # 리스트를 역순으로 바꿈 # ordered dict에는 reverse함수가 따로 없어서 value 꺼내기 -> 순서 바꾸기 해줌
                        # layer에서 for-loop을 사용한다
        for layer in layers:
            dout = layer.backward(dout)
                    # 입력해준 dout으로 backward시키고, 그걸 가지고 또 backward,,,

            # 모든 레이어에 대해서 역전파가 끝나면, 가중치/편향들의 gradient를 찾을 수 있다
            gradients['W1'] = self.layers['affine1'].dW
            gradients['b1'] = self.layers['affine1'].db
            gradients['W2'] = self.layers['affine2'].dW
            gradients['b2'] = self.layers['affine2'].db
            # __init__ 에서 저장 가능하게 만들었기 때문에, 이런 코드들이 가능하다

        return gradients





if __name__ == '__main__':
    #MNIST 데이터 로드
    (X_train, Y_train), (X_test, Y_test) = load_mnist(one_hot_label= True)
    # 데이터 shape 확인
    print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
    # (60000, 784 -> input size) (60000, 10 -> output size) (10000, 784) (10000, 10)
    # train data -> 6만개, test data -> 1만개


    # 신경망 객체 생성
    neural_net = TwolayerNetwork(input_size= 784,
                                 hidden_size= 32, #성능에 따라 바꿔준다/ 또 하나의 파라미터가 되는 셈
                                 output_size= 10)

    # 파라미터 확인
    for key in neural_net.params:
        # 2층 신경망은 가중치와 편향을 저장하고 있는 변수가 있는데, 그 것의 이름이 params 이다
        print(key, ':', neural_net.params[key].shape)
    for key in neural_net.layers:
        print(key, ':', neural_net.layers[key])
    print(neural_net.last_layer) # <ch05.ex08_softmax_loss.SoftmaxWithLoss object at 0x000001E03CD6DC88> #SoftmaxWithLoss

    # softmax -> affine2 -> relu -> affine

    # predict(), loss() methods Test
    Y_pred = neural_net.predict(X_train[0])
    print('Y_pred1', Y_pred)
    print(np.argmax(Y_pred))
    # argmax: what is the biggest value among Y_pred1? 3 -> the one in index[3]
    loss1 = neural_net.loss(X_train[0], Y_train[0])
    print('loss1 =', loss1)

    Y_pred = neural_net.predict(X_train[:3])
    print('Y_pred2',Y_pred)
    print(np.argmax(Y_pred, axis = 1))
    # 여러개를 한번에 넘긴 것 (mini-batch와 같음)
    # axis = 1, df is returned in 2-dimensional df. We have to compare column-wise
    # [3 3 3] = 모든 값을 3번으로 찍고 있는 것
    loss2 = neural_net.loss(X_train[:3], Y_train[:3])
    print('loss2 =', loss2)

    # loss1과 2가 비슷하다, 왜냐하면 1개의 숫자로 찍었기 떄문에...
    # 평균이기 때문에 비슷한 값이 나온다

    # accuracy() method test
    print('Y_train',Y_train[:3])
    # Current model guesses all the values to be 3.
    # In print('Y_train',Y_train[:3]), the true values are: 5, 0, thus, the accuracy will be 0
    # since all the predicted values are three which is non-existent in the list of true values
    print('accuracy =', neural_net.accuracy(X_train[:3], Y_train[:3]))

    print('Y_train[:10]', Y_train[:10])
    print('accuracy[:10]', neural_net.accuracy(X_train[:10], Y_train[:10]))


    # gradient() method test
    gradients = neural_net.gradient(X_train[:3], Y_train[:3])
    print('gradients =', gradients)
    # 데이터의 모양은 print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape) 와 같이 설정되어 있다
    # 위의 값을 for-loop에 넣어서 보기:
    # key와 value를 동시에 꺼내기: items()
    for key in gradients:
        print(gradients[key].shape, end = ' ')
    print()









    # Question from 경진쓰
    # 반복하기
    # t = 1
    # def inner_fn():
    #     print(t)
    # t = 2 # 반복문 안에 t값이 없으면 바깥쪽의 t (1) 을 가져와서 사용하지만, 안쪽에 t가 있으면 이미 바뀌어 버린 다음이라 t =2라고 출력하다
    # inner_fn()

    # 파이썬이라 가능한 코딩
    # t = 1
    # def f():
    #     t2 = 2
    #     def g():
    #         print(t2) #t2를 호출
    #     g()
    # t2 = 11 # 얜 바뀌지 않고 출력 # t2가 안에도 있고 (=2), 바깥 쪽에도 있다 (=11); 하지만 Python은 가까운 쪽의 변수를 사용하므로, t2 =11가 나중에 나왔지만, t2 = 2에 가려져서 t2 = 2가 출력된다
    # f()

    # lambda를 사용할 때에 문제
    # f에는 함수 4개가 들어간다 -> 첫번째함수는 0을 출력, 두번쨰 함수는 1,... 하는 함수를 만들고 싶었는데,
    # 인덱스 n에 뭐를 줘도 같은 값을 준다 -> 왜?
    # f = []
    # for i in range(5):
    #     f.append(lambda : print(i))
    # print(i) # 항상 i의 값은 4라서
    # f[0]() # f[n]에 무슨 인덱스 값을 줘도 4를 리턴한다
    # closer이라고 부른다
    # 실행되는 시점의 i
    # 0,1,2,3,4 가 출력이 되는 것이 아니고,,,





