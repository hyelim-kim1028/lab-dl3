"""
Simple Convolutional Neural Network (CNN)
"""


# 228 페이지 그림 7-2
from collections import OrderedDict
from common.layers import Convolution, Relu, Pooling, Affine, SoftmaxWithLoss
import numpy as np

class SimpleConvNet:
    """
    X > Convolution > activation function > pooling
    : Convolution > activation function > pooling  = 1개의 활성곱 레이어/ 1 hidden layer
    : there can be multiples of layers

    1st hidden layer: Convolution (W,b) -> ReLU -> Pooling
    2nd hidden layer: Affine (W,b) -> ReLU (fully-connected network, 완전 연결층)
    출력층: Affine (W,b) -> SoftmaxWithLoss

    # batch_normalization을 넣는다고 하면 또 다른 파라미터 (gamma, beta)가 있다
    # 파라미터가 많아지면 gradient를 계산하는 시간이 길어진다
    """

    def __init__(self, input_dim = (1, 28, 28),
                 conv_params = {'filter_num':30,'filter_size': 5, 'pad': 0, 'stride':1},
                 hidden_size = 100, output_size = 10, weight_init_std = 0.01):
       """ 인스턴스 초기화 (변수들의 초기값을 줌) - CNN 구성, 변수들 초기화
        input_dim: 입력 데이터 차원, MINIST인 경우(1, 28, 28)
        conv_param: Convolution 레이어의 파라미터(filter, bias)를 생성하기 위해 필요한 값들
            필터 개수 (filter_num),
            필터 크기(filter_size = filter_height = filter_width),
            패딩 개수(pad),
            보폭(stride)
        hidden_size: Affine 계층에서 사용할 뉴런의 개수 -> W 행렬의 크기
        output_size: 출력값의 원소의 개수. MNIST인 경우 10
        weight_init_std: 가중치(weight) 행렬을 난수로 초기화 할 때 사용할 표준편차 
        """
       filter_num = conv_params['filter_num']
       filter_size = conv_params['filter_size']
       filter_pad = conv_params['pad']
       filter_stride = conv_params['stride']
       input_size = input_dim[1]
       conv_output_size = (input_size - filter_size + 2 * filter_pad) / \
                          filter_stride + 1
       pool_output_size = int(filter_num * (conv_output_size / 2) * (conv_output_size / 2))


       # CNN Layer에서 필요한 파라미터들
       self.params = dict()
       self.params['W1'] = weight_init_std * np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
       self.params['b1'] = np.zeros(filter_num)
       self.params['W2'] = weight_init_std * np.random.randn(pool_output_size, hidden_size)
       self.params['b2'] = np.zeros(hidden_size)
       self.params['W'] = weight_init_std * np.random.randn(hidden_size, output_size)
       self.params['b3'] = np.zeros(output_size)


       # CNN Layer(계층) 생성, 연결
       self.layers = OrderedDict()

        # 방법 1 __init__(self,W,b) 라고 주고,  self.W = W, self.b = b 를 선언
        # self.W = W # 난수로 생성하려고 해도 데이터의 크기(size)를 알아야 필터를 생성할 수 있다
        # self.b = b # bias의 크기는 필터의 크기와 같다. 마찬가지로 난수로 생성해도 크기를 알아야한다 => dimension 결정

        # 방법 2
        # input_dim = (1, 28, 28) = MNIST를 위한 클래스
        # dimension을 주도록 설정 + 필터갯수가 있도록 설정해줘야한다
        # convolution 할 때 필터를 몇번 만들 것인가 -> 난수로 만들어서 넣어줄 수 있다

                    # key값
       self.layers['Conv1'] = Convolution(self.params['W1'],
                                           self.params['b1'],
                                           conv_params['stride'],
                                           conv_params['pad'])  # W와 b를 선언
       self.layers['ReLu1'] = Relu() # x -> Convolution에서 전해주는 값
       self.layers['Pool1'] = Pooling(pool_h = 2, pool_w =2, stride =2)
       self.layers['Affine1'] = Affine(self.params['W2'],
                                        self.params['b2'])
       self.layers['Relu2'] = Relu()
       self.layers['Affine2'] = Affine(self.params['W3'],
                                        self.params['b3'])
       self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        """ network의 목적: 예측하는 것  """
        for layer in self.layers.vlaues():
            x = layer.forward(x)
            return x

    def loss(self, x, t):
        """ 순반향 전파가 모두 끝나고 손실 계산
        -> 이 손실을 꺼꾸로 보내면서 gradient를 계산
        """
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def accuracy(self):
        pass

    def gradient(self, x, t):
        # 순전파
        self.loss(x,t)

        # 역전파
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.vlaues())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        #결과저장
        grads = {}
        grads['W1'] = self.layers['Conv1'].dW
        grads['b1'] = self.layers['Conv1'].db
        grads['W2'] = self.layers['Affine1'].dW
        grads['b2'] = self.layers['Affine1'].db
        grads['W3'] = self.layers['Affine2'].dW
        grads['b3'] = self.layers['Affine2'].db



if __name__ == '__main__':
    # MNIST 데이터 로드
    # SimpleConvNet 생성
    # 학습 -> 테스트 > 미니배치로 보내고, 에포크 10~20 반복하는 학습 -> 테스트 세트에서 accuracy 확인
    pass

