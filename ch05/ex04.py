"""
숙제: sigmoid
# 활성화 함수 중 많이 사용되는 시그모이드 함수
# 시그모이드 함수의 정의: y = 1/ (1 + exp(-x))
    1) dy/dx = y(1-y) 를 증명
    2) 이걸을 이용해서 sigmoid 라는 클래스를 생성
        -> 시그모이드 뉴런작성 with forward & backward 메소드
    # 시그모이드의 입력값은 1개
"""
import numpy as np
from ch03.ex01 import sigmoid


class Sigmoid:
    def __init__(self):
        # x.self = None
        self.y = None # forward 메소드의 리턴값 y를 저장하기 위한 필드

    # def f_sigmoid(self,x):
    #     x.self = x
    #     return 1/ (1+ math.exp(-x))

    # def back_sigmoid(self, x):
    #     x.self = x
    #     y = 1/ (1+ math.exp(-x))
    #     return self.y(1- self.y)

    def forward(self, x):
        y = 1/ (1+ np.exp(-x)) # 행렬이 들어와도 계산이 가능하다
        self.y = y
        return y

    def backward(self, dout):
        # back propagation: 미분값에 곱해줘야한다
        # 정의: y(1 - y)
        return dout * self.y * (1 - self.y) # self.. 뭐야 순서에 왤케 민감ㅎ ㅠㅜ
                    # dy/dx

    # above, we know the rules and derivatives
    # But if not,
    # 아주작은 h에 대해서 [(f+h) - (f-h)]/h 계산
    h = 1e-7
    dx2 = (sigmoid(0. + h) - sigmoid(0. - h)) / h
    print('dx2 = ', dx2)


if __name__ == '__main__':
    # 시그모이드 생성
    sigmoid_gate = Sigmoid()
    # x =1 이 일 때, 함수의 값 리턴 (forwawrd)
    y = sigmoid_gate.forward(x = 0.)
    print(y) #x가 0일 때 sigmmoid((0)) = 0.5

    # backward x = 0 에서 sigmoid의 gradient (접선의 기울기)
    dx = sigmoid_gate.backward(dout = 1.)
    print('dx =', dx )

    #  시그모이드 뉴런에 x값을 주면, y 를 리턴해준다
    # 신경망층에 1개의 값이 아니라 배열을 줘도 계산할 수 있다








