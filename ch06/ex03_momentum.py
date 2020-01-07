"""
파라미터 최적화 알고리즘
1) 경사하강법 (sgd)
2) Momentum 알고리즘 (P = m * v)
v: 속도 (velocity)
m: 모멘텀 상수 (momentum constant)
lr: learning rate (학습률)
W: 우리가 찾고자/갱신하고자하는 파라미터 (i.e. W, b)
v = m * v - lr * dl/dW
W = W + v = W + m * v - lr * dl/dW
    # sgd에서 m*v 만큼 더 추가
# 질량이 더 크면 힘이 더 쎄져서 더 빨리 내려간다
"""
import matplotlib.pyplot as plt
import numpy as np
from ch06.ex01 import fn, fn_derivative

class Momentum:
    def __init__(self, lr = 0.01, m =0.9):
        self.lr = lr #학습률
        self.m = m #모멘텀 상수(속도 v에 곱해줄 상수)
        self.v = dict() #속도 #진짜 물리적인 속도는 아니다 # velocity diverges to different directions
                # 각 파라미터 방향의 속도를 저장하기 위해서

    def update(self, params, gradients):
        # params: 키값에 맞춰서 vector v를 만든다
        if not self.v: #dictionary에 원소가 없으면 (if self.v = dictionary에 원소가 있으면)
            for key in params:
                # 파라미터(x, y 등)와 동일한 shape의 0으로 채워진 배열 생성
                # 배열의 모양이 같아야한다 -> 전부 0으로 채워진 같은 shape의 배열들을 생성
                self.v[key] = np.zeros_like(params[key])
                # 속도를 바꿔간다 W = W + v = W + m * v - lr * dl/dW
                # 최초 v는 위에서 0으로 채워져있다
        # 속도 v, 파라미터 params 를 갱신 (update)하는 기능을 만들기
        for key in params:
            # v = m * v - lr * dl / dW
            # self.v[key] = self.m * self.v[key] - self.lr * gradients[key]
            # other way of writing:
            self.v[key] *= self.m
            self.v[key] -= self.lr * gradients[key]

            # W = W + v
            params[key] += self.v[key]


if __name__ == '__main__':
    # Momentum 클래스의 인스턴스를 생성
    momentum = Momentum(lr = 0.1, m = 0.05)
    # update 메소드 테스트
    # momentum.update(params, gradients)
    # 값을 어떻게 주지,,
    params = {'x':-7., 'y':2.} #파라미터 초기값
    gradients = {'x':0., 'y':0.} #gradient 초기값
    x_history = [] #param['x']가 갱신되는 과정을 저장할 리스트
    y_history = [] #param['y']가 갱신되는 과정을 저장할 리스트
    for i in range(30):
        x_history.append(params['x'])
        y_history.append(params['y'])
        gradients['x'], gradients['y'] = fn_derivative(params['x'], params['y'])
        momentum.update(params, gradients)

    # x,y의 갱신되는 값들을 출력
    for x, y in zip(x_history, y_history):
        print(f'({x},{y})')
        # y 값이 내려갔다가 살짝 올라가는 중

    # contour 그래프에 파라미터의 갱신 값 그래프를 추가
    x = np.linspace(-10,10,2000)
    y = np.linspace(-5,5,1000)
    X,Y = np.meshgrid(x,y)
    Z = fn(X,Y)

    mask = Z > 7
    Z[mask] = 0

    plt.contour(X,Y,Z, 10)
    plt.title('Momentum')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')

    plt.plot(x_history, y_history, 'o-', color = 'red')
    plt.show()
    # lr = 0.01에서 y의 절대값이 점점 줄고있다
    # lr = 0.1 점이 꼬불꼬불하게 나온다 # 0넘어가버림
    # sgd에서는 0.95까지 늘려야 0에 가까이 갔던 반면, momentum모양에서는 0.1 에서도 0이 넘도록 가버림
    # mask 생긴 후: 공이 swing하면서 내려가는 모양: 점이 띄엄띄엄있으면 훅 내려갔다가, 점이 촘촘하면 좀 더 느리게 간다 (롤러코스터 처럼)
    # Momentum(lr = 0.1, m = 0.05) 에서, 0을 지나가지 않았다
    # Momentum이라는 개념이 속도의 개념을 도입을 해서, 곡면에서 공이 내려가는 듯한 모습으로 파라미터들이 변화한다

    # 발산하지 않으면서 빨리 찾아가는 것이 가장 좋은 것(모델에 따라 다르다)
    
















