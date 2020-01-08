"""
파라미터 최적화 알고리즘 6) Nesterov's Accelerated Gradient
    모멘텀(momentum) 알고리즘을 좀 더 적극적(???)으로 활용한 알고리즘
    momentum: v = m * v - lr * gradient(dL/dW) (수학적으로: v <- m * v - lr * dL/dW)
            : m -> momentum v를 변경할 때 사용하는 하이퍼 파라미터
            : W = W + v = W - lr * dl/dW + m * v #v: 변경하기 전의 모멘텀이 들어온다 -> 이전 학습에서 학습된 파라미터
    적극적으로 활용 -> 갱신되기 전의 v를 갱신된 후의 v를 다시 한번 넣어주었다
                = W - lr * dl/dW + m * (m * v - lr * gradient(dL/dW))
                = W - m**2 * v - (1 + m)* lr * dL/dW
                # 떨어지는 아이를 더 빨리 가라고 밀어버리는 효과 *0*!!

 Momentum: v = m * v - lr * dL/dW
    -> W = W + v = W - lr * dL/dW + m * v                    --- (*)
                 = W - lr * dL/dW + m * (m * v - lr * dL/dW) --- (**)
                 = W + m**2 * v - (1 + m) * lr * dL/dW       --- (***)

original paper: http://proceedings.mlr.press/v28/sutskever13.pdf
"""

import numpy as np
from ch06.ex01 import fn, fn_derivative
import matplotlib.pyplot as plt

class Nesterov:
    def __init__(self, lr = 0.01, m = 0.9):
        self.lr = lr
        self.m = m
        self.v = dict()

    def update(self, params, gradients):
        if not self.v:
            for key in params:
                self.v[key] = np.zeros_like(params[key])
        # W - m**2 * v - (1 + m)* lr * dL/dW
        for key in params:
            self.v[key] = self.m * self.v[key] - self.lr * gradients[key]
            params[key] += self.m**2 * self.v[key] - (1 + self.m) * self.lr * gradients[key]

if __name__ == '__main__':
    # Nesterov 클래스의 인스턴스 생성
    nesterov = Nesterov(lr=0.1)

    params = {'x': -7., 'y': 2.}
    gradients = {'x': 0.0, 'y': 0.0}

    # 학습하면서 파라미터(x,y)들이 업데이트되는 내용을 저장하기 위한 리스트
    x_history = []
    y_history = []
    for i in range(30):
        x_history.append(params['x'])
        y_history.append(params['y'])
        gradients['x'], gradients['y'] = fn_derivative(params['x'], params['y']) # ax,ay 리턴 => gradients
        nesterov.update(params, gradients)
        print(f"({params['x'], params['y']})")

    x = np.linspace(-10, 10, 2000)
    y = np.linspace(-5, 5, 1000)
    X,Y = np.meshgrid(x,y)
    Z = fn(X, Y)

    mask = Z > 8
    Z[mask] = 0

    plt.contour(X,Y,Z, 10)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Nesterov')
    plt.axis('equal')
    plt.plot(x_history, y_history, 'o-', color = 'red')
    plt.show()






