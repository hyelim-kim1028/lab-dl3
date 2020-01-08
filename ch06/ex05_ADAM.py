"""
파라미터 최적화 알고리즘
4) ADAM
- Adapative Moment Estimate/ - 학습률 변화 + 속도(모멘텀) 개념 도입
- AdaGrad + Momentum

W: 파라미터
lr: 학습률 (learning rate)
t: timestamp (반복할 때마다 증가하는 숫자. update 메소드가 호출될 때마다 +1)
beta1, beta2: 모멘텀을 변화시킬 때 사용하는 상수들. 0 <= beta1,2 < 1
m: 1st momentum => 1st momentum ~ gradient(dL/dW) -> SGD의 gradient를 수정한다
v: 2nd momentum => 2nd momentum ~ gradient ** 2 ((dL/dW) ** 2) -> SGD의 학습률을 수정한다
m = beta1 * m + (1 - beta1) * grad
v = beta2 * v + (1 - beta2) * grad * grad
m_hat = m/(1-beta1 ** t)
v_hat = v/(1-beta2 ** t)
W = W - lr * m/sqrt(v)
    # m을 grad라고 생각하면, ada grad 와 같은 모습을 모인다
    # sqrt(v) -> grad ^ 2 이기 때문에, v에 루트를 씌움
"""

import numpy as np
from ch06.ex01 import fn, fn_derivative
import matplotlib.pyplot as plt

class Adam:
    def __init__(self, lr = 0.01, beta1 = 0.9, beta2 = 0.99):
        self.lr = lr # learning rate (학습률)
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = dict()
        self.v = dict()
        # time stamp 는 for loop 으로 줘야할까?
        self.t = 0


    def update(self, params, gradients):
        """
        m,v, t += 1
        """
        self.t += 1 # 업데이트가 호출 될 때마다 timestamp 1씩 증가
        if not self.m: # m 이 비어있는 dictionary 일 때/ 원소가 없을 때
            for key in params:
                # 1st, 2nd 모멘텀을 파라미터의 shape과 동일하게 생성
                self.m[key] = np.zeros_like(params[key])
                self.v[key] = np.zeros_like(params[key])
                # 나는 처음에 if not self.m and self.v 라고 했는데 왜 이렇게 안했을까?

        # epsilon: 0으로 나누는 경우를 방지하기 위해서 사용할 상수
        epsilon = 1e-8

        for key in params:
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * gradients[key]  # beta1, beta2는 0~1 사이에 있는 숫자라고 가정
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * gradients[key] **2 # numpy calculates gradient[key] by element
            # t = [for t in range] => self.t += 1
            m_hat = self.m[key]/(1 - self.beta1 ** self.t) # t가 증가할수록 denominator converges to 0, thus the formula gets closer to m => which on the other hand means that the m_hat gets closer to its gradient
            v_hat = self.v[key]/(1 - self.beta2 ** self.t)

            params[key] -= (self.lr /(np.sqrt(v_hat) + epsilon)) * m_hat
                        # 새로운 lr                            # 새로운 gradient

if __name__ == '__main__':
    params = {'x': -7., 'y': 2.0} # 파라미터 초기값
    gradients = {'x': 0.0, 'y': 0.0} #gradients 초기값

    # Adam 클래스의 인스턴스를 생성
    adam = Adam(lr = 0.3) # 생성자 호출
    # Adam() has three parameters: lr, beta1, beta2; all with its default values

    # 학습하면서 파라미터 (x,y)들이 업데이트되는 내용을 저장하기 위한 리스트
    x_history = []
    y_history = []
    for i in range(30):
        x_history.append(params['x'])
        y_history.append(params['y'])
        gradients['x'], gradients['y'] = fn_derivative(params['x'], params['y'])
        adam.update(params, gradients)
        # 파라미터 값 출력
        print(f"({params['x']}, {params['y']})") # 가장 마지막에 업데이트된 내용 1개는 보이지 않는다 => 제일 처음값은 보이지 않는다 -> 출력된 위치 때문에
        # x좌표는 커지고, y좌표는 적어지고 있다 -> 0으로 가고 있다
        # 두 값 모두 많이 가지는 못했다

    # 그래프 그려서 모델의 이동모양 확인하기
    # contour그래프/ 등보선 그래프 활용
    x = np.linspace(-10, 10, 2000)
    y = np.linspace(-5,5,1000)
    X, Y = np.meshgrid(x,y) # 이거랑 이 아래 코드 잘 모르겠다
    Z = fn(X, Y)

    mask = Z > 8
    Z[mask] = 0

    plt.contour(X, Y, Z, 10)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Adam')
    plt.axis('equal')
    # x_history, y_history를 plot
    plt.plot(x_history, y_history, 'o-', color = 'red')

    plt.show()