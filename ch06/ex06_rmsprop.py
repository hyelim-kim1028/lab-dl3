"""
파라미터 최적화 알고리즘 5) RMS prop
rmsprop -> RMS Propagation
파라미터 최적화의 가장 기본 적인 방법은 SGD: W = W - lr * dL/dW => 이 값을 새로운 파라미터로 설정
    SGD의 단점은 학습률(lr)을 학습하는 동안에 변경할 수 없다.
    이러한 단점들을 극복하기 위해서 다양한 variation들이 등장했는데, 그것이 ada grad
    AdaGrad: W = W - (lr/sqrt(h)) * dL/dW
    - h를 이용해서 학습하는 동안에 학습률을 계속 바꿔버리자 (h = h + (dL/dW)**2))
    - AdaGrad의 단점: 학습을 오래하다 보면 h가 굉장히 커질 수 있고, 그러면 갱신되는 양이 0이 되는 경우가 발생 -> 더 이상 학습 효과가 발생하지 않는다 (= 학습이 되지 않는다)
    different variations:
    momentum (속도라는 개념을 도입해서 SGD의 속도문제를 해결)

    AdaGrad의 갱신량이 0이 되는 문제를 해결하기 위한 알고리즘: RMSprop
     rho: decay-rate(감쇄율)를 표현하는 하이퍼 파라미터
     h = rho * h + (1 - rho) * (dL/dW)**2
    # rho가 발생할 확률 (1-rho) 가 발생하지 않을 확률,,,??
    # 왜 0 이 되지 않습니까? 둘 중 하나가 남아있기 때문에
    # 결국 h는 엄청 작은 값으로 이루어지게 될 텐데, 그 문제를 (1-rho)로 해결해보자~
    # h -> 학습량을 변화시키는데 사용
   W - (lr/sqrt(h)) * dL/dW
   h = rho * h + (1 - rho) * (dL/dW)**2
"""
import numpy as np
from ch06.ex01 import fn,fn_derivative
import matplotlib.pyplot as plt

class RMSProp:
    def __init__(self, lr = 0.01, rho = 0.99):
        self.lr = lr #학습률 (learning rate)
        self.rho = rho # decay rate #h 가 커지는 것을 방지
        self.h = dict()

    def update(self, params, gradients):
        if not self.h:
            for key in params:
                self.h[key] = np.zeros_like(params[key])

        epsilon = 1e-8
        for key in params:
            self.h[key] = self.rho * self.h[key] + (1 - self.rho) * gradients[key] ** 2
            params[key] -= (self.lr / (np.sqrt(self.h[key])+epsilon)) * gradients[key]


if __name__ == '__main__':
    params = {'x': -7., 'y':2.} # 파라미터 초기값
    gradients = {'x': 0.0, 'y':0.0} # gradients 초기값

    # RMS Prop 클래스의 인스턴스를 생성
    rms = RMSProp()

    # 학습하면서 파라미터 (x,y)들이 업데이트되는 내용을 저장하기 위한 리스트
    x_history = []
    y_history = []
    for i in range(30):
        x_history.append(params['x'])
        y_history.append(params['y'])
        gradients['x'], gradients['y'] = fn_derivative(params['x'], params['y'])
        rms.update(params, gradients)
        #파라미터값 출력
        print(f"({params['x']}, {params['y']})")

    x = np.linspace(-10,10,2000)
    y = np.linspace(-5, 5, 1000)
    X, Y = np.meshgrid(x,y)
    Z = fn(X,Y)

    mask = Z > 8
    Z[mask] = 0

    plt.contour(X, Y, Z, 10)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('RMS')
    plt.axis('equal')
    plt.plot(x_history, y_history, 'o-', color = 'red')

    plt.show()

# error occured




