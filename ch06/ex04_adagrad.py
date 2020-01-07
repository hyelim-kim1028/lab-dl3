"""
파라미터 최적화 알고리즘
 3) ada grad (Adaptive Gradient)
# 파라미터: W, b 와 같은 값들을 일컫는다
    SGD: W = W - lr * gradient # lr/학습률이 고정되어 있다
    # 실행할때, 반복할 때 lr을 바꾼다 -> ada grad
    AdaGrad에서는 학습률을 변화시키면서 파라미터를 최적화함
    처음에는 큰 학습률로 시작, 점점 학습률을 줄여나가면서 파라미터를 갱신한다
    h = h + grad * grad             #h += grad **2 # 왜 이렇게 안해
    lr = lr/sqrt(h)
    W = W - (lr/sqrt(h)) * grad
            # grad가 변화하면서 h의 값도 변화하므로, 계속 변화하는 값이 나온다
"""
import matplotlib.pyplot as plt
import numpy as np
from ch06.ex01 import fn, fn_derivative

class AdaGrad:
    def __init__(self, lr = 0.01):
        self.lr = lr #학습률
        # h 를 저장하고 있다가 이전에 저장하고 있던 h를 가지고 변경해야한다. 그러므로 h값도 field로 가지고있어야한다
        self.h = dict() # h 는 gradient -> gradient라는 x방향의 gradient, y 방향의 gradient 등으로 값이 나올 것이다

    def update(self, params, gradients):
        if not self.h:
            for key in params:
                self.h[key] = np.zeros_like(params[key])
        for key in params:
            self.h[key] += gradients[key] * gradients[key]  # h += grad **2 # 왜 이렇게 안해
            # 행렬의 dot 곱하기가 아니라 element-wise 곱하기 (왜지)
            # gradient 가 제곱이 되고 있어서 h는 항상 양수가 들어와야한다
            epsilon = 1e-8
            # 계속해서 작아지는 값인데, h가 0이 되면 안되므로 epsilon을 더해주었다 (분모가 0이 되면 무한대가 되어버린다)
            params[key] -= (self.lr / np.sqrt(self.h[key] + epsilon)) * gradients[key]
        # 계속 이 부분에서 dict/float 이 아니라 계산이 안되는 에러가 발생했는데, self.h값에도 [key]를 줘서 element-wise 곱셈을 했다
        # 컴퓨터가 계산해주는 실수 계산은 정확한 계산이 없다. 어짜피 근사값을 계산하는 것 이므로, 아주 작은 값(epsilon) 정도는 더해줘도 상관 없다


if __name__ == '__main__':
    adagrad = AdaGrad(lr = 1.5)

    params = {'x':-7.0, 'y':2.0} # parameter 초기값
    gradients = {'x':0.0, 'y':0.0} #gradient의 초기값
    x_history = [] 
    y_history = []
    for i in range(30):
        x_history.append(params['x'])
        y_history.append(params['y'])
        gradients['x'], gradients['y'] = fn_derivative(params['x'], params['y'])
        adagrad.update(params, gradients)
        # 파라미터 갱신 과정 출력
        print(f"({params['x']}, {params['y']})")

    x = np.linspace(-10, 10, 2000)
    y = np.linspace(-5, 5, 1000)
    X,Y = np.meshgrid(x,y)
    Z = fn(X,Y)

    mask = Z > 7
    Z[mask] = 0

    plt.contour(X,Y,Z, 10)
    plt.title('ADA Gradient')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.plot(x_history, y_history, 'o-', color = 'red')
    plt.show()





