"""
신경망 학습 목적: 신경망/레이어들을 통과하면서  예측이 얼마나 정확한가/부정확한가에 대한 손실을 계산
               : 손실을 가지고 W와 b가 변화시켜가면서 손실값을 향상시켜준다 (손실값을 가능한 낮추는 파라미터 W,b등을 찾는 것)

파라미터(parameter):
    - 파라미터 : 가중치(weight), 편향(bias)
    - 하이퍼 파라미터 (hyper parameter) : 우리가 원하는 학습률, epoch, batch 크기, 신경망에서의 뉴런 갯수, 신경망의 은닉층 갯수
     -> 하이퍼 파라미터를 어떻게 잘 찾아낼 것이냐? 자동으로 찾아낼 수 있게 할 것이냐?

ch06의 목표: 파라미터를 어떻게 갱신할 것인가?
 - 파라미터를 갱신하는 방법들: SGD, Momentum, AdaGrad, Adam
        # SGD: W = W - rl * dW 의 내용
        # from ch05 ex11_1
         for key in neural_net.params:
                neural_net.params[key] -= learning_rate * gradients[key]
        # W = W - lr*dL/dW
        # b = b - lr * dL/db
 - 하이퍼 파라미터를 최적화 시키는 방법

 # 왜 SGD 이 외에 방법들이 나왔는가? SGD를 구현해보면서 설명
"""
import numpy as np
from ch06.ex01 import fn_derivative, fn
import matplotlib.pyplot as plt

class Sgd:
    """
    # SGD : Stochastic Gradient Descent (확률적 경사 하강법)
    W = W - lr * dl/dW      #dl/dW = gradient(기울기)
    W: 파라미터 (가중치 or 편향, either)
    lr: 학습률(learning rate) # 너무 크면 발산 or 너무 작으면 오래걸림
    dl/dW: 변화율(gradient) -> 찾는 방법: 1) 오차역전파(미분연쇄법칙) 2) (f(x+h) - f(x))/ h

    """
    def __init__(self, learning_rate = 0.01):
        # lr을 개발자로부터 입력받을 수 있도록 한다
        self.learning_rate = learning_rate

    def update(self, params, gradients):
        """ 파라미터 params와 변화율 gradients가 주어지면,
            파라미터들을 갱신하는 메소드.
            params, gradients: 딕셔너리 {key: value, ...} """
        for key in params:
               # W = W - lr * dL/dW
            params[key] -= self.learning_rate * gradients[key]


class momentum():
    """
    Momentum: 속도에다 어떤 값을 곱한 것?
    # 학습률을 낮추더라도, 속도라는 개념을 도입하면 더 빨리 최소값을 찾아 갈 수 있지 않을까?
    # 원과 원사이가 더 넒은 곳이 원만한 기울기를 갖고, 원과 원 사이가 더 좁은 곳이 더 steep 한 것
    """
    pass


if __name__ == '__main__':
    # sgd 클래스의 객체(인스턴스)
    sgd = Sgd(learning_rate=0.95) # init 메소드가 호출되고, default arg를 가지기 때문에 값이 있는 경우임 -> 그러기 때문에 업데이트가 실행될수도 있다
    # 0.01, 0.1, 0.95 로 숫자를 바꿔가며 실험

    #ex01 모듈에서 작성한 fn(x, y)함수의 최솟값을 임의의 점에서 시작해서 찾아감.
    init_position = (-7, 2)
    # 신경망에서 찾고자 하는 파라미터의 초깃값
    params = dict()
    params['x'], params['y'] = init_position[0], init_position[1]

    # 각 파라미터에 대한 gradient
    gradients = dict()
    gradients['x'],gradients['y'] = 0, 0

    # 각 파라미터 갱신 값들을 저장할 리스트
    x_history = []
    y_history = []
    for i in range(30):
        x_history.append(params['x'])
        y_history.append(params['y'])
        gradients['x'], gradients['y'] = fn_derivative(params['x'], params['y']) #fn_derivative returns a tuple of (x에 대한 편미분, y에 대한 편미분)
        # module01의 fn(x,y)을 사용
        # 최초에 (-7, 2)을 넣고, x와 y에 대한 편미분을 각각 해주면서 새로운 gradients를 구한다
        # (-7,2) -> (0.7, 4) ... 이런식으로 param x와 y가 갱신된다
        # X = X - lr * af/ax, Y = Y - lr * af/ay
        sgd.update(params, gradients)

    # print(x_history)
    # print(y_history) # 값들이 0에 가까워진다
    # (x,y) 다시 출력
    for x, y in zip(x_history, y_history):
        print(f'({x}, {y})')
    # 최솟값(0,0)을 찾아가는 중,,,

    # f(x,y)함수를 동고선으로 표현
    x = np.linspace(-10, 10, 200) # -10부터 10까지의 점을 200개의 구간으로 나눔
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = fn(X,Y)

    mask = Z > 7 # Z가 7보다 큰 아이들은 0 으로 줘서 -> 그래프에서 불필요한 원이 안보이게 했다
    Z[mask] = 0

    plt.contour(X, Y, Z, 15, cmap = 'binary')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')

    # 동고선 그래프에 파라미터(x,y)들이 갱신되는 과정을 추가
    plt.plot(x_history, y_history, 'o-', color = 'red')
    plt.show()
    # learning_rate = 0.01 일 때, 거의 변화가 없다. 0.1 일때 -> 0을 향해 가는 어떤 것은 보인다. 0.95 -> y좌표가 출렁거리고 있다
    # 왔다갔다 하는 것 보다는 한쪽 계곡만 타고 내려오는 방법은 없을까? 다른 알고리즘이 파생이 됨
    # 왔다갔다하면 느리기도 하고, there is a potential for the exponential to explode




















