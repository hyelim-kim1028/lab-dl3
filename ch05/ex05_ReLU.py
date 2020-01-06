"""
ReLU function 구현
"""
import numpy as np


class Relu:
    """
    ReLU (Rectified Linear Unit)
    - Rectified: 어떤 특정값 이하에서는 리턴을 내보내지 않겠다, 이상에서는 내보낸다
    - 정류/신호가 흐르게 하는 임계값
    relu(x) = x (if x > 0), 0 # x가 0보다 크면 x, 아니면 0 = max(0, x) -> relu의 forward가 됨
    relu_prime(x) = 1 (x > 0 이면 1이고 아니면 0) -> relu의 도함수, backward가 됨
        # 미분한 relu 함수
    # 장점: 암산으로도 계산이 가능하다
    - 행렬이 들어와도 계산이 도니다
    """
    def __init__(self):
        # relu 함수의 input 값(x)가 0보다 큰지 작은지를 저장할 field
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0) # x가 0보다 작으면 True, 크면 False (음수면 0, 양수면 1)
                # x가 0보다 작은 아이들을 저장 => 후에 backward에서 사용
        return np.maximum(0,x)

    def backward(self, dout):
        # print('making 전:', dout)
        dout[self.mask] = 0  # 델타값이 들어오면 # 원래 x에서 마이너스였으면 0, 나머지는 그대로
        # print('masking 후:', dout)
        dx = dout # 들어온 값을 그대로 내보낸다
        return dx







if __name__ == '__main__':
    # ReLU 객체를 생성
    relu_gate = Relu()
    # x = 1 일때, Relu 리턴값
    y = relu_gate.forward(1)
    print('y=',y)

    # array를 Relu함수에 넘겨보기
    np.random.seed(103)
    x = np.random.randn(5)
    print('x =', x)
    y = relu_gate.forward(x)
    print('y =', y) # print relu 함수의 리턴값
    print('mask = ', relu_gate.mask) #relu_gate의 필드 mask
    # 왜 리턴된 값만 false지,,,
    # mask는 무엇이낙

    # backward propagation (역전파) 함수 테스트
    delta = np.random.randn(5)
    dx = relu_gate.backward(delta)
    # masking 전이 양수냐 음수냐가 중요한것이 아니라, 제일 처음 forwarding 될 때의 x의 값들이 중요하다
    # 지금 하는 것: gradient 계산
    # 나갔을 때의 값을 기억하고 있다가 backward할 때 T/F를 골라낸다
    # False면 선택하지 않으니까 남아있는 것





