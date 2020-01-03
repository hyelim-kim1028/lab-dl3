"""
softmax_loss
"""
import numpy as np
from ch03.ex11 import softmax
from ch04.ex03 import cross_entropy


class SoftmaxWithLoss:
    def __init__(self):
        self.y_true = None #정답 (예측을 하려는 정답 레이블을 저장하기 위한 필드). one-hot-encoding이라고 가정
        self.y_pred = None #softmax 함수의 출력(예측 레이블)을 저장하기 위한 필드
        self.loss = None #cross_entropy 함수의 출력(손실, 오차)를 저장하기 위한 필드

    def forward(self, X, Y_true):
        self.y_true = Y_true
        self.y_pred = softmax(X)
        self.loss =cross_entropy(self.y_pred,self.y_true)
        # 실제값을 요청해야한다
        return self.loss

    def backward(self, dout = 1):
        if self.y_true.ndim == 1: #1차원 ndarray
            n = 1
            # 1차원 array를 n으로 나누면 자기의 원소 숫자 만큼 나누는 것과 같아져서 안됨
        else: # 2차원 ndarray
            n = self.y_true.shape[0] #one-hot-encoding 행렬의 row의 갯수
        dx = (self.y_pred - self.y_true)/n # 오차들의 평균 # 행렬?도 가능 (미니배치처럼 이미지를 100장도 보낼 수 있음 -> 100장의 오차들의 평균)
        return dx

    # 가장 마지막 노드를 얻으려고 하는 것
    # 분류 문제에서는 soft max를 사용한다
    # 실제 레이블들을 입력을 받아서 CE (손실)을 계산한다 -> 비용이 얼마나 드느냐
    # LOSS가 역전파의 마지막이므로, 1이 들어온다 (이 값자체도 임의의 값이라고 하자)
    # y가 1만큼 변할 때, 손실의 변화 => y-t

if __name__ == '__main__':
    np.random.seed(103)
    x = np.random.randint(10, size =3)
    print('x =', x)

    y_true = np.array([1., 0., 0.]) # one-hot-encoding
    # forward
    print('y =', y_true)

    swl = SoftmaxWithLoss() #객체 생성, self 이외에 다른 파라미터는 없으므로 주지 않아도 된다
    loss = swl.forward(x, y_true)  #forward propagation
    print('loss =', loss)
    # 실제값 * 에측값 해서 다 더한 것 = 손실

    print('y_pred =', swl.y_pred)
    # ENTROPY는 0에 가까워져가면 더 크다, 1에 가까워질수록 적어진다
    # 이유는 0에 가까워질수록 무한대에 더 가까워지기 때문 (sigma t*log(p) 일 때는 -무한대에, -sigma t*log(p) 일 때는 + 무한대에)
    # 우리가 예측한 값이 틀렸다면 불확실함-> 엔트로피도 크다

    dx = swl.backward() #back propagation(역전파)
    print('dx =', dx)

    # 손실이 가장 큰 경우
    print()
    y_true = np.array([0, 0, 1])
    loss = swl.forward(x, y_true)
    print('y_pred =', swl.y_pred) # x가 변화하지 않았기 때문에 예측값들은 변화하지 않는다, softmax의 리턴값은 변화하지 않는다
    print('loss = ', loss) # 실제값과 예측값이 완전히 다르다 => 불활실성이 커졌다 -> 엔트로피가 커졌다
    # 0으로 가까워지면 불확실성이 적어진다
    print('dx =', swl.backward())

    # 손실이 가장 작은 경우
    print()
    y_true = np.array([0, 1, 0])
    loss = swl.forward(x, y_true)
    print('y_pred =', swl.y_pred)
    print('loss = ', loss)
    print('dx = ', swl.backward())

    # 예측과 실제가 비슷하면 손실이 적고, 벗어나면 손실이 큰것
    # 손실이 적다 불확실성이 적다, 손실이 크다 불확실성이 크다
    # 손실이 적으면 변화율도 적고, 손실이 적으면 변화율이 적다 (+- 보다는 숫자로 비교)
    # 예측이 적으면 변화율도 적다 # 이러면 W가 변화되는 양, b가 변화되는 양 모두 준다 => 그런식으로 수정해나가며 계산할 것이다




