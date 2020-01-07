"""
파라미터 최적화 알고리즘
4) ADAM
- Adapative Moment Estimate/ - 학습률 변화 + 속도(모멘텀) 개념 도입
- AdaGrad + Momentum

W: 파라미터
lr: 학습률 (learning rate)
t: timestamp (반복할 때마다 증가하는 숫자. update 메소드가 호출될 때마다 +1)
beta1, beta2: 모멘텀을 변화시킬 때 사용하는 상수들
m: 1st momentum
v: 2nd momentum
m = beta1 * m + (1 - beta1) * grad
v = beta2 * v + (1 - beta2) * grad * grad
m_hat = m/(1-beta1 ** t)
v_hat = v/(1-beta2 ** t)
W = W - lr * m/sqrt(v)
    # m을 grad라고 생각하면, ada grad 와 같은 모습을 모인다
    # sqrt(v) -> grad ^ 2 이기 때문에, v에 루트를 씌움
"""

import numpy as np

class Adam:
    def __init__(self, lr = 0.01, beta1 = 0.9, beta2 = 0.99):
        self.lr = lr # learning rate (학습률)
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = dict()
        self.v = dict()


    def update(self, params, gradients):
        """
        m,v, t += 1
        """
        if not self.m and self.v:
            for key in params:
                self.m[key] = np.zeros_like(params[key])
                self.v[key] = np.zeros_like(params[key])

        for key in params:
            m = self.beta1 * m + (1 - self.beta1) * gradients[key]
            v = self.beta2 * v + (1 - self.beta2) * gradients[key] * gradients[key]
            m_hat = m/(1 - self.beta1 ** t)
            v_hat = v/(1 - self.beta2 ** t)
            epsilon = 1e-8
            params[key] -= self.lr * m/(np.sqrt(v) + epsilon)
