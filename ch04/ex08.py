"""
weight 행렬에 경사 하강법 (gradient descent) 적용
apply gradient descent in neural networks (신경망)
"""
import numpy as np
from ch03.ex11 import softmax
from ch04.ex03 import cross_entropy
from ch04.ex05 import numerical_gradient

class SimpleNetwork:
    def __init__(self):
        np.random.seed(1230)
        self.W = np.random.randn(2, 3) #(row,column)
                        # randn/ n for normal distribution
        # 가중치 행렬의 초기값들을 임의로 설정

    def predict(self, x):
        z = x.dot(self.W)
        y = softmax(z)
        return y
        # W가 ~라고 가정했을 때, 예측값 => y

    # 손실함수 만들기
    def loss(self, x, y_true):
        """ 손실 함수 (loss function) - Cross Entropy
            (분류 문제라고 가정) """
        y_pred = self.predict(x) #입력이 x일 때 y_pred 소환
        ce = cross_entropy(y_pred, y_true) # 크로스 엔트로피 계산 # 오차를 계산해서 리턴
        return ce
    # ce를 줄여가는 gradient (기울기)를 찾기

    def gradient(self, x, t):
        """ x: 입력, t: 출력 실제 값 (정답 레이블) """
        fn = lambda W: self.loss(x, t) #오차의 최솟값 구하기 -> 손실함수 사용 (loss)
                                       # [0. , 0., 1. ]
        return numerical_gradient(fn, self.W)
                                # fn: entropy 함수

    # my_solution
    # def gradient(self, x, t):
    #     sum = 0
    #     for x_i in x:
    #         fn = lambda W: self.loss(x_i, t)
    #     sum += sum
    #     # x -= lr * grad
    #     return numerical_gradient(sum, self.W)
    #     # gradient를 W를 바꾸는데 사용하기

if __name__ == '__main__':
      # SimpleNetwrok 클래스 객체를 생성
      network = SimpleNetwork() #생성자 호출 -> __init__() 메소드 호출
      print('W =', network.W)
      # randomly created W

      # x = [0.6, 0.9일 때 y_true = [0, 0, 1]라고 가정
      x = np.array([0.6, 0.9])
      y_true = np.array([0.0, 0.0, 1.0])
      print('x =', x)
      print('y_true', y_true)

      y_pred = network.predict(x)
      print('y_pred', y_pred)
      # 우리가 하고 싶은 일: y_pred를 0,0,1에 맞춰가는 것
      # error of (y-pred - y_true) 를 기준으로 맞춰간다 => cross entropy

      ce = network.loss(x, y_true)
      print('cross entropy =', ce)

      lr = 0.1
      for i in range(101):
          # gradient(x, y_true, 100) # 음
          g1 = network.gradient(x, y_true)
          # print('g1', g1)
    # red point beside the line number -> brake point
    # the lines are automatically stopped after the line
    # debug (right click or upper right of the GUI)
    # result of lines above the brake point is returned
    # three arrows: croocked -> run the code/ step-into, downward -> read the details of the function, downward with catalogue -> read the details of my functions (manually made by me)
      # arrow upward -> get out of the code
          # lr = 0.1 #learning rate
          network.W -= lr * g1 # 2차원 # W = w - lr * gradient
          print(f'W = {network.W},\n y_pred = {network.predict(x)}, \nce = {network.loss(x,y_true)}' )


# W는 가중치의 행렬이다
# 실제값에 근접한 결과/예측값이 나올 수 있도록 W를 바꿔주는 과정
# W값을 어떻게 변경할 것인가 (+-a?) -> 오차를 줄여주는 방향의 척도로 loss function을 사용
# cross entropy: 오차를 계산해주는 함수
# gradient를 계산할 때 손실함수를 사용한다
# gradient: 손실함수가 어떻게 변화하는가 (변화율)
# -> 오차/손실/비용함수에 변화를 주는 것 (W -> g)
# 모양은 W와 똑같이 나온다

# gradient: 어떤 접선의 기울기
# 우리가 하고 싶은 것: 어떤 그래프 y = f(x) 의 최솟값을 찾아가는 것
# 가중치의 모든 점에서 gradient를 계산한다

# bias까지 고려한다면, 손실함수를 주고, bias 행렬을 주고, W와 같은 과정을 거쳐야한다



