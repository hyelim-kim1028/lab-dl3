import numpy as np

# p. 83 포스트잇 그림에 대해 코드짜기
from ch03.ex01 import sigmoid

x = np.array([1,2])
# z = x @ W + b 라는 행렬식을 만족 할 수 있도록 W를 만들고, z 를 출력하라
# @ = dot
W = np.array([[1, 2, 3],
              [4, 5, 6]])
b = np.array([1,2,3])
a1 = x.dot(W) + b
print('a1 =', a1)

# 출력 a1에 활성화 함수를 sigmoid 함수로 적용
# z1 = sigmoid(a1)
z1 = sigmoid(a1)
print('sigmoid(a1) =',z1)
# sigmoid에서 5가 넘어가는 숫자들은 거의 비슷한 숫자들이 나온다
# 그렇기 때문에 z1의 결고값이 별로 다르지 않다

# 또 다른 은닉계층을 하나 더 넣는다
# 두번째 은닉층에 대한 가중치 행렬 W2와 bias 행렬 b2를 작성
W2 = [[0.1, 0.4],
      [0.2, 0.5],
      [0.3, 0.6]]
b2 =np.array([0.1, 0.2])
a2 = z1.dot(W2) + b2
print('a2 =', a2)

# a1에 활성화 함수(sigmoid)적용
y = sigmoid(a2)
print('sigmoid(a2) =',y)

# w와 b를 찾는 것이 machine learning 의 목표












