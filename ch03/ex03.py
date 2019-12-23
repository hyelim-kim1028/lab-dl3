"""
# 그림 3-14 (p.82)
"""
import numpy as np

x = np.array([1,2])
W = np.array([[1, 4],
              [2, 5],
              [3, 6]])
y = W.dot(x) + 1

b = 1
y = W.dot(x) + b
print(y)

x = np.array([[1, 2]])
print(x.shape)
W = np.array([[1,2,3],
              [4,5,6]])
print(W.shape)
y = x.dot(W) + 1
print(y)

# bias 를  y1, y2, y3에 모두 다른 값을 준다면
# 그리고 각 y에서 activation함수를 거쳐서 출력값을 준다



