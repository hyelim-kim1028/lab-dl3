"""
"""

import numpy as np
from ch04.ex05 import numerical_gradient
import matplotlib.pyplot as plt

def fn(x):
    """ x = [x0,x1] """
    if x.ndim == 1:
        return np.sum(x**2)
    else:
        return np.sum(x**2, axis = 1)


x0 = np.arange(-1, 2)
print(x0) #[-2 -1  0  1  2]
x1 = np.arange(-1, 2)
print('x1 =', x1)

X, Y = np.meshgrid(x0, x1) #x0를 x1만큼 만복
print('X =', X)
print('Y =', Y)
# meshgrid: 매게 변수가 2개 짜리인 함수가 있을 때 사용
# https://docs.scipy.org/doc/numpy/reference/generated/numpy.meshgrid.html

# Y = np.meshgrid(x0, x1)
# print('Y =', Y)
# X = x0가 x1만큼 반복; Y = x1의 원소들로 반복

# Error:
# gradients = numerical_gradient(fn, np.array([X,Y]))
# print(gradients)
# 안되는 이유: X(2차원 배열) + Y(2차원 배열)을 다시 배열로 묶었다 -> 고려 대상 밖

# X와 Y의 2차원 배열을 1차원으로 풀어준다
X = X.flatten()
Y = Y.flatten()
print('X = ', X)
print('Y = ', Y)
# (X,Y) 를 좌표로 묶으려고 하는 것
XY = np.array([X,Y])
print('XY =', XY)

gradients = numerical_gradient(fn, XY)
print('gradients = ', gradients)

x0 = np.arange(-2, 2.5, 0.25)
# 구간을 더 잘게 쪼개기
# print('x0 =',x0)
x1 = np.arange(-2, 2.5, 0.25)
# print('x1 =', x1)
X,Y = np.meshgrid(x0, x1)
X = X.flatten()
Y = Y.flatten()
XY = np.array([X,Y])
gradients = numerical_gradient(fn, XY)
plt.quiver(X, Y, -gradients[0], -gradients[1],
           angles = 'xy')
# quiver: https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.quiver.html
plt.ylim([-2, 2])
plt.xlim([-2,2])
plt.xlabel('x0')
plt.ylabel('x1')
plt.show()
# 각점의 기울기를 평면에 투영해본 것
# 2D reflection of arrows in gradient descent
# x가 클수록 기울기가 급박?하고, 작을수록 기울기가 완만하다
# 방향은 (0,0) -> minimum value로 향해가고 있다 (f(x) = x0**2 + x1**2)

# gradient 에 방향이 있다면 변화률을 이렇게 볼 수 있다
# 각 방향에 대해서 얼마만큼 변하느냐
# 3차원은 2차원으로 투영이 가능하니까 이렇게 볼 수 있지만, 차원수가 더 커지면 이 방법은 사용 불가

# 교재의 github에서 소스코드를 받아볼 수 있다


