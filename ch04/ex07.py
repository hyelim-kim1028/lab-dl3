"""
경사 하강법 (gradient descent)
- 임의의 점에서 gradient를 계산해서, arrow의 direction을 gradient를 보고 결정/계산을 해서
- 초기값에서 시작해서 기울기를 바꿔가며 최소/최대값을 찾는 것

x_new = x - lr * df/dx
# 상수값(learning rate)을 곱해서 방향을 바꾸고, 그 과정을 반복하면 함수f(x)의 최솟값을 찾게 된다
epoch = 반복값
"""

import numpy as np
from ch04.ex05 import numerical_gradient
import matplotlib.pyplot as plt

def gradient_method(fn, x_init, lr = 0.01, step = 100):
    """
    :param fn:
    :param x_init:
    :param lr: learning rate
    :param step: 몇번 반복할 것인지
    :return:
    """
    x = x_init #점진적으로 변호시킬 변수
    x_history = [] #x가 변화되는 과정을 저장할 배열 #나중에 로그를 보기 위함 -> 과정의 기록
    for i in range(step): #step 횟수만큼 반복
        x_history.append(x.copy()) #x의 복사본을 x 변화 과정에 기록
        # if we dont use copy(): only shows the same number -> the address of array x
        # because x is an array -> if we only append x [x_history.append(x)], the array returns the address of array x -> @123 @123 @123 -> 이렇게 append 되는 것
        # -> but what we need is the "values" of array x  -> we have to use the function copy() -> @123 @124 @125 ...
        grad = numerical_gradient(fn, x) #점 x에서의 gradient 계산 (gradient가 있어야 새로운 점을 계산할 수 있다)
        x -= lr * grad #x_new = x_init - lr * grad: x를 변경
    return x, np.array(x_history)
        # x -> min val


def fn(x):
    if x.ndim == 1:
        return np.sum(x**2)
    else:
        return np.sum(x**2, axis = 1)





if __name__ == '__main__':
    init_x = np.array([4.])
    # when we put 4 -> error, so we changed to 4.0
    # numpy.core._exceptions.UFuncTypeError: Cannot cast ufunc 'subtract' output from dtype('float64') to dtype('int32') with casting rule 'same_kind'
    x, x_hist = gradient_method(fn, init_x,lr = 0.1)
    print('x =',x)
    print('x_hist =', x_hist)

# when lr is too big, the graph explodes (goes up and never reach the min value) (hyper-parameter)
# when lr is too small, the movement is too slow, it takes too much time finding the min val
# lr - we can find the appropriate value 1) from experiments manually 2) make a code for it

# resumen:
# 학습률(learning rate/lr)이 너무 작으면 최솟값을 찾아가는 시간이 너무 오래 걸리고
# 너무 크면, 최소값을 찾지 못하고, 발산하는 경우가 생길수가 있음


    init_x = np.array([4., -3.])
    x, x_hist = gradient_method(fn, init_x, lr = 0.1, step = 100)
    print('x =', x)
    print('x_hist =', x_hist)

    #x_hist (최솟값을 찾아가는 과정)을 산점도 그래프
    plt.scatter(x_hist[:,0], x_hist[:, 1]) # (X좌표, Y좌표) 선택
    plt.xlim([-5,5])
    plt.ylim([-5,5])
    plt.axvline(color = '0.8') # v for vertical
    plt.axhline(color = '0.8')
    plt.show()
    # 맨 오른쪽 (-4, 3)에서 시작해서 0으로 향해가는 그래프 /(0,0)이 최솟값

    # 시작값을 바꿈
    init_x = np.array([-4., 3])
    x, x_hist = gradient_method(fn, init_x, lr=0.1, step=100)  # lr 이 너무 작으면 접근해나간느 속도가 매우 느리다 (i.e. 0.0001)  #lr =1. 로 두면, 두 점만 왔다갔다,,, # lr > 1 이면 explodes (1.01 만 되도 발산)
    # 발산, 수렴
    print('x =', x)
    print('x_hist =', x_hist)

    # x_hist (최솟값을 찾아가는 과정)을 산점도 그래프
    plt.scatter(x_hist[:, 0], x_hist[:, 1])  # (X좌표, Y좌표) 선택
    plt.xlim([-5, 5])
    plt.ylim([-5, 5])
    plt.axvline(color='0.8')  # v for vertical
    plt.axhline(color='0.8')
    plt.show()
    # 맨 왼쪽 (-4., 3)에서 시작해서 0으로 향해가는 그래프 /(0,0)이 최솟값

    # 동심원 그리기
    plt.scatter(x_hist[:, 0], x_hist[:, 1])  # (X좌표, Y좌표) 선택
    # 동심원: x**2 + y**2 = r**2 -> y**2 = r**2 - x**2
    for r in range(1,5):
        r = float(r) # convert to integer -> float
        x_pts = np.linspace(-r, r, 100)
        y_pts1 = np.sqrt(r**2 - x_pts**2)
        y_pts2 = -np.sqrt(r**2 - x_pts**2)
        plt.plot(x_pts, y_pts1, ':', color = 'gray')
        plt.plot(x_pts, y_pts2, ':', color = 'gray')

    plt.xlim([-5, 5])
    plt.ylim([-5, 5])
    plt.axvline(color='0.8')  # v for vertical
    plt.axhline(color='0.8')
    plt.show()























