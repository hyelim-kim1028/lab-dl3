"""
perceptron
 - 두개의 입력값이 있다 (x1, x2)
 - 출력: y = x1 * w1 + x2 * w2 + b    # w: 가중치/ weight, b: bias
         1. a = x1 * w1 + x2 * w2 + b  계산
         2. y = 1 (a > 임계값)  or 0 (a <= 임계값)   #임계값 = threshold
                # the condititions for a may differ from one situation to the other
                # the function decides on the condition is 활성화 함수
# 신경망의 뉴런(neuron)에서는 입력 신호의 가충치 합을 출력값으로 변환해주는 함수가 존재
 -> 활성화 함수 (activation function)
"""

import numpy as np
import math
import matplotlib.pyplot as plt

# 가장 많이 사용하는 활성화 함수: 계단 함수
def step_function(x):
    """
    step function
    :param x: numpy.ndarray
    :return: step function 출력 (0 또는 1)로 이루어진 numpy.ndarray
    """

    # if x > 0: # ndarray가 되는 true/false 리스트를 리턴해주는데, 뭐가 뭔지 모르겠을 때: The truth value of an array with more than one element is ambigous when trying to index an array 와 같은 에러가 난다
    #     return 1
    # else:
    #     return 0 # 문제가 되는 코드: 리스트가 된다
    y = x>0 #[false, false, ... True]
    return y.astype(np.int) # astype: boolean type -> int 로 변환

    # method 2
    # result = []
    # for x_i in x:
    #     if x > 0:
    #         result.append(1)
    #     else:
    #         result.append(0)
    # return np.array(result) # 왜 에러가 나지
    # OR result = [1 if x_i > 0 else 0 for x_i in x]


def sigmoid(x):
    """ sigmoid = 1/ (1 + exp(-x)) """
    # exp가 두 군데에 있다 -> numpy & math
    # return 1/ (1+math.exp(-x)) # math.exp accepts a scala # exp accepts array, tuple, etc.
    return 1/(1 + np.exp(-x)) #goodgood

    # bad code/ because numpy has an interable function
    # result = []
    # for x_i in x:
    #     result.append(1/np.exp(-x_i))
    # return 1/(1+np.exp(-1))

def relu(x):
    """ ReLU (Rectified Linear Unit)
        y = x, if x > 0
          = 0, otherwise """
    # 전기공학에서 전기의 양을 조절해주는? 정류해주는 (전기를 정제해주는) 역할
    # write a code that accepts an array

    # my_thought 1
    # if x > 0:
    #     return x.astype(np.int)
    # else:
    #     return 0

    # answer
    return np.maximum(0, x)


if __name__ == '__main__':
    x = np.arange(-3, 4)
    print('x =', x)

    # for x_i in x:
    #     print(step_function(x_i), end = ' ')

    print('y =', step_function(x))

    # Error
    # what we want #[0 0 0 0 1 1 1]

    # teacher's method
    # x가 numpy의 배열이다. numpy에서는 scalar 값이 1개 밖에 없다, 왼쪽은 벡터
    # numpy의 기본은 'element-wise', 원소 별로 비교한다 -> true/false의 boolean 리스트 생성
    # print('y =', step_function(x))
    # true와 false의 리스트를 1과 0의 리스트로 바꿔주면 된다

    #sigmoid function
    # print('sigmoid', sigmoid(x)) # error: ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
            # array cannot be put to x of sigmoid(x) / only for a scalar, not an array
    # for x_i in x:
    #     print(sigmoid(x_i), end=' ')
    # print()
    print('sigmoid = ', sigmoid(x)) # the same result as for x_i in x, but returned in list format

    # step 함수, sigmoid 함수를 하나의 그래프에 출력
    x = np.arange(-5, 5, 0.05)
    steps = step_function(x)
    sigmoids = sigmoid(x)
    plt.plot(x, steps, label = 'Step function')
    plt.plot(x, sigmoids, label = 'Sigmoid function')
    plt.legend()
    plt.show()

    x = np.arange(-3, 4)
    print('x_for_relu =', x)
    relus = relu(x)
    print('relu =', relus)
    # relu가 더 크다 (0보다 적은 값들은 0으로, 0보다 큰 값들은 자기 자신과 같은 값으로 return해 준다)
    plt.plot(x, relus)
    plt.title('ReLU')
    plt.show()
    


































