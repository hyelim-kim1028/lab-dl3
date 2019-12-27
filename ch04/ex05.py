"""
gradient descent
"""
import numpy as np


def numerical_diff(fn, x):
    """ Numerical Differential
    함수fn와 점x가 주어졌을 때, x에서의 함수fn의 미분(도함수)값 """
    h = 1e-4 # 0.0001 #너무 높게 잡으면 explode, 작게 잡으면 underflow, never reach the point
    return (fn(x + h) - fn(x-h))/(2*h)

def f1(x):
    """ made a function on the white board """
    return 0.001 * x**2 + 0.01 * x

def f1_prime(x):
    # 근사값을 사용하지 않은 함수 f1의 도함수를 직접 구한 것
    return 0.002*x + 0.01

def f2(x):
    """ 두 변수로 이루어진 함수
        x = [x0,x1] # x는 원소가 2개인 차원 리스트이다
         - 이렇게 하면 꼭 원소가 2개일 필요가 없다. """
    return np.sum(x**2)
    # 원소 n 개가 와도 broadcasting 으로 각 원소들은 제곱이 되고, 모든 원소들은 sum에 의해 더해진다
    # x0**2 + x1**2 + ...

# 변수가 2개라면 gradient도 각각의 변수들에서 나오게 된다
# # f(x0, x1) = x0**2 + x1**2
# # 편미분을 한다 (다른 변수를 상수라고 취급하고, 한가지 변수에 대해서만 미분을 한다)
# # af/ ax0 = 2x0 + 0 (x1는 상수로 취급하기 때문에 0이 된다)
# # x0 =3, x1 = 4 -> af/ax0 = 6
# # af/ ax1 = 2x1 + 0 (x0는 상수로 취급하기 때문에 1이 된다)
# # x0 =3, x1 = 4 -> af/ax1 = 8

# 편미분을 어떻게 하지 -> 상수취급,,,?

def f3(x):
    return x[0] + x[1]**2 + x[2]**3


def numerical_gradient(fn, x):
    """ fn: 독립 변수를 여러개 갖는 함수
        fn = fn(x0, x1, ... xn)
        x: n-dimensional array
        i.e. x = [x0, x1, ... xn]
        위와 같이 가정을 했을 때,
        점 x = [x0, x1, ... xn] 에서의 함수 fn = fn(x0, x1, ... xn)의
        각 편미분 (partial differential)들의 배열을 리턴
        """
    x = x.astype(np.float) #실수타입
    gradient = np.zeros_like(x) #np.zeros(shape = x.shape) 와 같은 것
            # 원소 2개짜리 array를 만들어서
    # 독립 변수 갯수 만큼 0으로 채워둔 배열
    h = 1e-4 # 아주 작은 값 생성 #0.0001
    # numercal diff에서는 모든 변수들에 관해서 +-를 한것 # 그럼 x가 상수 취급이 안됨 -> not 편미분
    # 그리고 일일히 분해를 해야한다
    for i in range(x.size):
        ith_value = x[i]
        x[i] = ith_value + h
        fh1 = fn(x)
        x[i] = ith_value - h
        fh2 = fn(x)
        gradient[i] = (fh1 - fh2) /2*h
        x[i] = ith_value # i번쨰 있던 원소를 다시 되돌려주는 작업이 필요하다 -> 다른 값이 상수로 취급되는 편미분의 특성 때문에
    return gradient




if __name__ == '__main__':
    estimate = numerical_diff(f1, 3)
    print('근사값:', estimate) #0.016000000000043757
    real = f1_prime(3)
    print('실제값:', real) # 0.016

    #f2 함수의 점 (3,4)에서의 편미분 값
    estimate_1 = numerical_diff(lambda x: x**2 + 4**2, 3) #접미분
                            # x0 = 3, x1 = 4일 때 미분값을 구해라
                            # x**2 은 변한다, ... 뭐야 ㅠㅠㅠ?
    print('estimate_1', estimate_1) # 6.000000

    estimate_2 = numerical_diff(lambda x: 3 **2 + x**2, 4)
    print('estimate_2', estimate_2) # 7.999999999999

    # f(x0, x1) = x0**2 + x1**2
    # af/afx0 에서 x0=3, x1 = 4 일때,  d/dx0(x0**2 + 4**2) 에서 x0 = 3
                                    # x0로 미분하자?
    # 편미분, X는 미분하고, 다른 X값은 상수로 생각한다
    # 독립변수의 갯수만큼 편미분이 나온다
    # 도함수들의 배열을 만들기 위해서 이 값을 계산해 주었다

    # 편미분의 정의:
    # af/ax0 = lim(h->0) (f(x0 + h_x1) - f(x0 - h_x1)) /((x0+h) - (x0-h))
    # af/ax1 = lim(h->0) (f(x0, x1 + h) - f(x0, x1 - h)) /2h

    # gradient
    gradient = numerical_gradient(f2, np.array([3,4]))
    print(gradient) #[6.e-08 8.e-08]

    # f3 = x0 + x1**2 + x2**3
    # 점 (1, 1, 1)에서의 각 편미분들의 값
    # df/dx0 = 1, df/dx1 = 2, df/dx2 = 3이 나와야한다
    gradient = numerical_gradient(f3, np.array([1,1,1]))
    print(gradient)

    #f4 = x0**2 + x0 * x1 + x1**2
    # 점 (1,2) dptjdml df/dx0 = 4, df/dx1 = 5
    def f4(x):
        return x[0]**2 + x[0] * x[1] + x[1]**2

    gradient = numerical_gradient(f4, np.array([1,2]))
    print(gradient)