import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d #3D 그래프를 그리기 위해서 반드시 import
import numpy as np

def fn(x,y):
    """ f(x,y) = (1/20) * x^2 + y^2 """
    return x**2/20 + y**2
    # return x**2 + y**2


# x가 커지는 비율보다 y가 커지는 비율이 훨씬 높다
# x와 y에 대한 편미분이 2개가 존재한다 # x에 대한 미분 그리고 y에 대한 미분
# af/ax = 1/10x, af/ay = 2y

def fn_derivative(x, y):
    return x/10, 2 * y # x에 대한 기울기, y에 대한 기울기를 리턴해주는 함수
    # """ 편미분 df/dx, df/dy 튜플을 리턴 """
    # return 2 *x, 2 * y # x에 대한 기울기, y에 대한 기울기를 리턴해주는 함수

if __name__ == '__main__':
    # x좌표들 만들기
    x = np.linspace(-10, 10, 1000) #x좌표 # 잘게 나누면 나눌수록 그래프가 예쁘게 나옴
    y = np.linspace(-10, 10, 1000) #y좌표
    # 3차원 그래프를 그리기 위해서 x좌표와 y좌표의 쌍으로 이루어진 데이터
    X, Y = np.meshgrid(x,y)
    Z = fn(X, Y)

    fig = plt.figure()
    ax = plt.axes(projection = '3d')
    # projectiion 파라미터를 사용하려면 mpl_toolkits.mplot3d 패키지가 필요
    ax.contour3D(X,Y,Z,
                 100, # 숫자를 적게주면 그래프가 듬성듬성 나타나고, 숫자를 늘려줄 수록 좀 더 세밀하게 그려준다
                 cmap = 'binary') # cmap stands for color map, binary gives black and white graph
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show() # 3차원 그래프!! 오오오 신기방기
               # 그래프를 그릴 축들

    # 등고선 (contour) 그래프
    plt.contour(X, Y, Z, 50, cmap = 'binary')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal') # 가운데가 희미하게 보임 -> 가까울수록 진하다, 멀수록 연하다 -> 촘촘할 수록 경사가 급하다 (위아래로는 급하고, 양옆으로는 완만한 경사를 보인다)
    plt.show()
    # 3차원을 2차원으로 축소시켜놓고 기울기를 파악하면서 n차원의 그래프를 볼 때 감을 잡는 정도로 사용하겠다 ^0^!

