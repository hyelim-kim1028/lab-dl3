"""
mini-batch에서 사용되는 soft max 함수 만들기
"""
import pickle

import numpy as np

from dataset.mnist import load_mnist


def softmax(X):
    """
    두가지 가정:
    1) X - 1차원 [x_1, x_2, ... , x_n]
    2) X - 2차원 [[x_11, x_12, ..., x_1n],
                  [x_21, x_22, ..., x_2n]]
    """
    dimension = X.ndim
    if dimension == 1:
        m = np.max(X) #1차원 배열 전체의 max/최댓값를 계산
        X = X - m # 0보다 같거나 작은 숫자로 변환 (이유: exponential의 overflow를 방지)
            # broadcast (m -> scalar value, X is an array)
        y = np.exp(X)/np.sum(np.exp(X)) # np.exp(X) 1개의 값 / 전체의 합
        # 배열의 크기 그대로 나온다: 모든 원소당 값이 나온다 => 리스트
        # 여기서도 broadcast가 일어남
    elif dimension == 2:
        # 이미지별로 최댓값을 찾아줘야한다 (row 뱡향으로 들어가 있음, axis = 1)
        # m = np.max(X, axis =1).reshape(len(X), 1) # len(X): 2차원 리스트 X의 row의 개수
        # X = X - m
        # sum = np.sum(np.exp(X), axis =1).reshape((len(X),1))
        # y = np.exp(X)/ sum # sum 은 모든 원소의 sum 인데 우리가 필요한것 -> sum of each row => axis = 1 => (k,1) (k,)을 계산하는 꼴 => sum()이라는 변수를 새로 지정해서 계산
        # 에러: 2차원에서 행개수는 k개, 열의 개수는 n개이다.

        # (k,n) (k,) 의 행렬은 계산이 안된다
        # X = X - m이 가능해지려면 reshape 을 해줘야
        Xt = X.T  # X의 전치 행렬(transpose)
        m = np.max(Xt, axis=0)
        Xt = Xt - m
        y = np.exp(Xt) / np.sum(np.exp(Xt), axis=0)
        y = y.T
    return y


if __name__ == '__main__':
    np.random.seed(2020)
    # 1차원 softmax 테스트
    a = np.random.randint(10, size = 5)
    print(a)
    # softmax = 0~1사이의 숫자로 바꿔주고, 모두 더하면 1이 되고, 크기의 비율을? 적용해주고, 순서를 지켜준다
    print(softmax(a)) # 합해주면 거의 1이 됨

    # 2차원 softmax
    A = np.random.randint(10, size = (2,3))
                    # 10 = 0 부터 10보다 작은 양의 정수들 중에서 (2,3)의 모양을 갖게 랜덤한 숫자들을 생성
    print(A)
    print(softmax(A)) # 행끼리 더해서 1이 되어야한다






















