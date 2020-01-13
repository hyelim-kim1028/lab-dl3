"""
padding: 결과 행렬의 크기를 변화시키기 위해서 사용
"""

import numpy as np

if __name__ == '__main__':
    np.random.seed(113)

    # 1차원 ndarray
    x = np.arange(1, 6) # [1 2 3 4 5]
    print(x)

    # convolution은 항상 원본 데이터보다 크기가 작아진다.
    # 그래서 convolution을 계속하다보면 convolution을 사용하지 못하는 단계에 도달한다
    # convolution이라고 하는 연산에 참여하는 횟수가 원소들별로 다르다는 문제점이 있다
    # 어떻게하면 참여횟수를 모두 똑같이 만들어 줄 수 있을까? => 패딩의 개념
    # for the further explanations, refer to the notes

    # numpy의 pad()
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.pad.html
    x_pad = np.pad(x,               # 패딩 넣을 배열
                   pad_width= 1,    # 패딩 크기
                   mode = 'constant',   # 패딩에 넣을 숫자 타입 (상수, 최댓/소값, 평균 등)
                   constant_values= 0)  # 숫자 타입이 상수일 경우, 상수로 지정할 값
    print(x_pad)
    # padding이 앞뒤로 들어가있다. # 축의 양쪽 끝에 똑같은 크기로 넣어주는 것 (pad_width)
    # mode = 어떤 숫자를 넣을것인가 => 가지고 있는 원소들 중에 최댓값/평균등을 넣을 수도 있다. 무조건 똑같은 숫자로 넣을 수도 있다 ( constant)
    # mode = 'constant' 라면, 어떤 상수를 넣어줄 것인가를 명시해야한다 -> constant_values = 0

    x_pad = np.pad(x, pad_width= (2,3),  # pad_width = (before padding, after padding)으로 주면 앞뒤로 다른 크기(len)를 줄 수 있다
                   mode = 'constant',
                   constant_values= 0)
    print(x_pad)

    # mode = 'minimum' test
    x_pad = np.pad(x, pad_width= 2,  # pad_width = (before padding, after padding)으로 주면 앞뒤로 다른 크기(len)를 줄 수 있다
                   mode='minimum')
    print(x_pad)
    # 앞뒤로 2개씩 패딩을 줬는데, 데이터 중 가장 적은 값 1 이 패딩으로 앞뒤로 들어가 있다
    # returns ndarray

    # 2차원 ndarray
    x = np.arange(1, 10).reshape((3,3))
    x_pad = np.pad(x, pad_width= 1, mode = 'constant', constant_values= 0)
    print(x_pad)

    # axis = 0 방향 before-padding = 1
    # axis = 0 방향 after-padding = 2 # 이 것들을 0번 방향과 1번 방향에 대해 반복
    x_pad = np.pad(x, pad_width = (1,2),
                   mode = 'constant', constant_values= 0)
    print(x_pad)
    # axis = 0 과 axis = 1 모두 각각 (1,2) 패딩을 해준다 axis 0 => 위로 1, 아래로 2; axis 1 => 왼쪽으로 1, 오른쪽으로 2

    x_pad = np.pad(x, pad_width=((1, 2), (3,4)), # padding for axis = 0, padding for axis = 1
                   mode='constant', constant_values=0)
    print(x_pad)



































