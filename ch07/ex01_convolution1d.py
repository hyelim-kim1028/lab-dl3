"""
1차원 Convolution, Cross-Correlation 연산
# 합성곱과 교차상관은 서로 거울을 보듯 비추는? 관계이다
"""

import numpy as np

def convolution_1d(x, w):
    """ x, w: 1d ndarray, len(x) >= len(w)
        x와 w의 합성곱 연산 결과를 리턴 """
    # if len(x) >= len(w):
    #     return len(x) - len(w) + 1
    # How teacher solved:
    # w_r = np.flip(w) #w를 반전 시킨다
    # nx = len(x) # x의 원소의 개수
    # nw = len(w) # w의 원소의 개수
    # n = nx - nw + 1 # convolution 연산 결과의 원소 개수
    # conv = []
    # for i in range(n):
    #     x_sub = x[i: i + nw]
    #     fma = np.sum(x_sub * w_r) # fused multiply-add
    #     conv.append(fma)
    # return np.array(conv)
    # How we can shorten the code using cross_correltation_1d() method
    w_r = np.flip(w)
    conv = cross_correlation_1d(x, w_r)
    return conv

def cross_correlation_1d(x, w):
    """ # x와 w는 모두 1d array 이고, x의 크기는 w의 크기보다 크다 라고 가정하다
        x와 w의 교차 상관(cross-correlation) 연산 결과를 린턴
        -> convolution_1d() 함수가 cross-correlation_1d를 사용하도록 수정하기
    """
    nx = len(x)
    nw = len(w)
    n = nx - nw + 1
    cross_corr = []
    for i in range(n):
        x_sub = x[i: i+nw]
        fma = np.sum(x_sub * w)
        cross_corr.append(fma)
    return np.array(cross_corr)


# x크기가 ,,, convoluion의 크기가 커진다구?!
# 원본의 크기가 결과값으로 나올 수 있도록



if __name__ == '__main__':
    np.random.seed(113)
    x = np.arange(1, 6)
    print('x =', x) # x = [1 2 3 4 5] # array
    w = np.array([2,1])
    print('w =', w) # w = [2 1]

    # w Conv w (in numpy, dot production is represented with @)
    # x.dot(W) is the same as x@w
    # dot에서는 (n,m) @ (m,p) 로 같아야한다
    # x * w 에서는 두 원소의 shape이 모두 같거나 broadcasting을 할 수 있어야한다

    # 일차원에서의 conv연산은 x*w
    # W를 반전시킨다 (거울 보는 것 처럼)
    # 원소별로 곱하기 & 더하기를 한다 (FMA)
    # 나머지 설명: 노트에 있다
    # 1칸씩 움직이며 fma 을 해줄 수 있지만, 보폭(stride)를 조절할 수 있다
    # stride의 기본값은 1 이다

    # cross correlation: 반전 시키지 않음, convolution: 반전 시킨 후 해나감
    # 왜 반전 시키지 않고 그냥 할까? -> 처음에 난수로 만들 꺼라서


    # x Conv w # Convolution 합성곱 연산
    # 1. w를 반전 시킨다
    # w_r = np.array([1,2]) # 1과 2 밖에 없는 array이라서 간단히 해줌
    # 혹은 numpy 의 flip 이라는 함수를 사용
    w_r = np.flip(w)
    print('w_r =', w_r)

    #2) FMA (Fused Multiply-Add)
    conv = []
    for i in range(4):
        x_sub = x[i:i+2] # w의 원소가 2개니까 x에서 2개의 원소를 꺼내서 enforced to do the broadcasting
        fma = np.sum(x_sub * w_r) # FMA -> multiplication -> addition 을 해주는 operation을 일컫는다
        conv.append(fma) # conv에다 fma값을 append 해준다
    conv = np.array(conv)
    print('conv =', conv) #1차원 convolution 연산
    #1차원 convolution 연산의 크기(원소의 갯수) = len(x) - len(w) + 1
    # padding을 사용해서 변경 가능

    # convolution_1d 함수 테스트
    conv = convolution_1d(x, w)
    print('convolution_1d =',conv)

    x = np.arange(1, 6)
    w = np.array([2, 0, 1])
    conv = convolution_1d(x,w)
    print(conv) #[7 10 13]

    # 교차상관 cross-correlation 연산
    # 합성곱 (convolution) 연산과 다른 점은 w를 반전 시키지 않는다

    # 지금은 w에 큰 비중을 안는 이유: 랜덤 넘버로 뽑아 왔기 때문에 -> 굳이 반전 시킬 필요가 있는가? -> 교차상관을 사용하던가 합성곱을 사용하던가 상관이 없다
    # 나중에는 반전 시키고, 그걸 가지고 ,,ㅇㄹ 마더ㅑ;
    # 신경망(CNN/ Convolutional Nueral Network, 합성곱 신경망)에서는 대부분의 경우 합성곱 연산 대신 교차 상관을 사용함
    # 어짜피 가중치 행렬은 난수로 생성한 후 gradient descent 등을 이용해 갱신하기 때문에 구별하지 않는다

    # cross_correlation_1d() method test
    cross_corr = cross_correlation_1d(x, w)
    print('cross_corr =', cross_corr)






















