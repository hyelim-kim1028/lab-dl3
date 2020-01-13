"""
2D convolutional neural network/ 2차원 convolution (합성곱) 연산
"""

import numpy as np

def convolution_2d(x, w):
    """ x와 w는 2차원 2d ndarray 이다. x.shape이 w.shape보다 크다 라고 가정한다.
        x와 w의 교차상관 연산 (cross-correlation) 결과를 리턴 """
    # x_sub1과 x_sub2를 보면 패턴이 나온다 -> for구문으로 바꿀 수 있다
    # how can we give the range? (wh * ww)? is it valid all throughout?
    # conv = []
    # x_sub = []
    # fma = []
    # xh, xw = x.shape[0], x.shape[1]
    # wh, ww = w.shape[0], w.shape[1]
    # for i in range(w):
    #     for j in range(x):
    #         x_sub[j] = x[j : j + wh, i: ww + i]
    #         x_sub.append(x_sub[j])
    #         fma[j] = np.sum(x_sub[j] * w)
    #         fma.append(fma[j])
    # conv.append(x_sub)
    # conv = np.array(x_sub)
    # return conv

    # teacher's solution
    # convolution 결과 행렬 (2d ndarray)의 모양(shape)을 예상: (rows, cols)
    rows = x.shape[0] - w.shape[0] + 1
    cols = x.shape[1] - w.shape[1] + 1
    conv = [] #결과를 저장할 리스트
    for i in range(rows): # 오른쪽으로의 진행이 끝나면 i에서 위에서 아래로의 진행
        for j in range(cols): #오른쪽으로 진행/ 컬럼의 숫자를 높여주며 진행
            # 제일 처음: subset 찾아내기
            x_sub = x[i: (i + w.shape[0]), j:(j + w.shape[1])]
            fma = np.sum(x_sub * w)
            conv.append(fma) # 한번 진행할 때마다 fma을 구해주고 붙여주기
    conv = np.array(conv)
    return conv.reshape(rows, cols)


if __name__ == '__main__':
    np.random.seed(113)

    x = np.arange(1, 10).reshape((3,3)) # 1~9사이의 3x3행렬
    print(x)

    w = np.array([[2,0], [0,0]])
    print(w)

    # 2d 배열 x의 가로(width), 세로(height)
    xh, xw = x.shape[0], x.shape[1]
    #2d 배열 w의 가로 ww와 세로 wh
    wh, ww = w.shape[0], w.shape[1]

    # x_sub1
    x_sub1 = x[0:wh, 0:ww] #x_sub1 = x[0:2, 0:2] -> 0이상 2미만
                           # 2라고 한 이유는 w의 크기와 맞추기 위해서 이므로, wh와 ww로 대체될 수 있다
    print(x_sub1)
    fma1 = np.sum(x_sub1 * w) # numpy 에서 * 는 원소별로 곱셈해준다
    print(fma1)

    # x_sub2
    x_sub2 = x[0:wh, 1:1+ww] #x[0:2, 1:3]
    print(x_sub2)
    fma2 = np.sum(x_sub2 * w)
    print(fma2)

    # x_sub3
    x_sub3 = x[1: 1+wh, 0: ww] # x[1:3, 0:2]
    print(x_sub3)
    fma3 = np.sum(x_sub3 * w)
    print(fma3)

    # x_sub4
    x_sub4 = x[1:1+wh, 1:1+ww] #x[1:3, 1:3]
    fma4 = np.sum(x_sub4 * w)
    print(fma4)

    conv = np.array([fma1, fma2, fma3, fma4]).reshape((2,2))
    print('conv =', conv)

    # convultion_2d() function test
    convd = convolution_2d(x,w)
    print(conv)

    x = np.random.randint(10, size = (5, 5)) # 0 부터 9까지의 정수, (행 = 5, 열 = 5)
    w = np.random.randint(5, size = (3,3))
    print('x =','\n', x)
    print('w =','\n', w)
    # final result would be 3x3 ~> (5-3+1) x (5-3+1)
    print('conv_2d =', '\n', convolution_2d(x, w))
    # 이미지 처리... 필터링에 활용된다

    # 교차 상관 <-> convolution
    # w의 교차의 차이
    




