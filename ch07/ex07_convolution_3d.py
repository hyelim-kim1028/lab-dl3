"""
convolution_3d
# (3,4,4) & (3,3,3) 의 배열들을 가정을 하고 코드를 구현한다
"""

import numpy as np
from scipy.signal import correlate

def convolution3d(x,y):
    """
    x.shape = (c, h, w), y.shape = (c, fh, fw)
    h >= fh, w >= fw 라고 가정
    """
    h, w = x.shape[1], x.shape[2] #source의 height/width
    fh, fw = y.shape[1], y.shape[2] # 필터의 height/width
    oh = h - fh + 1
    ow = w - fw + 1
    # result = []
    result = np.zeros((oh, ow))
    for i in range(oh):
        for j in range(ow):
            x_sub = x[:,i: (i + fh), j:(j + fw)]
                # color에 : 까지 준 것: color은 RGB = 3개, 모두 선택되야되는것이 맞는 것
                # fma = np.sum(x_sub * y)
                # result.append(fma)
            result[i,j] = np.sum(x_sub * y)
    # return np.array(result).reshape((oh,ow))
    return result
    # 내 코드는 각 레이어의 파이널값 1개씩을 준 그런 리턴값을 주었다

if __name__ == '__main__':
    np.random.seed(114)

    # (3, 4, 4) 짜리 shape의 3차원 ndarray를 만든다
    x = np.random.randint(10, size = (3,4,4))
    print('x =', '\n', x)
    # convolution을 해줄 w를 만든다 w.shape = (3, 3, 3)
    w = np.random.randint(5, size = (3,3,3))
    print('w =', '\n', w)

    conv1 = correlate(x, w, mode = 'valid') # default = 'full' # 모든 원소와 곱해주는 # valid는 안에서만 움직이는 것
    print('conv1 =','\n',conv1)

    # 윗쪽과 동일한 결과를 작성
    conv2 = convolution3d(x,w)
    print('conv2 =','\n',conv2)

    x = np.random.randint(10, size = (3, 28, 28))
    print('x =', x)
    w = np.random.rand(3, 16, 16)
    print('w =', w)
    conv1 = correlate(x, w, mode = 'valid')
    conv2 = convolution3d(x, w)
    print('conv1 =', conv1.shape,'\n', conv1) #(1, 13, 13)
    print('conv2 =', conv2.shape, '\n', conv2) #(13, 13)

    # 필터를 여러개 둔다(여러개의 w)
    # 전제 조건은 c의 값은 모두 동일하다 (c = 3)

    # 미니배치 => 100개의 이미지를 한꺼번에 보내서 검사해보겠다 # 4차원 데이터
    # bias는 어떤 식이 계산이 됬으면, 모든 행에 똑같이 더해줄 값
    # ~행렬의 덧셈

    # convolution class가 구현되어있다
    # 고차원을 n-1Dimension으로 펼친다.






