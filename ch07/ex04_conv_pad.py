"""
# padding을 이용한 convolution
"""

import numpy as np
from scipy.signal import convolve, correlate, convolve2d, correlate2d

from ch07.ex01_convolution1d import convolution_1d

if __name__ == '__main__':
    x = np.arange(1, 6)
    w = np.array([2, 0])
    print('x =', x)
    print('w =', w)
    conv1 = convolution_1d(x, w) # flip까지 된 제대로된 convolution
    print(conv1)
    # [ 4  6  8 10]
    # 일반적인 convolution(x,w)의 결과의 shape는 (4,)
    # convolution 연산에서 x의 원소 중 1과 5는 연산의 1번만 기여한다. # 2,3,4는 각 2번씩 기여한다.

    # x의 모든 원소가 convolution 연산에서 동일한 기여를 할 수 있도록 padding을 넣고 싶다
    x_pad = np.pad(x, pad_width= 1, mode = 'constant', constant_values= 0)
    # [0 4  6  8 10 0] # 변화된 x를 패딩에 넣었음
    # padding의 크기는 w의 크기보다 1개 적게끔 주면 된다
    print(convolution_1d(x_pad,w)) #[ 2  4  6  8 10  0]
    # 일반적으로 크기가 더 커진다

    # convolution 결과의 크기가 입력 데이터 x와 동일한 크기가 되도록 padding
    # 동일한 기여를 하도록 패딩을 넣는다 -> 둘중(?)에 하나를 잘라버려야한다
    # 현 상황: x는 홀수, w는 짝수 이므로, 한쪽에만 넣어준다  # before에만 넣어준다
    x_pad = np.pad(x, pad_width=(1,0), mode = 'constant', constant_values= 0)
    print(convolution_1d(x_pad,w))
    # 1,2,3,4는 동일한 기여를 하지만, 5는 동일한 기여를 하지 못한다

    # padding을 뒷쪽에 넣게 된다면
    x_pad = np.pad(x, pad_width=(0,1), mode='constant', constant_values=0)
    print(convolution_1d(x_pad, w)) #

    # convolution의 결과는 크기가 작아지고, 각 원소의 기여도가 다르기 때문에
    # padding을 사용해서 1) 기여를 동일하게 할 수 있도록 도와주던가 2) 동일한 크기가 될 수 있도록 도와준다

    # numpy.convolve 가 있고, sci_py에 convolve가 있다
    # scipy.signal.convolve() 함수
    conv = convolve(x, w, mode = 'valid')
    # default for mode = 'full'
    print('conv =',conv) # 일반적으로 사용했었던 convolution이 반환해줬던 값과 같은 값을 주었다

    conv_full = convolve(x, w, mode = 'full') # x의 모든 원소가 동일하게 연산에 기여
    print('conv_full',conv_full) # 양옆에 패딩을 1개씩 줘서 모든 원소들이 동일하게 기여할 수 있게 해주는 값으로 반환

    conv_same = convolve(x, w, mode = 'same') # x의 크기와 동일한 리턴
    print('conv_same',conv_same) # 앞쪽에 패딩을 넣은 값과 같은 값을 반환해 주었다

    #scipy.signal.correlate() 함수
    cross_corr = correlate(x, w, mode = 'valid')
    print(cross_corr)
    print(correlate(x, w, mode = 'full')) # full: 모든 원소들이 동일한 기여를 하게함
    print(correlate(x, w, mode = 'same'))

    # 2D correlate function()
    # scipy.signal.convolve2d(), scipy.signal.correlate2d()

    # 4x4 2d ndarray
    x = np.array([[1,2,3,0],
                 [0,1,2,3],
                 [3,0,1,2],
                 [2,3,0,1]])

    # 3x3 2d ndarray
    w = np.array([[2, 0, 1],
                  [0, 1, 2],
                  [1, 0, 2]])

    # x와 w의 교차 상관 연산 (valid, full, same)
    print('valid_cross_corr =', '\n',correlate2d(x, w, mode = 'valid')) # 4x4 / 3x3 => 4-3+1 x 4-3+1
    print('full_cross_corr =', '\n',correlate2d(x, w, mode = 'full')) #
    print('same_cross_corr =', '\n',correlate2d(x, w, mode = 'same'))

    # stride = 보폭
    # stride 가 커지면 줄어드는 효과가 더 눈에 띄게 나타난다

    # 왜 3차원을 해야할까?
    # batch 처리까지하면 4 차원이고, 이미지 한개만 넘긴다고 할 때 3차원


















