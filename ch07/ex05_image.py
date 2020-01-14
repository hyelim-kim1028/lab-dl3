"""
CNN(Convolutional Neural Network)
# 합성곱 신경망이 왜 생겼나요? 원래 합성곱은 image-processing에서 많이 사용하던 변환 연산
# 이미지, 음성, 동영상 처리와 같이 일종의 신호들을 처리하는 곳에서 어떤 신호들을 변환하기위해 사용하던 연산
# https://docs.scipy.org/doc/scipy/reference/signal.html #그래서 이게 signal processing 을 하는 모듈이라고 설명하는듯
# 신경망을 만드는 노드로써 사용하기 좋음.
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.signal import convolve, correlate
# signal: 신호 => 영상, 이미지, 음성 신호

# jpg 파일 오픈
img = Image.open('sample.png')
print(img)
img_pixel = np.array(img)
# JPG(24-bit) > RGB 가 각 8 비트씩 있는 것
print(img_pixel.shape) #(height, width, color-depth) # (854, 1280, 3) # 3- RGB
                       #(n_row, n_col)
                       # color-depth는 패키지마다 제일 처음에 나올수도 있고, 제일 마지막에 나올수도 있다
# (1320, 880, 3)
# 1320 X 880 짜리 2d-array?가 3개가 있다 # 그 3가지 layer를 조합하면 색깔로 나타난다


# 머신 러닝 라이브러리에 따라서 color 표기의 위치가 달라짐
# Tensorflow: channel-last 방식. color-depth가 n-D 배열의 마지막 차원
            # n차원 배열 > 3차원 배열에서의 마지막 > 그게 color-depth 가 된다
# Theano: channel-first 방식. color-depth가 n-D 배열의 첫번째 차원 (color-depth, height, width)
# Keras: 둘 다를 지원한다

plt.imshow(img_pixel)
plt.show() #이미지를 볼 수 있다

# 이미지의 RED값 정보
print(img_pixel[:, :, 0]) # 0번 채널: 제일 처음 -> Red?
red = img_pixel[:, :, 0]
# plt.imshow(red)
# plt.show() # 오오오 근데 이것이 진짜 red..?

#(3,3,3) 필터
filter = np.zeros((3,3,3)) # (h, w, c)에서 c만 원본 이미지를 따르고 나머지는 알아서 주면 된다
filter[1, 1, 0] = 1.0
transformed = convolve(img_pixel, filter, mode = 'same')/255 # 255로 나눈 이유: 0~1사이의 값으로 나누기 위해서
plt.imshow(transformed)
plt.show()
# 이미지 위에다 초록색을 덮어준 것 -> 이게 convolution


# 반전이 되어버리기 때문에 cross_correlation을 사용하면 색상이 많이 다르게 나온다
filter = np.zeros((3,3,3))
filter[1,1,0] = 1.0
transformed = correlate(img_pixel, filter, mode = 'same')/255
plt.imshow(transformed)
plt.show()
# 초록? 노랑빛이 많이 맴돌게 출력된다

# full padding: 보호구가 된다 (단점: 크기가 너무 커진다)

# pool & same: 차가 있어야 가능
# smae은 원본과 같은 크기가 출력되고, 영
# full은 커진다, deli는 작아진다,,,

# 왜 컨벌류션을 해야할까? image 블루로 다시 시작!
# 이미지가 갖는 특징들을 읽너낼 수 있게함

filter = np.zeros((8,8,3))
filter[0,0,:] = 255
transformed = correlate(img_pixel, filter, mode = 'same')
plt.imshow(transformed.astype(np.uint8))
plt.show()

# 이미지 한장은 (height, width, color) 이렇게 세가지를 갖고, color 에서 여러가지 레이어를 갖는 (RGB 각 1개의 레이어)
# height = nrow, width = ncol







