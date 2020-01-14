"""
image_index
"""

from PIL import Image
import numpy as np
from dataset.mnist import load_mnist
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # 이미지 파일 오픈
    img = Image.open('sample.png')
    # 이미지 객체를 numpy 배열 형태(3차원 배열)로 변환
    img_pixel = np.array(img)
    print('img_pixel:', img_pixel.shape) # (height, width, color-depth)
    # print(img_pixel) # [[R, G, B], [R, G, B], [R, G, B], ...] # RGB가 width만큼 있고, 이런 레이어들이 height만큼 있다

    (x_train, y_train), (x_test, y_test) = load_mnist(normalize= False, flatten= False)
    # 훈련 데이터에서 1장만 꺼내기
    print('x_train', x_train.shape) #(60000, 1, 28, 28) #(n_sample, color, height, width)
    print('x_train[0] =', x_train[0].shape) #이미지 1장의 모양 #(1, 28, 28) #(color, width, height)
    # matplotlib으로 그릴 수 없다, 인덱스를 뒷쪽으로 옮겨줘야, 인덱스의 모양을 바꿔줘야 그릴 수 있다
    # plt.imshow(x_train[0]) #TypeError: Invalid shape (1, 28, 28) for image data
    # 축을 바꿔줘야 matplotlib이 그릴 수 있다

    # (c, h, w) 형식의 이미지 데이터는 matplotlib을 사용할 수 없음. (h, w, c)로 해야함
    num_img = np.moveaxis(x_train[0], 0, 2) # c라는 축을 index 2로 가져가겠다
    print(num_img.shape) #(28, 28, 1)
    num_img = num_img.reshape((28, 28)) # 단색인 경우 2차원으로 변환 # 2차원으로 바꾸면 된다
    plt.imshow(num_img, cmap='gray')
    plt.show()

    # (3,4,4) & (3,3,3) 의 배열들을 가정을 하고 코드를 구현한다



