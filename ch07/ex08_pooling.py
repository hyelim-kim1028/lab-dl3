import numpy as np


# pooling layers for preventing overflow-ish phenomena
# A problem with the output feature maps is that they are sensitive to the location of the features in the input.
# One approach to address this sensitivity is to down sample (= pooling) the feature maps.
# the other reason for doing so is to make the calculation faster/ working speed
from PIL import Image

from dataset.mnist import load_mnist
import matplotlib.pyplot as plt

def pooling1d(x, pool_size, stride = 1):
    """

    :param x: 원본 데이터  (x = 1차원 데이터)
    :param pool_size:
    :return:
    """
    n = x.shape[0] #len(x)
    result_size = (n - pool_size) // stride + 1
                # x의 갯수 - w의 사이즈 + 1 => 보폭이 1칸씩 움직일 때
                # 계속 봐온 formula to get the final shape of ... // 전체 보폭으로 나누어주었다 + 1
                # 몫만 사용한다 (늘 나누어 떨어지지 않음) => 이 공식은 교재의 234 [식7.1]
    result = np.zeros(result_size)
    for i in range(result_size):
        # 최댓값 혹은 평균을 찾겠다
        x_sub = x[i * stride: i * stride + pool_size]
        result[i] = np.max(x_sub)
    return result



def pooling2d(x, pool_h, pool_w, stride = 1):
    """

    :param x: 2-dim ndarray
    :param pool_h: pooling window height
    :param pool_w: pooling window width
    :param stride: 보폭
    :return: max-pooling
    """
    h,w = x.shape[0], x.shape[1]
    ow = (w - pool_w) // stride + 1
    oh = (h - pool_h) // stride + 1
    output = np.zeros((oh, ow))
    for i in range(oh):
        for j in range(ow):
        # 최댓값 혹은 평균을 찾겠다
            x_sub = x[(i * stride): ((i * stride) + pool_h),
                      (j * stride): ((j * stride) + pool_w)]
            output[i, j] = np.max(x_sub)
    return output


if __name__ == '__main__':
    np.random.seed(114)
    x = np.random.randint(10, size = 10) # 원소의 갯수가 10개짜리인 배열
    print(x)
    # 두개씩 묶어서 이동시켜가면서 최댓값을 찾는다 => max pulling
    # 큰 값을 대표값으로 찾겠다 = max pulling
    # 부분 집합의 대표값을 찾아서 대표값들로 이루어진 새로운 배열을 만든다
    # 원본 데이터가 너무 커서 이것을 줄여도 데이터의 특징을 유지하면 좋겠다 라는 생각에서 시작
    # 특징을 유지하는 방법 중 하나: 사이즈가 큰것을 찾는 것

    pooled = pooling1d(x, pool_size = 2, stride = 2)
    print(pooled)

    pooled = pooling1d(x, pool_size = 4, stride = 2)
    print(pooled)

    pooled = pooling1d(x, pool_size= 3, stride= 3)
    print(pooled)

    # pooling2d() test
    x = np.random.randint(99, size = (8, 8))
    print(x)
    pooled = pooling2d(x,2,2,2) #2x2 모양의 필터로 확인, stride = 2
    print(pooled)

    x = np.random.randint(100, size = (5,5))
    print(x)
    pooled = pooling2d(x, 3,3,3)
    print(pooled)

    x = np.random.randint(100, size=(5, 5))
    print(x)
    pooled = pooling2d(x, 3, 2, 2)
    print(pooled)

    # pooling은 왜 하는 걸까?
    # convolution 연산은 same을 많이 사용한다 -> 그러면 계속 같은 숫자를 사용
    # 과적합(overfitting) => element를 대표값을 갖도록 하되, 숫자룰 줄여버림 => 계산의 스피드가 빨라짐

    # pooling은 압축알고리즘에서 나왔다: 이미지를 축소해서 적응 용량으로 저장할 수 있게 함 i.e. bmp -> jpg 는 용량이 적지만 bmp와 같은 해상도로 표현한다 (= 압축되었다)
    # convultion은 필터를 입히는 것과 같은

    # Exercise 1
    # MNIST 데이터 세트 로드
    (X_train, Y_train), (X_test, Y_test) = load_mnist(normalize=False, flatten=False)

    # 그 중 이미지 아무거나 한장 선택 # shape = (1, 28, 28) -> (28, 28)로 변환
    X_img =  X_train[0]
    img_2d = X_img.reshape((28,28))
    print(img_2d.shape)
    # print(img_2d) # 숫자가 클수록 밝다
    #other way of reshaping
    # img_2d = X_img[0, :, :]
    # print('img_2d.shape:', img_2d.shape)

    # 선택된 이미지를 pyplot을 사용해서 출력
    plt.imshow(img_2d, cmap = 'gray')
    plt.show()

    # window size/shape (4, 4), stride = 4 -> output_shape = (7,7)
    window = pooling2d(img_2d, 4,4,4)
    print(window.shape) #(7, 7)
    plt.imshow(window, cmap = 'gray')
    plt.show()

    # Exercise 2
    # 이미지 파일을 오픈: (height, width, color)
    img = Image.open('sample.png')
    img_pixel = np.array(img)
    print(img_pixel.shape)  # (1320, 880, 3)

    # Red, Green, Blue에 해당하는 2차원 배열들을 추출
    img_r = img_pixel[:, :, 0] # Red panel
    img_g = img_pixel[:, :, 1] # Green panel
    img_b = img_pixel[:, :, 2] # Blue panel
    # print(img_r, img_g, img_b)

    # 각각의 2차원 배열을 window shape = (16,16), stride = 16으로 pooling
    window_r = pooling2d(img_r, 16, 16, 16)
    window_g = pooling2d(img_g, 16, 16, 16)
    window_b = pooling2d(img_b, 16, 16, 16)
    # pooling된 결과 (shape)를 확인, pyplot
    print(window_r.shape) #(82, 55)
    # plt.imshow(window_r)
    # plt.imshow(window_g)
    # plt.imshow(window_b)
    # plt.show()
    img_pooled = np.array([window_r, window_g, window_g]).astype(np.uint8)
    print(img_pooled.shape) # (3, 82, 55) -> 이래서 바로 그래프를 그려주지 못한다.
    img_pooled = np.moveaxis(img_pooled, 0, 2) # 0번하고 2번을 바꾼다
    print(img_pooled.shape) #(82, 55, 3)
    plt.imshow(img_pooled)
    plt.show() # Error: Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
               # int 타입으로 바꿔줘야한다 # .astype(np.uint8) 이걸 붙여줌

    # 4차원 혹은 n차원의 배열을 풀어서 2차원으로 만들 수 있다
