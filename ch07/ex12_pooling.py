"""

"""
from common.util import im2col
import numpy as np
from dataset import mnist
from dataset.mnist import load_mnist
import matplotlib.pyplot as plt

class Pooling:
    def __init__(self, fh, fw, stride = 1, pad =0):
        # in the text book fh is written as pool_h and so on
        self.fh = fh # pooling 윈도우의 높이(height)
        self.fw = fw # pooling 윈도우의 너비(width)
        self.stride = stride # pooling 윈도우의 이동시키는 보폭
        self.pad = pad # 패딩 크기
        # backward에서 사용하게 될 값
        self.x = None # pooling 레이어로 forward 되는 데이터
        self.arg_max = None # 찾은 최댓값의 인덱스

    def forward(self, x):
        """ 순방향 전파 메소드
            x: (samples, channel, height, width) 모양의 4차원 배열
        """
        # 구현
        # x를 펼친다
        self.x = x

        n, c, h, w = self.x.shape # 샘플 개수, 채널, 높이, 너비
        oh = (h - self.fh + 2 * self.pad) // self.stride + 1
        ow = (w - self.fw + 2 * self.pad) // self.stride + 1

        # 1) x --> im2col --> 2차원 변환
        col = im2col(x, self.fh, self.fw, self.stride, self.pad)

        # 2) 채널 별 최댓값을 찾을 수 있는 모양으로 x를 reshape
        col = col.reshape(-1, self.fh * self.fw)

        # 3) 채널 별로 최댓값을 찾음.
        self.arg_max = np.argmax(col, axis = 1)
        out = np.max(col, axis = 1)
        # max: 최댓값을 찾아주는 함수 argmax: 최댓값의 위치를 저장해주는 함수

        # 4) 최댓값(1차원 배열)을 reshape & transpose
        out = out.reshape(n, oh, ow, c).transpose(0, 3, 1, 2)

        # 5) pooling이 끝난 4차원 배열을 리턴
        return out



if __name__ == '__main__':
    np.random.seed(116)

    # pooling 클래스의 forward 메소드를 테스트
    # x를 (1, 3, 4, 4) 모양으로 무작위로 (랜덤하게) 생성, 테스트
    x = np.random.randint(10, size = (1, 3, 4, 4))
    print(x)

    # pooling 클래스의 인스턴스 생성
    pooling_forward = Pooling(fh = 2, fw = 2, stride = 2, pad = 0)
    # pooling 에서는 수축이 목적이기 때문에 패딩은 많이 사용하지 않는다
    out = pooling_forward.forward(x)
    print(out)

    # MNIST 데이터를 로드
    (X_train, Y_train), (X_test, Y_test) = load_mnist(normalize=False, flatten=False)

    # 학습 데이터 중 5개를 batch로 forward
    x = X_train[:5]
    out = pooling_forward.forward(x)
    print('out_shape:', out.shape)

    # # 학습 데이터를 pyplot으로 그림
    # forwarding된 결과(pooling 결과)를 pyplot으로 그림.
    for i in range(5): #5개의 학습 데이터를 선택해서
        ax = plt.subplot(2, 5, (i + 1), xticks = [], yticks = [])
                        # nrow, ncol,
        plt.imshow(x[i].squeeze(), cmap = 'gray')
        ax2 = plt.subplot(2, 5, (i + 6), xticks = [], yticks = [])
        plt.imshow(out[i].squeeze(), cmap = 'gray')
    plt.show()
    # xticks = [], yticks = [] 를 주면 옆에 grid 가 생기고 안생기느냐의 차이

    # # forwarding 된 결과 (pooling 결과)를 pyplot으로 그림
