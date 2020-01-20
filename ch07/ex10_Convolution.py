"""
Convolution class 만들기
"""
from PIL import Image
from common.util import im2col
import numpy as np
from dataset.mnist import load_mnist
import matplotlib.pyplot as plt

class Convolution:
    def __init__(self, W, b, stride = 1, pad = 0):
    # W: 가중치 행렬(filter), b: bias
        self.W = W # weight - filter
        self.b = b # bias
        self.stride = stride
        self.pad = pad
        # 중간 데이터: forward에서 생성되는 데이터 -> backward에서 사용되기 때문에 저장해야한다
        self.x = None
        self.x_col = None
        self.W_col = None
        # gradients
        self.dW = None
        self.db = None
        # Affine 처럼 convolution을해서 보내는 것


    def forward(self, x):
        """ x : 4 차원 이미지 (mini-batch) 데이터 """
        self.x = x

        #How the book solved (and the author is so genius)
        fn, c, fh, fw = self.W.shape
        n, c, h, w = self.x.shape

        oh = (h - fh + (2 * self.pad)) // self.stride + 1
        ow = (w - fw + (2 * self.pad)) // self.stride + 1

        # 이미지와 필터를 풀어헤친다
        self.x_col = im2col(self.x, fh, fw, self.stride, self.pad)
        self.W_col = self.W.reshape(fn, -1).T
        # W(fn, c, fh, fw)과 같이 생겼다 --> W_col (fn, c * fh * fw) --> W_col(c * fh * fw, fn)

        out = self.x_col.dot(self.W_col) + self.b
        #  np.dot(self.x_col, self.W_col) = self.x_col.dot(self.W_col)
        # This happens often in numpy ex.np.mean(x) == x.mean()
        out = out.reshape(n, oh, ow, -1).transpose(0, 3, 1, 2)
        # 2차원을 4차원으로 바꿔주고, 축의 방향도 돌려주고 리턴

        return out


        # How I solved
        # self.x_col = im2col(self.x, filter_h = self.W[0], filter_w = self.W[1], stride = 1, pad = 0  )
        # self.w_col = self.W.reshape(len(self.W), -1)
        #
        # oh = self.x[0] - self.W[0] + (2 * self.pad) // self.stride + 1
        # ow = self.x[1] - self.W[1] + (2 * self.pad) // self.stride + 1
        #
        # # dot  할 때, when (n, m) and (i, j), if m = i, do it , else: w_col.T
        # if len(self.W_col[1]) == len(self.x_col[0]):
        #     return self.x_col.dot(self.W_col).reshape(len(x), oh, ow, -1).transpose(0, 3, 1, 2)
        # elif len(self.W_col[1]) == len(self.x_col[1]):
        #     self.w_col = self.w_col.T
        #     return self.x_col.dot(self.W_col).reshape(len(x), oh, ow, -1).transpose(0, 3, 1, 2)
            # can we say n as len(x) HAHAHAHAHHA fuckkk


    def backward(self,x):
        pass



if __name__ == '__main__':
    np.random.seed(115)
    # Convolution을 생성
    # MNIST 데이터를 forward
    W = np.zeros((1, 1, 4, 4), dtype = np.uint8) # filter: (fn, c, fh, fw) => 여기서 c=1인 이유는 minst데이터는 컬러가 1가지 밖에 없기 때문!
                                                 # dtype: 8bit 부호없는 정수 (음수는 없고 0이거나 0보다 크거나 같은 정수)
                                                # 그냥 난수로 만들어 줄 수도 있다
    W[0, 0, 1, :] = 1 # 4x4행렬이 있는데
    # W[fn, c, h, w] => 세로로 넓게 만들어주는 필터
    b = np.zeros(1)
    conv = Convolution(W, b) #Convolution객체의 생성자를 호출

    # MNIST 데이터를 forward
    (X_train, Y_train), (X_test, Y_test) = load_mnist(normalize=False, flatten=False)
    input = X_train[0:1] #(1, 1, 28, 28)
    print('input:', input.shape) #(1, 28, 28) #우리가 만든 클래스는 3차원은 처리하지 못함
    # 여기에 대한 에러처리가 되어있지 않음
    # 이미지는 1장만 보내고 싶은데 4차원이여야한다 -> slicing을 이용 X_train[0] #(1, 28, 28) -> X_train[0:1]
    out = conv.forward(input)
    print(out.shape)

    # 그래프 그리기
    img = out.squeeze()
    print('img:', img.shape) # 왜 안줄었어
    # 3차원 -> 2차원, 2차원 -> 1차원으로 만들어주는 함수!
    plt.imshow(img, cmap='gray')
    plt.show()

    # 다운로드 받은 이미지 파일을 ndarray로 변환해서 forward
    # (h,w,c) 으로 나오면 -> transpose를 먼저하고 시작
    img = Image.open('sample.png')
    img_pixel = np.array(img)
    # 처음 불러왔을 때: 3차원이다 -> 처리불가! -> 변형해줌
    img_pixel2 = img_pixel.reshape((1, 1320, 880, 3))
    print('img_pixel_original:',img_pixel2.shape) # (1, 1320, 880, 3)
    img_pixel2 = img_pixel2.transpose(0, 3, 1, 2)
    print('img_pixel_transposed:',img_pixel2.shape) #(1, 3, 1320, 880)

    W = np.zeros((1, 3, 4, 4), dtype=np.uint8)
    W[0, 0, 1, :] = 1  # 4x4행렬이 있는데
    b = np.zeros(1)
    conv = Convolution(W, b)

    out_peng = conv.forward(img_pixel2)
    print(out_peng.shape) #(1, 1, 1317, 877)
    # ValueError: shapes (1155009,48) and (16,1) not aligned: 48 (dim 1) != 16 (dim 0)
    # W의 c가 맞지 않아서 였을까 - yes

    # 그래프 그리기
    # img = img_pixel.squeeze()
    # print('img_peng',img.shape) # 3차원으로 되어있어서 안됨
    # squeeze() : Remove single-dimensional entries from the shape of an array.
    # because in this case, the c = 3 which is not a single-dimensional entry
    # I can do 1) refer to ex08 and separate the c 2) flatten?

    # Red, Green, Blue에 해당하는 2차원 배열들을 추출
    # img_r = img_pixel[:, :, 0] #Red panel
    # img_g = img_pixel[:, :, 1] #Green panel
    # img_b = img_pixel[:, :, 2] #Blue panel
    # how can I give this value to the parameter of c instead of 3?
    # rejected: couldn't figure out how hehe

    # img = img_pixel.flatten()
    # print('img_peng', img.shape) #img_peng (3484800,)

    # img = out_peng.reshape((1317, 877))
    img = out_peng.squeeze() #응 그냥 됨...^0^;;
    plt.imshow(img)
    plt.show()

