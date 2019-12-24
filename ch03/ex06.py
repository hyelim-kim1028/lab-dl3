"""
MNIST 숫자 손글씨 데이터 세트
# .pkl : https://fileinfo.com/extension/pkl
# http://yann.lecun.com/exdb/mnist/
"""
from PIL import Image
import numpy as np
from dataset.mnist import load_mnist

def img_show(img_arr): # 이미지 출력 함수
    """ Numpy 배열 (ndarray) 로 작성된 이미지를 화면에 출력 """
    img = Image.fromarray(np.uint8(img_arr)) #Numpy 배열 형식을 이미지로 변환
                        # 0 ~255 까지만 들어갈 수 있는데, refers to 8 bit (2^8까지 들어갈 수 있는것, in python 0 ~ 2^(8)-1 까지 가능)
                        # -128 ~ 127 or -2^7 ~ (2^7-1) 까지 가능하다 (음수와 양수로 반반한 것)
                        # 8 bit는 1 bit에 (0,1) 을 갖기때문에 2^8의 variety 를 갖는다
    img.show()


if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = load_mnist(normalize= False)
                                        # 자동으로 데이터세트 패키지 아래 파일들을 만들어 주었다
    # 인터넷이 연결되어있어야한다
    # 두개의 튜플을 리턴해주는 함수 (첫번째 튜플: 학습 이미지 데이터, 학습 데이터 레이블) (두번째 튜플: 테스트 이미지 데이터, 테스트 데이터 레이블)
    print('X_train shape', X_train.shape)
    # (60000, 784): 28x28 크기의 이미지 60,000 개 -> 1차원으로 펼쳐져있어서 그대로는 사용 불가!
    print('y_train shape:', y_train.shape)
    # (60000,): 6만개 손글씨 이미지 숫자(레이블)

    # 학습 세트의 첫번째 이미지
    img = X_train[3]
    # 28x28 이라는 행과 열의 모양을 갖게끔 저장?을 해준다
    img = img.reshape((28, 28)) #28x28 pixel # 1차원 배열을 28x28 형태의 2차원 배열로 변환
    print(img) #255 가 가장 진한, 0은 하양 => i.e. 43은 희미하게 써져있는 것
               # colors had been saved in each pixel
    img_show(img)

    # color 이미지는 RGB 3가지 색깔이 필요해서 -> 3가지 레이어가 필요하다
    # 투명도를 더하고 싶다면 1장 더! ㅎㅎ

    # one_hot_label, flatten
    (X_train, y_train), (X_test, y_test) = load_mnist(normalize= True, flatten = False, one_hot_label= True)
                                          # todo al contrario del default (que pasta)
    print('X_train.shape:', X_train.shape) # (60000, 1, 28, 28) # 28x28 크기의 흑백 이미지 60,000개
                                           # 이전에는 (60000, 784) 라고 들어가 있었는데 flatten = False가 되어서 펼쳐 놓았을 때: 784, 그렇지 않았을 때: 28x28
                                           # 1: 흑백 이미지 (컬럼이미지는 3이 온다 => 3개의 판이 오기 때문에)
                                            # flatten = False인 경우, 이미지 구성을 (컬러, 가로, 세로) 형식으로 표시함
                                                                    # 컬러를 나타내는 레이어의 갯수? 정도인듯
    print('y_train.shape', y_train.shape)
    # one_hot_label = True 인 경우 #(60000, 10)을 리턴
    # 행과 열을 가지고 있다 (이전 -> 1차원으로 (60000, ) 리턴)
    # one_hot_encoding 형식으로 숫자 레이블을 출력
    # 5 -> [0 0 0 0 0 1 0 0 0 0] # 5의 자리만 1로 해주고 나머지는 0으로 표시
    # 9 -> [0 0 0 0 0 0 0 0 0 1]
    print('y_train[0]:', y_train[0]) # y_train[0]: [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
                                     # 여러개의 레이블 중에서 딱 하나의 핫한 아이가 있다 라는 의미 (하테하테 ㅇ-ㅇ!!)

    # 위에서 normalize를 false로 줬을 때: 64 251 253 220 이런 값들이 들어왔다
    # normalize 했을 때는: # 0.09411765 0.44705883 0.8666667 이런 값들이 온다!  (normalized values)
    img = X_train[0]
    print(img)
    # normalize = True 인 경우, 각 픽셀의 숫자들이 0 ~ 1 사이의 숫자들로 정규화됨


