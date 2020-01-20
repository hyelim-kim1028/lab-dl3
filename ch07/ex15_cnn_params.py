"""
CNN이 사용하는 파라미터 (filter W, bias b)의 초깃값과 학습 끝난 후의 값 비교
"""
import numpy as np
import matplotlib.pyplot as plt

# 여러개의 값이 한 화면에 보이게끔 해준다
from ch07.simple_convnet import SimpleConvNet
from common.layers import Convolution


def show_filters(filters, num_filters, ncols = 8):
    """ CNN 필터를 그래프로 출력 """
    nrows = np.ceil(num_filters / ncols) # ceil = ceiling # 올림한 숫자 # floor() - 소숫점 버려버리기
    # num_filters = 30일 때, ncol = 8면, nrow = 4 => 컬럼 갯수가 있으면 row는 계산하면 된다
    for i in range(num_filters): #필터 개수만큼 반복
        # subplot 설정
        plt.subplot(nrows, ncols, (i + 1), xticks = [], yticks = [])
        # subplot에 이미지 그리기
        plt.imshow(filters[i, 0], cmap = 'gray')
                        # 5 * 5 행렬이 추출된다
                        # 4차원 행렬 [ [3차원], [3차원] ... ] 에서, i번째
                        # -> [3차원] 리스트 한개를 꺼내서 ->
                        # [i,0]은 이 3차원 리스트에서 0번째 판한개를 꺼내주는 것
    plt.show()

if __name__ == '__main__':
    cnn = SimpleConvNet()
    # 학습시키기 전 파라미터 - 임의의 값들로 최기화된 필터
    before_filters = cnn.params['W1']
    print(before_filters.shape) # (30, 1, 5, 5) # (filter_num, n_channel, filter size 5 * 5)
    # 학습 전 파라미터를 그래프로 출력
    show_filters(before_filters, num_filters= 30, ncols=8)

    # 학습 끝난 후 파라미터
    # after_params = cnn.load_params('cnn_params.pkl')
    # after_filters = after_params['W1'] # load_params 에 리턴값이 없어서 에러가 났다

    # pickle 파일에 저장된 파라미터를 cnn의 필드로 로드
    cnn.load_params('cnn_params.pkl')
    after_filters = cnn.params['W1']
    # 학습 끝난 후 갱신(업데이트)된 파라미터를 그래프로 출력
    show_filters(after_filters, 30, 8)


    # 실제 이미지에 적용
    cnn = SimpleConvNet()
    # 학습시키기 전 파라미터 - 임의의 값들로 최기화된 필터
    before_filters = cnn.params['W1']
    print(before_filters.shape)  # (30, 1, 5, 5) # (filter_num, n_channel, filter size 5 * 5)
    # 학습 전 파라미터를 그래프로 출력
    show_filters(before_filters, num_filters=16, ncols=4)

    # 학습 끝난 후 파라미터
    # after_params = cnn.load_params('cnn_params.pkl')
    # after_filters = after_params['W1'] # load_params 에 리턴값이 없어서 에러가 났다

    # pickle 파일에 저장된 파라미터를 cnn의 필드로 로드
    cnn.load_params('cnn_params.pkl')
    after_filters = cnn.params['W1']
    # 학습 끝난 후 갱신(업데이트)된 파라미터를 그래프로 출력
    show_filters(after_filters, 16, 4)

    # 학습 끝난 후 갱신된 파라미터를 실제 이미지 파일에 적용
    lena = plt.imread('lena_gray.png') # imread는 png만 바로 shape으로 적용하게 해준다
    # jpg는 PIL를 사용해서 열어야한다
    print(lena.shape) #(256, 256) #이미지 파일이 numpy array로 변환됨 -> 2차원

    # 이미지 데이터를 convolution layer의 forward()메소드에 전달
    # 우리의 convolution 레이어는 오직 4차원만 받음 -> 레나 이미지를 4차원으로 만들어줘야한다
    # 2차원 배열을 4차원 배열로 변환
    lena = lena.reshape(1, 1, *lena.shape) # *lena.shape = (256, 256)
    for i in range(16): # 필터 16개에 대해서 반복
        w = cnn.params['W1'][i] # 갱신된 필터w
        # b = cnn.params['b1'][i] #갱신된 bias
        b = 0 # did not use bias because the image might get distorted
        w = w.reshape(1, *w.shape) #3d -> 4d
        conv = Convolution(w,b) # convolution 레이어 생성
        # 필터 - 학습이 끝난 상태의 필터 (5x5의 작은 필터를 보낸것)
        out = conv.forward(lena) #이미지 필터에 적용
        # pyplot을 사용하기 위해서 4d-> 2d
        out = out.reshape(out.shape[2], out.shape[3])
        # subplot에 그림
        plt.subplot(4, 4, i+1, xticks = [], yticks = [])
        plt.imshow(out, cmap = 'gray')
    plt.show()














































