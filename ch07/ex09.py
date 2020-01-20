"""
im2col 함수를 사용한 convolution 구현
"""
import numpy as np
from common.util import im2col

if __name__ == '__main__':
    np.random.seed(115)

    #p.238 그림 7-11참조
    # 가상의 이미지 데이터 1개를 생성
    # (n, c, h, w) = (이미지의 갯수, color-depth , height, width)
    x = np.random.randint(10, size = (1, 3, 7, 7))
    print(x, ', shape:', x.shape)

    # (3, 5, 5) 크기의 필터 1개 생성
    # (fn, c, fh, fw) = (이미지의 갯수, color-depth , filter-height, filter-width)
    w = np.random.randint(5, size = (1, 3, 5, 5))
    print(w, ',shape:', w.shape)
    # 필터를 stride =1, padding = 0으로 해서 convolution 연산
    # 필터를 1차원으로 펼침 -> c * fh * fw = 3 * 5 * 5 = 75

    # 이미지 데이터 x를 함수 im2col에 전달
    x_col = im2col(x, filter_h = 5, filter_w= 5, stride= 1, pad = 0)
    print('x_col:', x_col.shape) # (9, 75) # 9: oh * ow/ 마지막 convultion 모양 (3 x 3) # 75: c * fh*fw/필터를 1차원으로 펼쳐둔 것

    # 이미지도, 필터도 4차원 -> 1차원으로 풀어준다
    # 4차원 배열인 필터 w를 2차원 배열로 변환 -> dot product을 하기위해서
    # w_col = w.reshape(fn, -1)
    w_col = w.reshape(1, -1)
    print(w_col.shape) #(1, 75) # row의 갯수가 1개, 모든 원소들을 column으로
    #  One shape dimension can be -1. In this case, the value is inferred from the length of the array and remaining dimensions.
    w_col = w_col.T
    print(w_col.shape) #(75, 1)
    # 그럼 결과가 (9,1) 이 나오고 그것 3x3 배열로 다시 만들어 준다
    # dot product (w_col은 transpose를 해줘야 dot product이 가능하다)

    # 2차원으로 변환된 이미지와 필터를 dot product 연산
    out = x_col.dot(w_col)
    print('out:', out.shape)

    # dot product의 결과를 (fn, oh, ow, ?) 와 같은 형태로 변형 (reshape)
    # (fn 필터의갯수, output height, output width)
    out = out.reshape(1, 3, 3, -1) # reshape을 잘 못하면 섞여버린다 (location-wise)
    # (n, fh, fw, color depth) # n = number of images
    # 우리가 원하는 축 방향으로 되어있지 않다
    print('out:', out.shape) # (1, 3, 3, 1) = (fn, oh, ow, c)
    # 우리는 (fn, c, oh, ow)의 모양으로 되어있기를 원한다
    out = out.transpose(0, 3, 1, 2)
    print(out, 'shape:', out.shape)

    # Exercise1
    # refer to the figure 7-12 of p.238, pic 7-19 of p. 244
    # 가상으로 생성한 이미지 데이터 x와 2차원을 변환환 x_col를 사용
    # (3, 5, 5) 필터를 10개 생성 -> w.shape(10, 3, 5, 5) 로 만들어라
    #                               => 4차원의 w를 만들기 10개의 3차원짜리 w(3,5,5)
    w = np.random.randint(10, size = (10, 3, 5, 5))
    print(w, ',shape:', w.shape)

    # w를 변형(reshape) -> (n, c*fh*fw) 형식 으로 변형되어야한다 (7-19)
    w_col = w.reshape(10, -1)
    print('w_col',w_col.shape)

    w_col = w_col.T
    print('w_col.T:', w_col.shape)
    print('x_col:', x_col.shape)
    # x_col @ w.T -> dot 연산의 결과 = convolution의 결과 => dot연산의 결과를 reshape (oh x ow)

    out = x_col.dot(w_col) # (9,75) @ (75, 10)
    print('out:', out.shape)
    out = out.reshape(1, 3, 3, -1) #(1, 3, 3, 10) 같은 얘기임
    print('out:', out.shape) #out: (1, 3, 3, 10)

    # reshape된 결과에서 네번쨰 축이 두번째 축이 되도록 전치
    out = out.transpose(0,3,1,2)
    print(out, 'shape:', out.shape) # 판 1개는 3x3입니다

    # 위는 convolution의 forward에 들어갈 순서와 동일한 순서로 계산되었다

    print('========')
    # p.239 그림 7-13, p.244 그림 7-19 참조
    # (3, 7, 7) 이미지 12개 데이터를 난수로 생성 -> (n, c, h, w) = (12, 3, 7, 7) 로 만들어라
    # 앞으로는 배열의 차원을 키워서 보낼꺼예요
    x = np.random.randint(10, size = (12, 3, 7, 7))
    print('x =', x.shape)

    #(3,5,5) shape의 필터 10개 난수로 생성 -> (fn, c, oh, ow) = (10, 3, 5, 5)
    w = np.random.randint(5, size = (10, 3, 5, 5))
    print('w_shape =', w.shape)

    # stride = 1, padding = 0 일 때, , output height, output width =?
    # oh = (h - fh + 2 * p) // s + 1 = (7 - 5 + 2 * 0) // 1 + 1 = 3
    # ow = (w - fw + 2 * p) // s + 1 = (7 - 5 + 2 * 0) // 1 + 1 = 3

    # 이미지 데이터 x를 함수 im2col에 전달
    x_col = im2col(x, filter_h = 5, filter_w = 5, stride = 1 , pad = 0)
    print('x_col =', x_col.shape) #(108, 75) = (n * oh * ow, c * fh * fw)
                        # 108: 12* 3 * 3 = n * oh * ow
                        # 75: color_depth * fh * fw

    # 필터 w를 x_col과 dot 연산을 할 수 있도록 reshape & transpose: w_col -> shape?
    w_col = w.reshape(10, -1) # = (fn, c * fh * fw)
    print(w_col.shape)
    w_col = w_col.T # 그냥 w_col = w.reshape.T -> print('w_col', w_col.shape) 이 깔끔할 듯
    print('w_col =', w_col.shape)

    # x_col @ w_col
    out = x_col.dot(w_col) #(108, 75)@(75, 10)
    print('out_dot',out.shape) #(108, 10)

    # @ 연산의 결과를 reshape & transpose # (n * oh * ow, fn)
    out = out.reshape(12, 3, 3, -1) # n * oh * ow를 세개의 축으로 불리하겠다
    out = out.transpose(0,3,1,2) #
    print(out, 'shape:', out.shape) #(12, 10, 3, 3)

    # channel-first 방식을 사용해서 transpose를 계속 해줘야하는 것 (축변경)
    # tensor-flow는 channel을 마지막에 둔다 # transpose 할 필요가 없다
