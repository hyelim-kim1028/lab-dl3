"""
pooling
- 4차원 데이터를 2차원으로 변환 후에 max pooling을 구현
"""
import numpy as np
from common.util import im2col

if __name__ == '__main__':
    np.random.seed(116)

    # 가상의 이미지 데이터 (c, h, w) = (3, 4, 4) 1개를 난수로 생성 -> (1, 3, 4, 4)  # 4차원 데이터를 다룬다
    x = np.random.randint(10, size = (1, 3, 4, 4))
    print(x, 'shape:', x.shape)

    # 4차원 데이터를 2차원 ndarray로 변환
    col = im2col(x, filter_h = 2, filter_w= 2, stride = 2, pad = 0)
    # the concept of pad is quite tricky, I want some practice questions

    print(col, 'shape :', col.shape) #(4, 12) = (n * oh * ow, c * fh * fw)
                                     # 4: 박스를 몇번 움직일 수 있는가 = n = 1, oh = 2, wh = 2
                                     # 12: fh = 2, fw = 2, c = 3  = 2 * 2 * 3
    # [[5. 6. 1. 1. 3. 5. 3. 9. 3. 6. 3. 8.] -> [5,6,1,1] = R 혹은 첫번째 판의 첫 window의 위치와 correspond 하는 숫자, [3. 5. 3. 9.] = G 혹은 두번째 판의 첫 window 위치와 일치하는 값들,,,
    #  [1. 2. 8. 7. 9. 3. 6. 0. 5. 7. 5. 7.]
    #  [2. 8. 3. 3. 3. 7. 1. 5. 3. 5. 7. 6.]
    #  [7. 3. 9. 4. 3. 1. 3. 7. 9. 3. 8. 4.]]
    # 최댓값을 찾을 수 없기 때문에 모양을 변경해준다 # ..??? 왜 안된다고 할까, 4개씩 끊어서 그냥 하면 안되나 ㅠㅠ 너무 간단하게 생각하는건가

    # max pooling: 채널별로 최댓값을 찾음
    # 채널별 최댓값을 쉽게 찾기위해서 2차원 배열의 shape을 변환
    # -> 변환된 행렬의 각 행에는, 컬러 채널 별로 윈도우/필터에 포함된 값/원소들로만 이루어짐
    col = col.reshape (-1, 2 * 2) # 컬럼을 4개씩 자른다 # 2 * 2 = (fh, fw) 라는 의미로 이렇게 씀

    # 변경한것 출력
    print(col, 'shape:', col.shape) # (12,4) # transpose는 아니다
    # 교재에서 나온 숫자의 순서와 다르다

    # 어떻게 최댓값을 찾을 것인가 # 컬럼의 인덱스가 증가하는 방향,axis = 1,으로 최댓값을 찾는다

    # 각 행(row)에서 최댓값을 찾음
    out = np.max(col, axis = 1) #axis를 반드시 줘야한다
    print(out, 'shape:', out.shape)
    # [6. 9. 8. 8. 9. 7. 8. 7. 7. 9. 7. 9.] shape: (12,)
    # 각 행의 최댓값!

    # 4차원 그대로 출력되어야한다
    # 1차원 pooling의 결과를 4차원으로 변환 (n, oh, ow, c) -> (n, c, oh, ow)
    out = out.reshape(1,2,2,3)
    print(out)
    out = out.transpose(0, 3, 1, 2) # 축을 틀어서 보는 방향을 바꾼다
    print(out)
    







