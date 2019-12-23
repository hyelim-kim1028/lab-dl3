"""
선형대수 (Linear Algebra)
- 행렬의 내적(dot product): A의 ncol과 B의 nrow가 같아야 한다
- A (n * m) * B (m * l) = C(n * l)
- A * B = C 에서 C 는 n * l의 ncol과 nrow를 갖는다
-

# 변수를 선언할 때, 대문자로 선언하면 (i.e. A, B, C...), 2차원 이상의 ndarray (행렬, 행과 열을 갖는 경우)
# 소문자로 선언할 때엔 1차원 ndarray
# python 에서는 변수 선언을 small letter/lower case로 쓸 것을 권장한다
"""

import numpy as np

x = np.array([1,2])
W = np.array([[3,4], [5,6]])
dp = x.dot(W)
print(dp)
print(W.dot(x)) #원래는 안되야 맞지만 numpy가 알아서 해준 것 # 한쪽이 1차원이면 알아서 계산해준다
                # A의 m과 B의 m이 일치하지 않기 때문에 안되는게 맞음

A = np.arange(1, 7).reshape(2,3) #arange = set a range for ndarray
print(A) # Before applying reshape(2,3) = 1-dimensional array
        # After applying reshape(2,3) = became a array with an array with two rows and three columns
B = np.arange(1,7).reshape(3,2)
print(B)
print(A.dot(B))
print(B.dot(A))
    # 행렬의 내적/곱셈은 교환 법칙이 성립하지 않는다

# ndarray.shape -> (x, y)
# 차원마다 보여주는 모습이 다르다 # 1차원 (x, ), 2차원(x, y), 3차원(x, y, z)
x = np.array([1,2,3])
print(x.shape) #(3,) # array가 가진 원소의 갯수가 3개인 것이지, nothing to do with the number of row/column
x = x.reshape((3,1))
print(x.shape) #(3, 1) # three rows with one column
# 1차원 배열과 2차원 배열은 속성이 달라서 dot 연산이 (모양이 같으면 가능하고, 다르면 불가능하다)

x = x.reshape((1,3))
print(x) #[[1 2 3]]
print(x.shape) # (1, 3) - one row with three columns
# reshape을 해버리면 dot 연산이 가능하기도 하고 불가능하기도 하다

# dot 연산: 신경망을 표현하려고! -> 행렬을 사용하면 쉽게 신경망을 표현할 수 있다 (p.82)

