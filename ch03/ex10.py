from ch03.ex05 import softmax
import numpy as np

if __name__ == '__main__':

    # 1차원 배열의 softmax -> 계산 잘 해준다
    x = np.array([1,2,3])
    s = softmax(x)
    print(s)

    X = np.array([[1, 2, 3],
                 [4, 5, 6]])
    s = softmax(X)
    print(s)

    # Numpy Broadcast(브로드캐스트)
    # Numpy array의 축(axis)
    # axis = 0 : row의 인덱스가 증가하는 축
    # axis = 1 : column의 인덱스가 증가하는 축
    # 실행은 되지만 어떻게 계산되었을까? scalar인 값이 broadcasting 이라는 단계를 거쳐서 배열 * 배열 계산이 됨
    # 하지만 X는?
    # 행별로 ... 해줘야한다

    # array와 scalar 간의 브로드캐스트
    # (n,m)의 array와 (m,) 1차원 array는 브로드 캐스트가 가능하다
    x = np.array([1,2,3])
    print('x =',x)
    print('x + 10 =', x + 10)
    # broadcasted 10 (a scalar value) to fit the number of elements in array x

    # 배열과 배열간의 broadcasting은 고민은 해보아야한다 (Broadcasting between arrays)
    # 이차원 array와 일차원 array간의 브로드캐스트에서는 (n,m) array와 (n,) 1차원 array인 경우에는
    # 1차원 array를 (n,1) shape으로 reshape를 하면 브로드캐스트 가능
    X = np.arange(6).reshape(2,3)
    print('X shape:', X.shape)
    print('X = \n',X)

    #
    a = np.arange(1,4)
    print('a', a)
    # when the number of column = number of ~ of the 1dimesional array
    # repeat the same numbers

    print(X + a) # why nottt

    b = np.array([10, 20])
    # 더하기가가능할까? 모양을 맞출수가 없다
    # 모양을 맞춘다: 한쪽 방향으로 맞춘다
    print()
    # How about in a separated column?

    #broad cating이 1이 ㄹ때
    # 양측의 컬럼이 같을 때
    # 브로드캐스팉ㅇ일 ㄹ=가능한 경우
    # 같은 인덱스 배열이면서 둘 중 하나가 -> 컬럼 개수랑 원소의 갯수가 같아야 브로드 캐스틍

    # b를 reshape 해준다
    # print(b'b shape', b.shape)
    # print(X+b)

    np.random.seed(2020)
    X = np.random.randint(10, size = (2,3))
    print('X =', X)

    # 1차원 -> 배열, multi-dimension은 행렬이라고 일컫는다

    #1. 행렬 X의 모든 원소들 중 최댓값(m)을 찾아서, X에서 찾은 최댓값을 사용하여 X - m을 계산해서 출력

    # my_idea -> flatten 함수를 써서 1차원으로 만들어준 다음, 거기서 max를 찾으면 전체의 m을 찾을 수 있지 않을까
    # flattened = X.flatten
    # print(f'flattened = {flattened}') # 주소값만 준다 # 어떻게해야 출력해주지
    # max_flat = np.max(flattened)
    # print('max_flat',max_flat)

    # teacher's solution
    m = np.max(X)
    print(f'm = {m}')
    result = X - m # (2,3) shape의 2차원 배열과 스칼라 간의 broadcast
    print(f'X - m = \n {result}')


    #2. X의 axis = 0방향의 최댓값들을 찾아서, x의 각 원소에서 그 원소가 속한 컬럼의 최댓값을 뺀 행렬을 출력
    # 원소 3개 짜리 1차원 배열
    max0 = np.max(X, axis=0)
    print(f'max0 = {max0}, \n shape = {max0.shape} \n X - max0 = \n {X -max0}')
    # [6,8,3]의 값을 갖는 1차원 리스트. A의 컬럼의 갯수와 B의 행의 갯수 (max0) 가 같이 때문에 계산이 수월하다.
    # 1차원 리스트는 필요한 행의 수만큼 broadcasting 되어서 원소들끼리 계산되어진다

    # 3. X의 axis = 1방향의 최댓값들 (각 row들의 최댓값)을 찾아서, X의 각 원소에서 그 원소가 속한 row의 최댓값을 뺀 행렬을 출력
    max1 = np.max(X, axis =1)
    print('max1',max1, 'shape', max1.shape) # 행별 최댓값
    # max1 을 사용해서 바로 빼주려고 했을 때: operands could not be broadcast together with shapes (2,3) (2,)
    # 그래서 꼼수를 쓰려고 reshape을 시켜줬는데:
    # max1.reshape(2,3) ValueError: cannot reshape array of size 2 into shape (2,3)
    # print(X - max1)

    # 위의 코드에서 잘 못된점은 A의 컬럼의 갯수와 B의 행의 갯수 (max0) 가 같지 않다
    # (2,3) != (2,) 그래서 B를 (2,1)로 reshape하면 컬럼방향으로 늘려서 계산이 가능해 진다
    m = max1.reshape(2,1)
    result = X - m
    print('max1_result: \n',result)

    # reshape을 하지 말고 A를 transpose해주면 A의 ncol과 B의 nrow의 갯수가 맞아서 계산이 가능해진다
    X_t = X.T #transpose(전치행렬): 행렬의 행과 열을 바꾼 행렬
    print(X_t)
    m = np.max(X_t, axis = 0) #전치 행렬에서 axis = 0 방향으로 최댓값을 찾음
    result = X_t - m # 전치 행렬에서 최댓값들을 뺌
    result = result.T # 전치 행렬을 다시한번 Transpose (원래 방향으로 돌림)
    print(result)

    # 4. X의 각 원소에서, 그 원소가 속한 컬럼의 최댓값을 뺸 행렬의 모든 원소의 합

    # my solution
    max0 = np.max(X, axis = 0)
    sum0 = sum(X - max0)
    print(sum0)

    # teacher's solution
    m = np.max(X, axis =0)
    result = X - m
    s = np.sum(result, axis = 0)
    print(s)

    # 5. X의 각 원소에서, 그 원소가 속한 row의 최댓값을 뺀 행렬의 row별 원소의 합계

    # my solution
    X_t = X.T
    max1 = np.max(X_t, axis = 0)
    result = sum(X_t - max1)
    final = result.T
    print(final)

    # teacher's solution
    m = np.max(X, axis =1)
    result = X - m.reshape((2,1))
    s = np.sum(result, axis = 1)
    print(s)

    # 표준화: 평균 mu, 표준편차 sigma -> (x - mu)/sigma
    # 각 컬럼 = feature (i.e. sepal length, sepal width, ..) => 각 컬럼 별로 평균을 계산 (axis = 0)
    # print(X)
    # mu = np.mean(X, axis = 0)
    # print('mu =',mu)
    # sigma = np.std(X, axis =0)
    # normalization = (X - mu)/sigma
    # print('normalization =', normalization) # RuntimeWarning: invalid value encountered in true_divide (WHYYY)
    # nan이 발생하는 경우는 std = 0 여서 인 것 같음!

    # teacher's solution
    # 컬럼 방향으로의 표준화는 수학 공식 그대로 가능
    print(X)
    X_new = (X - np.mean(X, axis = 0))/np.std(X, axis =0)
    # 평균이 0이되고. 표준편차가 1이되는것이 normalization

    # axis = 1 방향으로 표준화
    print('X  =', X)
    mu = np.mean(X, axis = 1)
    print('mu1 =', mu)
    sigma = np.std(X, axis = 1)
    print('sigma1 =', sigma)
    X_T = X.T
    X_n = ('X_n =',((X_T - mu)/sigma).T)
    print(X_n)

    # teacher's solution
    mu = np.mean(X, axis =1).reshape((2,1))
    sigma = np.std(X, axis = 1).reshape((2,1))
    X_new = (X-mu)/sigma
    print('X_new', X_new)

    # teacher's solution2
    X_t = X.T
    X_new = (X_t - np.mean(X_t, axis =0))/np.std(X_t, axis =0)
    X_new = X_new.T
    print(X_new)
    # 왜 이거랑 내꺼랑 같은 값을 주었을까?

    # 보통 컬럼방향으로 평균이 의미가 있고, 계산을 구하기 때문에 axis =1 인 경우는 많이 사용되지 않을 것
    # reshape를 너무 자주 써주면, 그것 때문에 모양을 맞추는 문제가 생길 수가 있다 (계산비용이 높을 수도 있음)
    # 그래서 transpose를 사용
    







