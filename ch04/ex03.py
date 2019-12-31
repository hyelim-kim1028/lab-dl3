"""
교차 엔트로피 (Cross Entropy)
    cross-entropy = -true_value * log(expected_value)
        - 일종의 확률 값이다.
        - 실제 확률 & 예측 활률을 교차해서 사용한다 => 그래서 cross entropy라고 부름
    entropy = -sum i[t_i * log(y_i)]
    #t_i: true_value i번째 (ith true value)
"""
import pickle

from ch03.ex11 import forward
from dataset.mnist import load_mnist
import numpy as np

def _cross_entropy(y_pred, y_true): #1차원만 사용 가능
    """ cross-entropy = -true_value * log(expected_value) """
    delta = 1e-7
    # 왜 델타,,,? 로그함수를 그렸을 때, 마이너스 무한대로 가는 구간이 있기 때문에, 그 것을 방지하고, 그 근처의 근사값을 구해서 마이너스 무한대로 가는 것을 방지
    # log0 = -inf 가 되는 것을 방지하기 위해서 더해줄 값
    return -np.sum(y_true * np.log(y_pred + delta)) # 0.0000001을 더해준 것 #sum of all the numbers in the array
                # vectors         vector    scalar (broadcasting)
                # np.log -> log applied all the elements in a list/array
                # element-wise multiplication (not dot product)


def cross_entropy(y_pred, y_true): # 위의 코드는 사용되지 않는다
    # 예측값의 dimension이 1차원일 때
    if y_pred.ndim == 1:
        ce = _cross_entropy(y_pred, y_true)
    elif y_pred.ndim == 2:
         ce = _cross_entropy(y_pred, y_true)/len(y_pred)
         # len(y_pred) => divided by the number of rows of y_pred
        # is the same as divided by the number of images since each row contains the information of an image
    return ce

# cross entropy를 어떻게 줄일 것인가? 가 main objective!
# 함수들의 미분을 계산해서 gradient descent (경사하강법) 적용

# 교차엔트로피에 대한 gradient descent를 계산해서 반대 부호로 변환. W의 가중치를 변환시킨 다음에 예측값을 계산하는 과정을 반복.
    # 교차 엔트로피 함수를 만든 이유: W를 조정하고, 신경망을 통과시키는 과정에 필요

if __name__ == '__main__':
    (T_train, y_train), (T_test, y_test) = load_mnist(one_hot_label = True)
    y_true = y_test[:10]

    with open('../ch03/sample_weight.pkl', 'rb') as file:
        network = pickle.load(file)
    y_pred = forward(network, T_test[:10])

    # 실제값과 예측값이 일치하는 경우
    print('y_true[0] =', y_true[0]) #7 # [0. 0. 0. 0. 0. 0. 0. 1. 0. 0.] # 원소 10개를 갖는 벡터
    print('y_pred[0] =', y_pred[0]) # 7 이미지가 될 확률이 가장 큼
    # 실질적으로 0*log(expected_value)이기 때문에, 1의 값밖에 남는게 없다
    # y_pred을 구해서 더해주는 (summation/sigma)
    print('ce =', cross_entropy(y_pred[0], y_true[0])) # 0.0029

    # 실제값과 예측값이 다른 경우
    print('y_true[8] =', y_true[8]) # 숫자 5이미지
    print('y_pred[8] =', y_pred[8]) # 6이될 확률이 가장 큼
    print('ce =', cross_entropy(y_pred[8], y_true[8])) #0.00293918838724494
    print('ce 평균 =', cross_entropy(y_pred, y_true)) #0.5206955424044282

    # ce가 적으면 true value & pred value가 같은 확률이 높다, ce가 높으면 그럴 확률이 적다

    # 만약 y_true 또는 y_pred가 one_hot_encoding이 사용되어 있지 않으면,
    # one_hot_encoding 형태롤 변환해서 Cross-Entropy를 계산
    np.random.seed(1227)
    y_true = np.random.randint(10, size=10)
    print('y_true =', y_true) # [4 3 9 7 3 1 6 6 8 8]

    # 여기서부터 틀려서 밑에서 이어서 한다
    # y_pred = np.array([4, 3, 9, 7, 3, 1, 6, 6, 8, 8])
    # # 실제값과 예측값이 모두 일치하면, entropy = 0, entropy의 평균도 0 이 나와야한다
    # print('ce 평균 =', cross_entropy(y_pred, y_true))
    # ce평균 = -100.305 # 이론상 0을 줘야하는데, 확률이 아니어서 이런 값을 준다
    # entropy는 plogp # p for probability -> p = 0 < p < 1 의 값이 들어가야한다
    # y_true 는 one_hot_encoding으로 바꿔주고, y_pred는 확률로 나타내서 사용하면 될 듯!

    # 여기서 부터 이어서 시작
    y_true_2 = np.zeros((y_true.size,10)) #row의 갯수, 컬럼의 갯수 (0~9까지의 숫자) -> 분류해야하는 X의 갯수
                                          # 10개의 0으로 채운 리스트들을 생성하고
    for i in range(y_true.size): # y_true의 인덱스에 1을 각인해줌
        y_true_2[i][y_true[i]] = 1
    print(y_true_2) #y_true의 값들을 기준으로 one_hot_encoding 시켜줌










