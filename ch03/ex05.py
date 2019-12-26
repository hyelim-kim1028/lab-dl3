import numpy as np
from ch03.ex01 import sigmoid


def init_network():
    """ 신경망(neural network)에서 사용되는 가중치 행렬과 bias 행렬을 생성
        교재 p.88 그림 3-2 참조
        입력층(input layer): 벡터/입력값 (x1, x2) # 1X2 행렬이라고 가정
        은닉층: 2개
        - 1st 은닉층: 뉴런 3개
        - 2nd 은닉층: 뉴런 2개
        # 은닉층 = hidden layer
        출력층(output layer): 출력값(y1, y2) #1X2 행렬
        W1, W2, W3, b1, b2, b3를 난수로 생성
        """
    # 가중치행렬, bias 행렬이 각각 3개씩 필요하다
    np.random.seed(1224)
    network = dict() # 가중치/bias 행렬을 저장하기위한 딕셔너리 -> 리턴값

    # x @ W1 + b1: 1x3 행렬
    # (1x2) @ (2x3) + b => W = array(2X3) 행렬이어야한다
    # In matrix multiplication, we have to match the number of columns of an array A and the number of rows of an array B
    # This is what we have to keep in mind when deciding for an array
    network['W1'] = np.random.random(size = (2,3)).round(2)
    network['b1'] = np.random.random(3).round(2)
                    # 1차원 리스트 (원소 3개 짜리)

    # z1 @W2 + b2: 뉴런의 갯수가 2개이고, 출력층이로 가야한다 => 1x2 행렬
    # (1x3) @ (3x2) + (1x2)
    network['W2'] = np.random.random(size = (3,2)).round(2)
    network['b2'] = np.random.random(size = (2)).round(2)

    # z2 @ W3 + b3: 1x2 행렬
    # (1x2) @ (2x2) + (1x2)
    network['W3'] = np.random.random((2,2)).round(2)
    network['b3'] = np.random.random(2).round(2)

    return network

# 앞으로 계속 진행한다는 뜻에서 forward 라는 함수 생성
def forward(network, x):
    """
    순방향전파(forward propagation). 입력 -> 은닉층 -> 출력

    :param network: 신경망에서 사용되는 가중치/bias 행렬들을 저장한 dict
    :param x: 입력 값을 가지고 있는 (1차원) 리스트 [x1, x2]
    :return: 2개의 은닉층과 출력층을 거친 후 계산된 출력값. [y1, y2]
    """
    # 가중치 행렬:
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    # bias 행렬
    b1, b2, b3 = network['b1'],network['b2'],network['b3']
    # 은닉층에서 활성화 함수: sigmoid 함수
    a1 = x.dot(W1) + b1
    z1 = sigmoid(a1) #첫번째 은닉층 전파
    z2 = sigmoid(z1.dot(W2) + b2) #두번째 은닉층 전파
    # 출력층: z2 @ W3 + b3
    # 일관성을 유지하기위해서 항등함수를 만들어서 사용 -> ?? 교재에서는 그렇게 함ㅇㅅㅇ!
    y = z2.dot(W3) + b3
    # return identity_function(y) # 출력층에 활성화 함수를 적용후 리턴
    return softmax(y) # 출력층의 활성화 함수로 softmax 함수를 적용 -> 숫자가 아닌 클래스를 예측하고 싶을 때 (i.e. iris의 품종)

# 출력층의 활성화 함수 1) 항등 함수 : 회귀(regression) 문제에서 보통 사용이 된다
def identity_function(x):
    return x #자기 자신을 리턴

# 출력층의 활성화 함수 2 - softmax: 분류(classification) 문제
def softmax(x):
    """ [x1, x2, x3 ... x_k, ..., x_n] 일 때,
        y_k = exp(x_k) / [sum i to n exp(x_i)]
                        # sigma i 부터 n 까지 exp(x_i)의 전체 총합을 계산
        softmax 함수의 리턴 값은 0 ~ 1의 값이 되고, 모든 리턴 값의 총합은 1이다 => 확률처럼 생각할 수 있다
        softmax 함수의 이런 특징 때문에 softmax 함수의 출력값으 확류로 해석할 수 있다
        그래서 이 함수가 분류의 문제에서 출력층 활성화함수로 많이 이용된다
        # 지수를 사용해서 infnite(무한대)에 쉽게 다다른다는 문제점이 있다
        # 그렇기 때문에 s
    """
    # return np.exp(x)/np.sum(np.exp(x))
    max_x = np.max(x) #배열의 x원소들 중 최대값을 찾음 (a scala)
                      # array: x = [1, 2, 3], scala: m = 3 일 때,
                      # x - m: [1, 2, 3] - 3 = [1, 2, 3] - [3, 3, 3] 처럼 계산해 준다
                      # broadcasting
                      # exp(x) = [e^1, e^2, e^3]
    y = np.exp(x - max_x) / np.sum(np.exp(x - max_x))
    return y
    # why the maximum number?
    # if we subtract the elements with the max.number


if __name__ == '__main__':
    network = init_network()
    print('W1 =',network['W1'], sep = '\n') # 2x3 행렬
    print('b1 =',network['b1'], sep = '\n') # 2x3 행렬
    print('W2=',network['W2'], sep = '\n') # 2x3 행렬
    print('b2 =',network['b2'], sep = '\n') # 2x3 행렬
    print('W3 =',network['W3'], sep = '\n') # 2x3 행렬
    print('b3 =',network['b3'], sep = '\n') # 2x3 행렬
    print(network)
    print(network.keys()) #values(): 값만 확인할 때, items(): key:value format으로 보여준다, keys: 키값만 리턴

    # forward() 함수 테스트
    x = np.array([1,2])
    y = forward(network, x)
    print('y = ', y) #[1.72377968 2.16059189]

    # 어떻게 활용할 것인가는 나중 문제 => 이 상황에서는 일종의 회귀문제이다
    # 출력층의 함수를 바꿔주면 숫자가 변할 수 있다 (i.e. 0~1 사이에 넣어주기)
    # 아무 함수도 사용하지 않는 경우 => 교재에서 사용한 항등함수를 이용한 경우 -> 회귀

    #softmax() 함수 테스트
    print('x=', x)
    print('softmax =', softmax(x))

    # x의 순서를 유지해준다
    x = [1,2,3]
    print('softmax =', softmax(x))
    # 크기차이가 난다
    # 반환값은 0~1사이로 유지해준다

    # disadvantage: exponential (지수승)을 사용해주기 때문에, 금방 값이 커져버려서 infinite을 reach 해버린다
    x = [1e0, 1e1, 1e2] # e = 10 # [1,10,100]
    print('x =', x)
    print('softmax = ',softmax(x)) #비율이 크기 때문에 100은 1, 나머지는 거의 0 (차이가 많이 난다)

    x = x = [1e0, 1e1, 1e2, 1e3]
    print('x = ', x)
    print('softmax =', softmax(x))
    # nan -> not an error, it refers to an infinity
    # 지수를 사용하는 것이 softmax의 가장 큰 문제

















