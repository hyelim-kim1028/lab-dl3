"""
6.2 가중치 최깃값: Y = X @ W + b

Y = X @ W + b
f(Y) = f(X@W + b)
activation function like ReLU, Sigmoid

신경망의 가중치 행렬(W)를 처음에 어떻게 초기화를 하느냐에 따라서
신경망의 학습 성능이 달라질 수 있다.
weight의 초깃값을 모두 0으로 하면 (또는 모두 균일한 값으로 하면) 학습이 이루어지지 않음.
그래서 weight의 초깃값은 정규 분포를 따르는 난수를 랜덤하게 추출해서 만든다
그런데, 정규 분포의 표준 편차에 따라서 학습의 성능이 달라짐
1) weight 행렬의 초기값을 N(0,1) 분포를 따르는 난수로 생성하면, 활성화 값들이 0과 1 주위에 치우쳐서 분포하게 된다
 -> 역전파에 gradient 값들이 점점 작아지다가 사라지는 현상이 발생하게 된다
 -> 기울기 소실 (gradient vanishing)
2) weight 행렬의 초기값을 N(0, 0.01) 분포를 따르는 난수로 생성하면, 활성화 값들이 0.5 부근에 집중됨
-> 뉴런 1개짜리 신경망과 다를바 없음 ( 뉴런을 100개씩 둘 이유가 없어짐)
-> 이런 현상을 딥러닝에서는 "표현력(representational power)이 제한이 된다" 라고 한다
-> x1, x2, x3 .. 에 대한 효과 등등, 다양한 효과들이 표현이 되어야하는데, 한쪽에만 치우친 값들만 보여줌,,, ㅜㅜ
3) Xavier 초깃값: 이전 계층의 노드(뉴런)의 개수가 n개이면, N(0, sqrt(1/n)) 인 분포를 따르는 난수로 생성하는 것.
-> 활성화 함수가 sigmoid 또는 tanh인 경우에 좋음. (relu에는 적당하지 않다)
4) He 초깃값: 이전 계층의 노드(뉴런)의 개수가 n개이면, N(0, sqrt(2/n))인 분포를 따르는 난수로 생성하는 것
-> 활성화 함수가 ReLU인 경우에 적당
"""

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1/(1 + np.exp(-x))

def tanh(x): #hyperbolic tangent
    return np.tanh(x)

def relu(x):
    return np.maximum(x, 0)

if __name__ == '__main__':
    #은닉층(hidden layer)에서 자주 사용하는 3가지 활성화 함수 그래프
    x = np.linspace(-5, 5, 1000)
    y_sig = sigmoid(x)
    y_tanh = tanh(x)
    y_relu = relu(x)
    plt.plot(x, y_sig, label = 'Sigmoid')
    plt.plot(x, y_tanh, label = 'tanh')
    plt.plot(x, y_relu, label = 'relu')
    plt.title('Activation Functions')
    plt.legend()
    plt.ylim((-1.5, 1.5))
    plt.axvline(color='0.9')
    plt.axhline(color='0.9')
    plt.axhline(1, color='0.9')
    plt.axhline(-1, color='0.9')
    plt.show()


    # std = 1 일 때, std = 0.01일 때 이 세가지 activation functions는 각각 어떻게 처리(?) 하느냐
    # 가상 신경망에서 사용할 테스트 데이터를 생성
    np.random.seed(108)
    x = np.random.randn(1000,100) # 미니배치라고 생각하면 된다 (1000 rows, 100 columns)
                    # (sample 갯수, feature 갯수)
                    # 정규분포를 따르도록 만들었다.
                    # 어느 단위를 맞추기 위해서 정규화를 한다 (randn: 평균과 표준편차를 갖는 것으로 데스트 세트 자체가 정규화 되었다)
                    # 첫번째 input layer에 n = 100 -> 1000개를 100개씩 나눈 미니배치라고 하면 된다
                    # [[x1, x2 ... x100], [x101, x102, ... x200], .... [...x1000]] -> we are retrieving each list as one array to put in the process of ANN

    # z-score, min-max normalization,,,
    node_num = 100 #은닉층의 노드(뉴런) 개수
    hidden_layer_size = 5 #은닉층의 갯수
    activations = dict() # 데이터가 각 은닉층을 지날 때 마다 출력되는 결과들 저장할 딕셔너리

    # 은닉층에서 사용하는 가중치 행렬 * 5 -> 은닉층의 갯수
    #     # 이 행렬이 평균 = 0, 표준편차 1인 정규분포(N(0,1))를 따르는 난수로 가중치 행렬을 생성
    #     # N(0,1): N for normalization, N(average, std)
    # w = np.random.randn(node_num, node_num)
    #         # 행 갯수는 input data의 컬럼 데이터와 같아야하고 (= node_num), column의 갯수는 은닉층의 노드와 같아야한다
    #         # a = X dot W
    # a = x.dot(w) # 두번째에서는 z가 x가 되어야한다
    # z = sigmoid(a) #첫번째 은닉층을 지난 결과값
    # activations[0] = z

    # my solution
    # for i in range(hidden_layer_size):
    #     if i == 0:
    #         w = np.random.randn(node_num, node_num)
    #         a = x.dot(w)
    #         z = sigmoid(a)
    #     else:
    #         w = np.random.randn(node_num, node_num)
    #         a = z.dot(w)
    #         z = sigmoid(a)
    #     activations[i] = z
    #     # print('z =', z)

    # 어짜피 x로 나오면 나온 값에서 올라가서 들어오기 때문에 if-구문까지 쓸 필요는 없었다

    # w 에 대한 for구문 만들기
    weight_init_type = {
        'std = 0.01': 0.01,
        'xavier': np.sqrt(1/node_num),
        'He': np.sqrt(2/node_num)
    }
    input_data = np.random.randn(1_000, 100)

    for k, v in weight_init_type.items():
        x = input_data
        # 처음에 데이터가 은닉층이 들어가야하는데, x = input_data 를 주지 않으면,
        # 한번 신경망을 통과한 아이를 다시 x로 줘서 첫번째 층부터 주는 것

    # teacher's solution
    # 입력 데이터 x를 5개의 은닉층을 통과 시킴
        for i in range(hidden_layer_size):
            # 은닉층에서 사용하는 가중치 행렬 * 5 -> 은닉층의 갯수
            #     # 이 행렬이 평균 = 0, 표준편차 1인 정규분포(N(0,1))를 따르는 난수로 가중치 행렬을 생성
            #     # N(0,1): N for normalization, N(average, std)
            # w = np.random.randn(node_num, node_num)
            # w = np.random.randn(node_num, node_num) * 0.01 #N(0, 0.01)
            # w = np.random.randn(node_num, node_num) * np.sqrt(1/node_num) #N(0, sqrt(1/n))
            # w = np.random.randn(node_num, node_num) * np.sqrt(2/node_num) #N(0, sqrt(2/n))
            w = np.random.randn(node_num, node_num) * v
            a = x.dot(w) # 활성화 함수 적용 -> 은닉층의 output
            # x = sigmoid(a)
            x = tanh(a)
            # x = relu(a)
            activations[i] = x # 그래프를 그리기 위해서 출력 결과를 저장

    # 교재 p.204의 히스토그램 그리기

    # x-axis = activations[i]
    # flatten z? because it's a multi-dimensional array?
    # how can I put the range huhuhuuhuhuh wtf

        for i, output in activations.items():
        # i = index in number # output = output data
            plt.subplot(1, len(activations), i+1)
            # subplot(nrows, ncols, index) -> 하나의 화면에 그래프를 몇개를 그릴 것이냐?!
            # In subplot(), index starts from 1 (python starts its index from 0) -> so, the author gave i + 1 in its index parameter
            # i는 양수, index >= 0
            plt.title(str(i+1) + "-layer")
            plt.hist(output.flatten(), 30, range = (-1,1))
        plt.show()

    # 난 왜 때문에 반대로 나오지







