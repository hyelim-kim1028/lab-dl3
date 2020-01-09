"""
weight compare
- 가중치 초기값에 대한 비교
- MNIST 데이터를 사용한 가중치 초깃값과 신경망 성능 비교

# p.209 와 같은 그래프 그려보기

- https://arxiv.org/pdf/1502.01852.pdf (HE)
"""


from ch06.ex02_sgd import Sgd
from ch06.ex03_momentum import Momentum
from ch06.ex04_adagrad import AdaGrad
from ch06.ex05_ADAM import Adam
from ch06.ex07_nesterov import Nesterov
from common.multi_layer_net import MultiLayerNet
from common.optimizer import RMSprop
from dataset.mnist import load_mnist
import numpy as np
import matplotlib.pyplot as plt

# 실험 조건 세팅
weight_init_types = {
    'std = 0.01': 0.01,
    'Xavier': 'sigmoid', # 가중치 초깃값: N(0, sqrt(1/n))
    'He': 'relu' # 가중치 초깃값: N(0, sqrt(2/n))
}

#각 실험 조건 별로 테스트할 신경망을 생성
neural_nets = dict()
train_losses = dict()


for key,type in weight_init_types.items():
    neural_nets[key] = MultiLayerNet(input_size= 784,
                                     hidden_size_list= [100, 100, 100, 100], # 4개의 레이어 그리고 각 레이어의 뉴런 갯수
                                     output_size= 10,
                                     weight_init_std = type)
                                     # activation = 'sigmoid')
    # the default activation is relu, but when changed to sigmoid, the result was really bad


    train_losses[key] = [] # 빈 리스트 생성 - 실험(학습)하면서 손실값들을 저장

# there are two keys: optimizers and weight_init_type
# need to arrange on this

#MNIST train/test 데이터 로드
(X_train, Y_train), (X_test, Y_test) = load_mnist(one_hot_label = True)

iterations = 2_000 # 학습 횟수
batch_size = 128 # 1번 학습에 사용할 샘플 개수 (미니 배치)
# optimizers = Sgd(learning_rate = 0.01)
optimizers = Adam(lr = 0.01) # optimizer or how we set the parameter

# 초기값이 중요한 역할을 할수도 있고, 어떤 최적화 알고리즘을 사용을 하느냐에 따라서 (regardless of initial values), 학습되는 양? 이 다를 수도 있다
# optimizer with Adam resulted in horrible

# optimizers = dict()
# optimizers['SGD'] = Sgd(learning_rate = 0.01) #파라미터 최적화 알고리즘
# optimizers['Momentum'] = Momentum(lr = 0.01)
# optimizers['Adagrad'] = AdaGrad(lr = 0.01)
# optimizers['ADAM'] = Adam(lr = 0.01)
# optimizers['RMSprop'] = RMSprop(lr = 0.01)
# optimizers['Nesterov'] = Nesterov(lr = 0.01)

np.random.seed(109)
# 2000번 반복 하면서
for i in range(iterations):
        # 그래프를 그려보고, 그 모양이 교제에 있는 것과 비슷하면, lr, optimizer 변경 해보기
        # 미니 배치 샘플 랜덤 추출
    # X_batch = X_train[i: i + batch_size]
        #     # Y_batch = Y_train[i: i + batch_size]
    mask = np.random.choice(len(X_train), batch_size)
    # mask = np.random.choice(X_train.shape[0], batch_size)
    X_batch = X_train[mask]
    Y_batch = Y_train[mask]

        # 테스트 할 신경망 종류마다 반복

    for key, net in neural_nets.items():
        # for key in optimizers: # 여기서 자꾸 에러가 나서 안됨
            # gradient 계산
            gradients = net.gradient(X_batch, Y_batch)
                # net is the same as neural_nets[key]
            # 파라미터 (W, b) 업데이트
            optimizers.update(net.params, gradients)
            # 손실 (loss) 계산 - 리스트 추가
            loss = net.loss(X_batch, Y_batch)
            train_losses[key].append(loss)
# 손실 일부 출력
    if i % 100 == 0:
        print(f'===== Training #{i} =====')
        for key,loss_list in train_losses.items():
            print(key, ':', loss_list[-1]) # val is a list type


# x-axis: 반복 횟수, y-axis: 손실 그래프
x = np.arange(iterations)
for key, loss_list in train_losses.items():
    plt.plot(x, loss_list, label = key)
plt.title('Comparison between Weight Initialization Methods')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.show()



# 배치 정규화:
# 첫번쨰 ~ 데이터를 입력 시킬 때: 데이터들을 전부 다 normalize 시켜 놓고서 (0~1 사이의 값들로 변형) 데이터를 보냈다
# 배치 정규화란 신경망에 들어가는 베치 데이터들을  매번 모든 층에서 정규화를 진행한다?
# 활성화 함수 지나기 직전에 정규화? 지난 다음에 정규화? 논쟁의 여지가 있다
# 활성화 함수 지나기 직전에 정규화 -> 들어오는 입력값을 정규화 후에 활성화 함수에 넣는다
# 지난 다음에 정규화 -> 정규호하고 나서 가중치행렬이 곱해 질때에 정규화를 하겠다 (가중치에 곱해지는 녀석만 정규화를 하겠다)
# 넣는 위치가 두군데이다
# 어찌됐든, 모든 층에 들어가는 입력 데이터들을 모두 정규화해서 보내겠다 -> 배치 정규화
# W행렬의 초기값이 어떻게 만들어지는가에 따라 영향을 많이 받았지만, 배치 정규화를 사용하면 초기값에 영향을 받지 않고, 학습속도가 빨라진다





