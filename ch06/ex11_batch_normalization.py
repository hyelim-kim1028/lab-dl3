"""
Batch_normalization (배치 정규화)
    : 신경망의 각 층의 미니 배치 (mini-batch)를 전달할 때마다
      정규화를 normalization을 실행하도록 강제하는 방법
    -> 학습 속도 개선
    -> 파라미터(W, b)의 초기값에 크게 의존하지 않음. (*****)
    -> Xavier, He,... 이런 고민 할 필요가 없어용
    -> 과적합(overfitting)이 잘 일어나지 않음/ 억제

- we had normalized the entire train data (not the mini-batch data)
- Also, even though the train data had been normalized, after passing Affinity class (X@W + b), numbers change, as well as the value for average and std
- That is why we need to do Batch normalization in each layer/은닉 계층

y = gamman * x + beta
gamma 파라미터: 정규화된 미니 배치를 scale-up/down
beta 파라미터: 정규화된 미니 배치를 이동시킨다 (bias)
# 배치 정규화를 사용할 때는 gamma와 beta를 초기값을 설정을 하고,
  학습을 시키면서 오차역전파를 통해 계속 업데이트 해줘야하는 값이다
  (파라미터가 2개 더 생긴셈)
# dL/dW 와 같이 forward, backward,update의 과정을 거쳐야한다

# network: predict & gradient 메소드를 가지고 있다
# 예측하기 위해서는 각 노드들에서 forward가 되어야하고, gardient를 구하기 위해서는 각 노드들에서 backward가 되어야한다
# affinity, softmax,,, 모두 forward, backward 가지고 있음,,, ㅎㅎ
# In batch normalization, we have to train and transform Gamma and Beta-values along with the Weight and Bias

# 합성곱이라는 개념을 알아야 common의 layer파일의 코드가 모두 이해된다
"""
from ch06.ex02_sgd import Sgd
from common.multi_layer_net_extend import MultiLayerNetExtend
from common.optimizer import Momentum
from dataset.mnist import load_mnist
import numpy as np
import matplotlib.pyplot as plt

#p.213의 그림 그려보기
# Batch Normalization을 사용하는 신경망과 사용하지 않는 신경망의 학습 속도 비교

np.random.seed(110)

# 배치 정규화를 사용하는 신경망
bn_neural_net = MultiLayerNetExtend(input_size= 784,
                                    hidden_size_list= [100, 100, 100, 100, 100], #뉴런 100개짜리 계층 5개 생성
                                    output_size= 10,
                                    weight_init_std= 0.3, # [W1, W2, W3, W4, W5]과 ~~~~ 했을 때, std = 0.01
                                    use_batchnorm= True)


# 배치 정규화를 사용하지 않는 신경망
neural_net = MultiLayerNetExtend(input_size= 784,
                                    hidden_size_list= [100, 100, 100, 100, 100], #뉴런 100개짜리 계층 5개 생성
                                    output_size= 10,
                                    weight_init_std= 0.3, # [W1, W2, W3, W4, W5]과 ~~~~ 했을 때, std = 0.01
                                    use_batchnorm= False)

# MNIST 데이터를 불러온다
(X_train, Y_train), (X_test, Y_test) = load_mnist(one_hot_label= True)

# 학습시간을 줄이기 위해서 학습 데이터의 개수를 줄임
X_train = X_train[:1000] # 데이터 1000개만 사용
Y_train = Y_train[:1000]

train_size = X_train.shape[0] # row의 갯수
learning_rate = 0.01
batch_size = 128
iterations = 20

# max_epoch = 20
neural_accuracy = [] # 배치 정규화를 사용하지 않는 신경망의 정확도를 기록하기 위한 변수
bn_neural_accuracy = [] # 배치 정규화를 사용한 신경망의 정확도를 기록하기 위한 변수

# select an optimizer (최적화 모델)
# 신경망이 한개 이상이라면 각 신경망에 맞는 모델을 사용해줘야하기 때문에 한개 이상의 모델을 사용해야한다
# 신경망마다 모양이 달라지기 때문에
# optimizer = Sgd(learning_rate)
# SGD가 아닌 경우에는 update에서 params를 W,b이 외에 더 선언해서 문제가 날 수 있다
# 파라미터 최적화 알고리즘이 SGD가 아닌 경우에는 신경망 개수만큼 optimizer 생성
optimizer = Momentum(learning_rate)
bn_optimizer = Momentum(learning_rate)

# epoch = 0


for i in range(iterations):
    # 미니 배치를 랜덤하게 선택 (0~999 숫자들 중 128개를 랜덤하게 선택)
    mask = np.random.choice(train_size, batch_size)
    x_batch = X_train[mask]
    y_batch = Y_train[mask]


    # 배치 정규화
    # 배치 정규화를 사용하지 않는 신경망에서 gradient 계산
    gradients = neural_net.gradient(x_batch, y_batch)
    # 파라미터 업데이트 -> W와 b 변경
    optimizer.update(neural_net.params, gradients)
    # 업데이트된 배치 데이터의 정확도를 계산
    accuracy = neural_net.accuracy(x_batch, y_batch)
    # 정확도를 기록
    neural_accuracy.append(accuracy)

    # 배치 정규화를 사용하는 신경망에서 gradient 계산
    bn_gradients = bn_neural_net.gradient(x_batch, y_batch)
    # 파라미터 업데이트 -> W와 b 변경
    bn_optimizer.update(bn_neural_net.params, bn_gradients)
    # 업데이트된 배치 데이터의 정확도를 계산
    bn_accuracy = bn_neural_net.accuracy(x_batch, y_batch)
    # 정확도를 기록
    bn_neural_accuracy.append(bn_accuracy)

    print(f'iteration #{i}: without = {neural_accuracy[i]}, with = {bn_neural_accuracy[i]}')
    # 왜 이렇게 출력되는거지?
    # 1. 쌤것과 나의 것의 값이 다르다
    # 2. 쌤껀 리스트를 그냥 줘도 나오는데, 내꺼는 [i]를 줘야지 그 순서의 것으로 나온다

    # 책
    # 두 신경망을 모두 학습 => y = 정확도 (accuracy)
    # for _network in (bn_neural_net, neural_net):
    #     gradients = _network.gradient(X_batch, Y_batch)
    #     optimizer.update(_network.params, gradients)
    #
    # accuracy 계산
    # bn_accuracy = bn_neural_net.accuracy(X_batch, Y_batch)
    # accuracy = neural_net.accuracy(X_batch, Y_batch)

    # bn_neural_accuracy.append(bn_accuracy)
    # neural_accuracy.append(accuracy)

    # epoch += 1
    # if epoch >= max_epoch:
    #     break



# mini_batch = 20 학습 시키면서, 두 신경망에서 정확도 (accuracy)를 기록

# 그래프를 그린다
x= np.arange(iterations)
plt.plot(x, neural_accuracy, label = 'without BN')
plt.plot(x, bn_neural_accuracy, label = 'Using BN')
plt.legend()
plt.show()


# 좀 더디게 올라가는 그래프: 초기값에 영향을 많이 받는다
# 가파르게 올라가는 그래프: 초기값에 상관없이 자신의 일을 잘 해냄 ㅎㅎ


# momentum: v의 방향의 속도 (파라미터 갯수만큼 그 파라미터에 대한 속도, 방향 -> 즉, 각 파라미터에대한 모멘텀이 있어야한다)

# 모델 테스트해보기
# mini-batch iteration 횟수 변경하면서 실험
# weight_init_std = 0.01, 0.1, 0.3, 1.0으로 바꿔가면서 실험을 해보면 배치 정규화를 사용할 때와 사용하지 않을 때를 비교할 수 있다


