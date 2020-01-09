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
from dataset.mnist import load_mnist
import numpy as np

#p.213의 그림 그려보기
# Batch Normalization을 사용하는 신경망과 사용하지 않는 신경망의 학습 속도 비교

# MNIST 데이터를 불러온다
(X_train, Y_train), (X_test, Y_test) = load_mnist(normalize= True)

learning_rate = 0.01
iteration = 20
batch_size = 128
train_size = X_train.shape[0]

# 배치 정규화를 사용하는 신경망
bn_neural_net = MultiLayerNetExtend(input_size= 784,
                                    hidden_size_list= [100, 100, 100, 100, 100], #뉴런 100개짜리 계층 5개 생성
                                    output_size= 10,
                                    weight_decay_lambda= 0.01, # [W1, W2, W3, W4, W5]과 ~~~~ 했을 때, std = 0.01
                                    use_batchnorm= True)


# 배치 정규화를 사용하지 않는 신경망
neural_net = MultiLayerNetExtend(input_size= 784,
                                    hidden_size_list= [100, 100, 100, 100, 100], #뉴런 100개짜리 계층 5개 생성
                                    output_size= 10,
                                    weight_decay_lambda= 0.01, # [W1, W2, W3, W4, W5]과 ~~~~ 했을 때, std = 0.01
                                    use_batchnorm= False)

# select an optimizer (최적화 모델)
optimizer = Sgd(learning_rate)

max_epoch = 20
neural_accuracy = []
bn_neural_accuracy = []

epoch = 0

for i in range(iteration):
    batch_mask = np.random.choice(train_size, batch_size)

    X_batch = X_train[batch_mask]
    Y_batch = Y_train[batch_mask]

    # 두 신경망을 모두 학습 => y = 정확도 (accuracy)
    for _network in (bn_neural_net, neural_net):
        gradients = _network.gradient(X_batch, Y_batch)
        optimizer.update(_network.params, gradients)

    bn_accuracy = bn_neural_net.accuracy(X_batch, Y_batch)
    accuracy = neural_net.accuracy(X_batch, Y_batch)

    bn_neural_accuracy.append(bn_accuracy)
    neural_accuracy.append(accuracy)

    epoch += 1
    if epoch >= max_epoch:
        break



# mini_batch = 20 학습 시키면서, 두 신경망에서 정확도 (accuracy)를 기록

# 그래프를 그린다










