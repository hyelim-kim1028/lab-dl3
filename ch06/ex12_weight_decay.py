"""
과접합(overfitting): 모델이 학습 데이터는 정확하게 예측하지만, 학습되지 않은 데이터에 대해서는 예측/정확도가 떨어지는 현상을 과적합이라고 한다
    *** overfitting이 나타나는 경우
        - 학습 데이터가 적은 경우
        - 파라미터가 너무 많아서 표현력 (representational power)이 너무 높은 모델
        - The more complex a formula, the more fitted it is to the data. Here, when enters a new data, error rate goes high
        - sometimes a simple formula works betters with new data

     *** overfitting이 되지 않도록 학습
     1) regularization (정칙화, 정규화...): L1, L2-regularization
        - 손실(비용) 함수에 L1 규제 (W) 또는 L2 규제(W**2)을 더해줘서,
          파라미터 (W, b)를 갱신(업데이트)할 때, 파라미터가 더 큰 감소를 하도록 만드는 것.
        - L + (1/2) * lambda * ||W|| ** 2
        - Loss = the final result of ANN process
            - CE, 오차제곱의 평균 등등 여러가지 방법으로 손실을 계산할 수 있다
        - regularization: 일부러 손실을 키워서 back propagation을 할 때 가중치를 더 많이 감소시키는 방법
        - 손실에다 어떤 값을 더 더해주는가? 더해주는 값에 따라 L1, L2 regularization이라고 부른다 (overfitting이 줄어든다)
        - L1 : L + lambda||W||, L2: L + (1/2) * lambda * ||W||**2  # 가중치 벡터에 제곱을 곱한다. #보통 L2 규제를 더 많이 사용한다.
            -> W = W - lr * (dL/dW + lambda * W)
            -> 파라미터가 더 큰 값이 더 큰 감소를 일으킴
                 # 가중치가 큰 값은 큰 값을 빠지게 되고, 가중치가 적은 값은 적은 값이 빠지게 되니까 => 가중치가 높은 원소에게 더 큰 벌을 준다
            - 가중치를 변경하기 위해서 원래 가중치 W - lr * dL/dW 를 하려는 것
            -> W = W - lr * (dL/dW + lambda)
            -> 모든 파라미터가 일정한 크기로 감소됨 => 모든항에 똑같은 값을 빼준다 (모두에게 똑같은 벌칙을 준다) => L1 규제
    2) dropout: 학습 중에 은닉층의 뉴런을 랜덤하게 골라서 삭제하고 학습시키는 방법.
                - 테스트 할 때는 모든 뉴런을 사용함

- overfitting을 줄이는 전략은 학습 시의 정확도를 일부러 줄이는 것임
    -> 학습 데이터의 정확도와 테스트 데이터의 정확도 차이를 줄임

- 학습이란 여러번 epoch를 반복하는 것
"""

# 데이터 준비
from ch06.ex02_sgd import Sgd
from common.multi_layer_net import MultiLayerNet
from dataset.mnist import load_mnist
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(110)

(X_train, Y_train), (X_test,Y_test) = load_mnist(one_hot_label=True)

# 신경망 생성
wd_rate = 0.15
neural_net = MultiLayerNet(input_size= 784,
                           hidden_size_list=[100, 100, 100, 100, 100],
                           output_size= 10,
                           weight_decay_lambda= wd_rate) #가중치 감소를 사용하지 않는다

# weigth_decay_lambda: 가중치 감소에 사용할 상수 값

# 학습 데이터 개수를 300개로 제한 -> overfitting 을 만들기 위해서
X_train = X_train[:300]
Y_train = Y_train[:300]
X_test = X_test[:300]
Y_test = Y_test[:300]

train_size = X_test.shape[0]
epochs = 200  # 1 에포크: 모든 학습 데이터가 1번씩 학습된 겨우
mini_batch_size = 100 # 1번 forward에 보낼 데이터 샘플 개수
# 미니 배치 사이즈를 100이라고 할 때, 100개씩 3번 보내면 -> 그걸 200번 반복하는 것을 3번 = 600번 반복
iter_per_epoch = int(max(train_size/mini_batch_size, 1)) # 3 (300/100)
train_accuracies = []    # 학습하면서 정확도를 각 에프크마다 기록  # 정확도는 200번을 기록하고, 그 정확도를 그래프로 그린다
test_accuracies = []

optimizer = Sgd(learning_rate = 0.01)


# epochs = 0
for epoch in range(epochs):
    for i in range(iter_per_epoch):
        x_batch = X_train[(i * mini_batch_size): ((i + 1) * mini_batch_size)]
        y_batch = Y_train[(i * mini_batch_size): ((i + 1) * mini_batch_size)]
        gradients = neural_net.gradient(x_batch, y_batch)
        optimizer.update(neural_net.params, gradients)

    # 각 미니배치를 돌고 나온 값의 accuracy를 구한다
    train_acc = neural_net.accuracy(X_train, Y_train)
    train_accuracies.append(train_acc)
    test_acc = neural_net.accuracy(X_test,Y_test)
    test_accuracies.append(test_acc)
    print(f'epoch #{epoch}: train = {train_acc}, test= {test_acc}')


# lambda값을 바꿔서 또 다시 반복해보고 정확도가 얼마나 차이나는지 확인해본다
# 테스트 데이터로 정확도를 측정하기!
x = np.arange(epochs)
plt.plot(x, train_accuracies, label = 'train accuracy')
plt.plot(x, test_accuracies, label = 'test accuracy')
plt.legend()
plt.title(f'Weight Decay (lambda = {wd_rate})')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()




















