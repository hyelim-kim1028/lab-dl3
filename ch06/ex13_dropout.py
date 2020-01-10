"""
drop out
"""

import numpy as np

from ch06.ex02_sgd import Sgd
from common.multi_layer_net_extend import MultiLayerNetExtend
from dataset.mnist import load_mnist
import matplotlib.pyplot as plt

np.random.seed(110)
x = np.random.rand(20) # 0.0 ~ 0.999999... 균등 분포에서 뽑은 난수
print(x)
mask = x > 0.5
print(mask)
print(x * mask) # false = 0, false * x = 0; true = 1, returns its original value

np.random.seed(110)

# 데이터 준비
(X_train, Y_train), (X_test, Y_test) = load_mnist(one_hot_label=True)

# 신경망 생성
dropout_ratio = 0.1
neural_net = MultiLayerNetExtend(input_size = 784,
                                 hidden_size_list= [100, 100, 100, 100, 100],
                                 output_size= 10,
                                 use_dropout= True,
                                 dropout_ration= dropout_ratio)


X_train = X_train[:500]
Y_train = Y_train[:500]
X_test = X_test[:500]
Y_test = Y_test[:500]

# Initial Idea
# # drop out 구현 (Affine > Batch > ReLU > Dropout 이니까 Relu값을 가지고 dropout을 구현한다)
#

train_size = X_test.shape[0]
epochs = 200
mini_batch_size = 100
iter_per_epoch = int(max(train_size/mini_batch_size, 1))

optimizer = Sgd(learning_rate = 0.1)

train_accuracies = []
test_accuracies = []

for epoch in range(epochs):
    indices = np.arange(train_size)
    np.random.shuffle(indices)
    for i in range(iter_per_epoch):
        x_batch = X_train[(i * mini_batch_size): ((i+1) * mini_batch_size)]
        y_batch = Y_train[(i * mini_batch_size): ((i + 1) * mini_batch_size)]
        gradients = neural_net.gradient(x_batch, y_batch)
        optimizer.update(neural_net.params, gradients)

    train_acc = neural_net.accuracy(X_train, Y_train)
    train_accuracies.append(train_acc)
    test_acc = neural_net.accuracy(X_test, Y_test)
    test_accuracies.append(test_acc)
    print(f'epoch #{epoch}: train = {train_acc}, test= {test_acc}')

# 그래프 그리기
x = np.arange(epochs)
plt.plot(x, train_accuracies, label='Train')
plt.plot(x, test_accuracies, label='Test')
plt.legend()
plt.title(f'Dropout (ratio={dropout_ratio})')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()


# 학습률을 낮추면서, 학습데이터와 테스트데이터 사이의 정확도의 차이를 줄임









