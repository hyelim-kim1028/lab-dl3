"""
test the two_layer neural network made in ex10_two_layer
"""

import numpy as np

from ch05.ex09 import loss
from ch05.ex10_two_layer import TwolayerNetwork
from dataset.mnist import load_mnist

if __name__ == '__main__':
    np.random.seed(106)

    # load the MNINST data set
    (X_train, Y_train), (X_test, Y_test) = load_mnist(one_hot_label= True)

    # 2층 신경망 생성
    neural_net = TwolayerNetwork(input_size=784,
                                 hidden_size= 32,
                                 output_size= 10) # traindata

    batch_size = 100
    iters_size = 600
    learning_rate = 0.1
    train_size = X_train.shape[0]
    iter_num = max(X_train.shape[0]/ batch_size, 1)
    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    for i in range(iters_size):
        batch_mask = np.random.choice(train_size, batch_size)
                            # choice, when train_size(n) = batch_size(n), shuffle randomly. However, when they have different sizes (train size > batch size),
                            # the function selects batch_size(m) randomly from train_size(n)
        x_batch = X_train[batch_mask]
        y_batch = Y_train[batch_mask]

    # 기울기 계산
        grad = neural_net.gradient(x_batch, y_batch)

    # 매게변수 갱신
        for key in ('W1', 'b1', 'W2', 'b2'):
            neural_net.params[key] -= learning_rate * grad[key]

    # 학습 경과 기록
        loss = neural_net.loss(x_batch, y_batch)
        train_loss_list.append(loss)

    if i % iter_num == 0:
        train_acc = neural_net.accuracy(X_train, Y_train)
        test_acc = neural_net.accuracy(X_test, Y_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(train_acc, test_acc)







