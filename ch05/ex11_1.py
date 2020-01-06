"""
test the two_layer neural network made in ex10_two_layer
"""

import numpy as np
from sklearn.utils import shuffle

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

    # epochs = 100 # 100번 반복

    batch_size = 100  # 한번에 학습시키는 입력 데이터 개수 -> X/ batch_size = number of sets where the mini_batch can be finished at once
    # what if we give 6001, we cannot make it into a for-loop because its iter_size is below 1
    # so we gave max(X_train.shape[0]/batch_size,1) so that the lines will still run even though it goes below 1
    learning_rate = 0.1
    iter_size = max(X_train.shape[0] / batch_size, 1)
    print(iter_size)

    # part I.
    # batch_size 개수만큼의 학습 데이터를 입력으로 해서 gradient 계산
    # 가중치/편향 행렬들을 수정
    # loss를 계산해서 출력
    # accuracy를 계산해서 출력

    epoch = 100
    for k in range(epoch):
        X_train, Y_train = shuffle(X_train, Y_train) # How can I call them
    # batch_size 개수만큼의 학습 데이터를 입력으로 해서 gradient 계산
        for i in range(batch_size):
            # forward - backward -> gradient계산 (함수만 불러오기) -> 가중치, 편향 변경
            # batch_size 개수만큼의 학습 데이터를 입력으로 해서 gradient 계산
            grad = neural_net.gradient(X_train, Y_train) # (X_train[:mini_batch +1], Y_train)
            # print(grad)

            # 가중치/편향 행렬들을 수정
            # grad 내가 계산? grad['W1']
            for key in ('W1', 'b1', 'W2', 'b2'):
                neural_net.params[key] -= learning_rate * grad[key]

     # loss() 계산
    loss1 = neural_net.loss(X_train, Y_train)

        # 첫번째 100개에 대해 gradient를 계산하고, 가중치와 편향을 변경, 두번째 100개에 대해 gradient를 계산하고 반복 -
        # 이 아이들이 다 끝났을 때 loss를 계산한다 (600번의 iteration이 끝나고 loss는 마지막에 1번 계산한다)
        # final = loss()

    # accuracy() 계산
    accuracy1 = neural_net.accuracy(X_train, Y_train)


    # 위의 steps를 100회 반복 (epoch = 100)
    # 반복 할 때 마다 학습 데이터 세트를 무작위로 섞는 shuffle 코드를 추가
    # np.random.shuffle
    # 각 epoch 마다 데이터를 테스트로 해서 accuracy를 계산
    # 100번의 epoch가 끝났을 때, epoch-loss, epoch-accuracy 그래프를 그림