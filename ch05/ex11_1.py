"""
test the two_layer neural network made in ex10_two_layer
"""

import numpy as np
from sklearn.utils import shuffle
from ch05.ex09 import loss
from ch05.ex10_two_layer import TwolayerNetwork
from dataset.mnist import load_mnist
import matplotlib.pyplot as plt
import pickle

if __name__ == '__main__':
    np.random.seed(106)

    # load the MNINST data set
    (X_train, Y_train), (X_test, Y_test) = load_mnist(one_hot_label= True)

    # 2층 신경망 생성
    neural_net = TwolayerNetwork(input_size=784,
                                 hidden_size= 32, # we can adjust -> hyperparameter
                                 output_size= 10) # traindata

    # epochs = 100 # 100번 반복

    batch_size = 100  # 한번에 학습시키는 입력 데이터 개수 -> X/ batch_size = number of sets where the mini_batch can be finished at once
    # what if we give 6001, we cannot make it into a for-loop because its iter_size is below 1
    # so we gave max(X_train.shape[0]/batch_size,1) so that the lines will still run even though it goes below 1
    learning_rate = 0.1
    epochs = 50 # 학습 회수

    # iter_size =  한번의 epoch (학습이 한번 완료되는 주기)에 필요한 반복 횟수
    # -> 가중치/편향 행렬들이 변경되는 회수
    iter_size = max(X_train.shape[0] // batch_size, 1)
    print(iter_size) # W와 b가 변경되는 횟수
    # batch_size 가 적으면 가중치가 변경되는 횟수가 많아지고, batch_size가 크면 가중치가 변경되는 횟수가 줄어든다

    # part I.
    # batch_size 개수만큼의 학습 데이터를 입력으로 해서 gradient 계산
    # 가중치/편향 행렬들을 수정
    # loss를 계산해서 출력
    # accuracy를 계산해서 출력

    # my_solution
    # epoch = 100
    # for k in range(epoch):
    #     X_train, Y_train = shuffle(X_train, Y_train) # How can I call them
    # # batch_size 개수만큼의 학습 데이터를 입력으로 해서 gradient 계산
    #     for i in range(batch_size):
    #         # forward - backward -> gradient계산 (함수만 불러오기) -> 가중치, 편향 변경
    #         # batch_size 개수만큼의 학습 데이터를 입력으로 해서 gradient 계산
    #         grad = neural_net.gradient(X_train, Y_train) # (X_train[:mini_batch +1], Y_train)
    #         # print(grad)
    #
    #         # 가중치/편향 행렬들을 수정
    #         # grad 내가 계산? grad['W1']
    #         for key in ('W1', 'b1', 'W2', 'b2'):
    #             neural_net.params[key] -= learning_rate * grad[key]
    #
    #  # loss() 계산
    # loss1 = neural_net.loss(X_train, Y_train)
    #
    #     # 첫번째 100개에 대해 gradient를 계산하고, 가중치와 편향을 변경, 두번째 100개에 대해 gradient를 계산하고 반복 -
    #     # 이 아이들이 다 끝났을 때 loss를 계산한다 (600번의 iteration이 끝나고 loss는 마지막에 1번 계산한다)
    #     # final = loss()
    #
    # # accuracy() 계산
    # accuracy1 = neural_net.accuracy(X_train, Y_train)


    # 위의 steps를 100회 반복 (epoch = 100)
    # 반복 할 때 마다 학습 데이터 세트를 무작위로 섞는 shuffle 코드를 추가
    # np.random.shuffle
    # 각 epoch 마다 데이터를 테스트로 해서 accuracy를 계산
    # 100번의 epoch가 끝났을 때, epoch-loss, epoch-accuracy 그래프를 그림


    # teacher's solution

    train_losses = [] # 각 epoch 마다 학습 데이터의 손실을 저장할 리스트
    train_accuracy = [] #각 epoch 마다 학습 데이터의 정확도를 저장할 리스트
    test_accuracies = [] #각 epoch 마다 테스트 데이터의 정확도를 저장할 리스트

    for epoch in range(epochs):
    # 학습 데이터를 랜덤하게 섞음
        idx = np.arange(len(X_train)) # [0, 1, 2 .... , 59999] # 6만개~ 오호호
        np.random.shuffle(idx)
        # print(idx)
        # 인덱스만 100개씩 잘라서 트레인 데이터를 뽑아내고, 정답도 뽑아 내고,,,
        # X_train 자체를 섞어버리면 안된다 -> 정답을 똑같은 순서로 섞을 수 없기 때문에

        for i in range(iter_size):
        # batch_size 개수만큼의 학습 데이터를 입력으로 해서 gradient 계산
            X_batch = X_train[idx[i * batch_size: (i+1) * batch_size]]
            Y_batch = Y_train[idx[i * batch_size: (i+1) * batch_size]]
            gradients = neural_net.gradient(X_batch, Y_batch)
            # 가중치/편향 행렬들을 수정
            for key in neural_net.params:
                neural_net.params[key] -= learning_rate * gradients[key]
            # 위의 코드들 -> W/b가 600번 바뀜

        # loss를 계산해서 출력
        train_loss = neural_net.loss(X_train, Y_train)
        train_losses.append(train_loss)
        print('train_loss:', train_loss)
        # accuracy를 계산해서 출력
        # 각 에포크마다 테스트 데이터로 테스트해서 accuracy를 계산
        train_acc = neural_net.accuracy(X_train, Y_train)
        train_accuracy.append(train_acc)
        print('train_loss:', train_acc) # 학습을 딱 한번 밖에 안 시켰지만 90퍼센트에 가까운 정확도를 보여준다
    # learning rate를 0.01로 바꾸었더니 train_loss 가 0.39로 떨어졌다
    # batch_size = 1, learning rate = 0.01 로 바꾸었을 때는 시간은 오래 걸렸지만 0.94라는 높은 정확도를 보여준다
    # cost-efficient 한 값들을 찾아서 알고리즘을 만드는 것이 관건
        test_acc = neural_net.accuracy(X_test, Y_test)
        test_accuracies.append(test_acc)
        print('train_loss:', test_acc)
    # 100번의 epoch가 끝났을 때, epoch-loss, epoch-accuracy 그래프를 그림
    # 각 에포크마다 epoch-loss와 epoch-accuracy를 저장해두어야 그래프를 그릴 수 있다

    # Graph 그리기
    # epoch - loss 그래프
    x = range(epochs)
    plt.plot(x, train_losses)
    plt.title('Loss-Cross Entropy')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
    # 학습 데이터 가지고 한계있다
    # X -> 에포크 반복 횟수 Y -> CE => epoch가 반복 될수록 CE가 줄어드는 것을 보여줌

    # epoch ~ 학습/테스트데이터 accuracy 그래프
    x = range(epochs)
    plt.plot(x, train_accuracy, label = 'train accuracy')
    plt.plot(x, test_accuracies, label = 'test_accuracy')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Test/Train Accuracy')
    plt.show()
    # The lines coincide in epoch = 5, the train accuracy goes up and the test accuracy remains(?)
    # the graph shows over-fitting of a graph

    # 신경망에서 학습이 모두 끝난 후, 파라미터(가중치/ 편향 행렬들)들을 파일에 저장
    # pickle 이용
    with open('ex11_1.pickle', 'wb') as f:
        pickle.dump(neural_net.params, f)

#pickle: 파이썬 기본 패키지로 저장되어 있는 것 중 하나