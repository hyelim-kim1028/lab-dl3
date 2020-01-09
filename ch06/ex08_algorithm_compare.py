"""
파라미터 최적화 알고리즘 6개의 성능 비교
Compare the efficiency(?)/성능 of algorithms in packaage ch06
- 성능을 비교하는 지표: 손실(loss) or 정확도 (accuracy)
"""
import matplotlib.pyplot as plt
import numpy as np

from ch05.ex10_two_layer import TwolayerNetwork #class
from ch06.ex02_sgd import Sgd
from ch06.ex03_momentum import Momentum
from ch06.ex04_adagrad import AdaGrad
from ch06.ex05_ADAM import Adam
from ch06.ex06_rmsprop import RMSProp
from ch06.ex07_nesterov import Nesterov
from dataset.mnist import load_mnist



if __name__ == '__main__':
    #MNIST(손글씨 이미지) 데이터 로드
    (X_train, Y_train), (X_test, Y_test) = load_mnist(one_hot_label= True)
    # train_size = X_train.shape[0]     # 60000만개를 모두 트레인 시킬 것은 아니라 넣어주지 않음
    # X_train.shape = (60000, 784)

    #최적화 알고리즘을 구현한 클래스의 인스턴스들을 dict에 저장
    optimizers = dict()
    optimizers['SGD'] = Sgd() # key값으로 알고리즘 이름을 준다 ex. ['SGD']  = 그리고 다른 파일에서 만든 알고리즘을 import 해준다
    optimizers['Momentum'] = Momentum()
    optimizers['Adagrad'] = AdaGrad()
    optimizers['ADAM'] = Adam()
    optimizers['RMS'] = RMSProp()
    optimizers['Nesterov'] = Nesterov()


    # 은직층 1개, 출력층 1개로 이루어진 신경망을 optimizer 개수만큼 생성
    # 각 신경망에서 손실들을 저장할 dict를 생성
    train_losses = dict()
    neural_nets = dict()
    for key in optimizers:
        neural_nets[key] = TwolayerNetwork(input_size = 784,
                                          hidden_size = 32,
                                          output_size= 10)
        train_losses[key] = [] #loss들의 이력(history)를 저장 # 알고리즘별 손실들을 넣어준다

    # 각각의 신경망을 학습시키면서 loss를 계산하고 저장/기록
    # 6만개의 훈련 데이터 중에서 128개만 랜덤하게 뽑아서, gradient를 계산하고, 각 알고리즘 별로 파라미터를 업데이트 할 수 있다.
    # 업데이트된 파라미터를 가지고 손실을 계산하다. 전체 데이터를 가지고 손실을 계산할 수 있다

    # 상수 정의
    iterations = 2_000 # 총 합습 횟수
    batch_size = 128 # 한 번 학습에서 사용할 미니 배치 크기
    train_size = X_train.shape[0] #60_000 개
    np.random.seed(108)
    for i in range(iterations): #2000번 학습을 반복
        # 학습 데이터 (X_train), 학습 레이블(Y_train)에서 미니 배치 크기 만큼 random하게 데이터를 선택
        batch_mask = np.random.choice(train_size, batch_size)
        # 0 ~ 59,999 사이의 숫자들 (train_size) 중에서 128(batch_size)개의 숫자를 임의로 선택
        # mask 라는 것은 배열/리스트를 만들어서 숫자 128개를 랜덤으로 뽑은 것 -> 그것이 choice함수가 한 일
        # choice: generates a random sample of 1-D array

        # 미니 배치 데이터/레이블 선택
        X_batch = X_train[batch_mask] # X[[111,9933,25...]] -> X에서 111번째 행을 뽑고, 9933번째 행을 뽑고,,, 이런식으로 128개를 뽑는다
        Y_batch = Y_train[batch_mask]
        # 이렇게 하면 두 batch에서 같은 값의 정보 + 답을 뽑아 낼 수 있다
        # 이 배열들을 가지고 훈련을 시킨다 -> gradient

        #선택된 학습 데이터와 레이블을 사용해서 gradient들을 계산
        # network 에서 계산 # 신경망 6개 -> optimizer 갯수만큼 신경망이 있다
        for key in optimizers:
            # 각각의 최적화 알고리즘을 사용해서 gradients를 계산한다
            gradients = neural_nets[key].gradient(X_batch, Y_batch)
            # 각각의 최적화 알고리즘의 파라미터 업데이트 기능을 사용
            optimizers[key].update(neural_nets[key].params, gradients) # neural_nets의 파라미터들을 업데이트 해벌임 ^0^!
            # 미니 배치의 손실을 계산
            loss = neural_nets[key].loss(X_batch, Y_batch) #key: names of algorithms
            # loss의 히스토리를 기록해야한다 -> train_losses에 append
            train_losses[key].append(loss)
                    # loss 데이터의 구조는 {'SGD' : [ -> list ]} -> dictionary

            # 하나의 배치파일에 대해서 각 알고리즘별로 저장된다. 끝난 후에 위에서 다시 시작할 때에는 새로운 128개로 다시 시작한다
            # 128개의 원소들을 선택 -> 각 알고리즘 별로 gradient 계산 -> loss 계산 후 저장 -> 반복

        # 출력
        # 신경망이 100번 반복할 때마다/ 100번째 학습마다 계산된 손실을 출력
        if i % 100 == 0:
            print(f'===== Training #{i} =====')
            for key in optimizers:
                print(key, ':', train_losses[key][-1]) # train_losses 리스트의 마지막 원소
                # train losses에 100개씩 있을 테니까, 거기서 마지막 1개만 출력하는 것
        # 손실이 크면: 틀린게 많다, 손실은 작으면 작을 수록 좋다
        # Adagrad, Adam, RMS가 100번만 하고도 제일 많이 떨어짐 -> SGD의 성능이 가장 좋지 않음 -> 손실을 떨어뜨리는 속도가 현저히 느리다

    # 그래프로 그리기
    # 각각의 최적화 알고리즘 별 손실 그래프
    x = np.arange(iterations) # 그래프에 x좌표: 학습 횟수
    # y좌표: 각각의 손실들 (train_loss가 갖고 있는 손실들)
    for key, losses in train_losses.items():
        plt.plot(x, losses, label = key)
    plt.title('Losses')
    plt.xlabel('Number of Training')
    plt.ylabel('Loss')
    plt.legend()
    plt.show(   )

























