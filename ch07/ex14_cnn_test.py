"""
SimpleConvNet (간단한 CNN)을 활용한 MNIST 손글씨 이미지 데이터 분류
"""
import numpy
from matplotlib import pyplot as plt

from ch07.simple_convnet import SimpleConvNet
from common.trainer import Trainer
from dataset.mnist import load_mnist

# 데이터 로드
(X_train, Y_train), (X_test, Y_test) = load_mnist(flatten=False)

# 테스트 시간을 줄이기 위해서 데이터 사이즈를 줄임.
X_train, Y_train = X_train[:5000], Y_train[:5000]
X_test, Y_test = X_test[:1000], Y_test[:1000]

# CNN 생성
cnn = SimpleConvNet()

# 테스트 도우미 클래스
trainer = Trainer(network=cnn,
                  x_train=X_train,
                  t_train=Y_train,
                  x_test=X_test,
                  t_test=Y_test,
                  epochs=20,
                  mini_batch_size=100,
                  optimizer='Adam',
                  optimizer_param={'lr': 0.01},
                  evaluate_sample_num_per_epoch=100)
# 테스트 실행
trainer.train()

# 학습이 끝난 후 파라미터들을 파일에 저장
cnn.save_params('cnn_params.pkl')

# 그래프(x축 - epoch, y축 - 정확도(accuracy))
x = numpy.arange(20)
plt.plot(x, trainer.train_acc_list, label='train accuracy')
# trainer파일에 train_acc_list = [] 의 empty list가 있어서 값들을 자동을 append 해준다
plt.plot(x, trainer.test_acc_list, label='test accuracy')
# 두개를 동시에 계산해주는 이유는 과적합(over-fit)이 됐는지 안됐는지 보려고
# 어느 정도 정체되서 1에 가지 안가는 구간이나 이상 구간이 있다면 과적합으로 볼 수 있다
plt.legend()
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()
# 거의 비슷한 성향을 보인다 -> 아마 과적합이 되지 않았거나, 약간의 오버핏 밖에 되지않음

# cnn_params.pkl이 저장되어 있다
# convolution이 가지고 있는 가중치가 어떤 모습으로 바뀌어있나 확인









