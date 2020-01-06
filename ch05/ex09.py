"""
Affine, ReLU, softmaxwithloss 클래스들을 이용한 신경망 구현
"""

import numpy as np

from ch05.ex05_ReLU import Relu
from ch05.ex07 import Affine
from ch05.ex08_softmax_loss import SoftmaxWithLoss

np.random.seed(106)

#입력 데이터: (1,2) Shape의 ndarray
X = np.random.rand(2).reshape((1,2))
print('X =', X)
# 실제 레이블(정답): 3개짜리 분류
Y_true = np.array([1, 0, 0])
print('Y =', Y_true)

# 첫번째 은닉층(hidden layer)에서 사용할 가중치/ 편향 행렬
# 첫번째 은닉층의 뉴런 개수 = 3개
# W1: (2,3), b1.shape: (3,)
W1 = np.random.randn(2,3)
b1 = np.random.rand(3)
print('W1 =', W1)
print('b1 =', b1)

# Affine 클래스에서 W1과 b1 합치기
affine1 = Affine(W1, b1)
# print('affine1 =',affine1) # only shows its direction
relu = Relu()
# print('affine_relu =', relu) # only shows its direction

# 출력층의 뉴런 갯수: 3개
# W shape: (3, 3), b.shape: (3,)
W2 = np.random.randn(3,3)
b2 = np.random.rand(3)
print('W2 =', W2)
print('b2 =', b2)

affine2 = Affine(W2, b2)
# 가장 마지막 레이어:
last_layer = SoftmaxWithLoss()

# forward의 반대를 타고가다보면 gradient를 구할 수 있다
# 마지막 layer의 뉴런의 갯수는 무엇을 분류하는 것이냐에 따라 다르다 (i.e. 숫자 이미지 = 10개, 개과 고양이 분류 = 2개, 등등)

# 각 레이어들을 연결: forward propagation

# Affine -> relu
Y = affine1.forward(X)
print('Y_shape = ',Y.shape)

Y = relu.forward(Y)
# 왜 Y는 넣어준걸까? affine1의 출력 값 -> Y!
# Y를 relu에 넣어줘야함 -> 변수를 공유해서 쓴다
print('Y shape =', Y.shape)

Y = affine2.forward(Y)
# print('Y_shape:'. Y_shape)

# Softmaxwith Loss
loss = last_layer.forward(Y,Y_true)
# 1,0, 0 엄만느 이 값ㅇ르 넣ㅇ므면 딘다
print('loss =', loss)  #cross entorpy -> 실제값 뿐만 아니라, 예측값도 있어야함. (저장 ~~)
# loss 가 여기에서는 last layer이라고 하는 것은 softaxiwiththeloss과 ㅏㄱㅌ다

print('y_pred =', last_layer.y_pred)
# lst_laeyr =
#oftmax (y의 예측값이 어떻게 되느냐를 분석한 것)
# 실제값과 비교해보면 나라와 많이 빗나갔다
# 벡 propagaion을 사용한다해서, ...

# gradient를 계산하기 위해서 역전파 (backward propagation)
learning_rate = 0.1
dout = last_layer.backward(1)
print('dout 1 =', dout) # 3개가 출력되었다 (원소 3개짜리)

# 여기서 부터 에러
dout = affine2.backward(dout) # 첫번째 변수에서 넘어온 dout을 값으로 줌
print('dout 2 =', dout)
print('dW2 =', affine2.dW)
print('db2 =', affine2.db)

dout = relu.backward(dout)
print('dout 3 =', dout)

dout = affine1.backward(dout)
print('dout 4 =', dout)
print('dW1 =', affine1.dW)
print('db1 =', affine1.db)

# 가중치/편향 행렬의 학습률과 gradient를 이용해서 수정
W1 -= learning_rate * affine1.dW
b1 -= learning_rate * affine1.db
W2 -= learning_rate * affine2.dW
b2 -= learning_rate * affine2.db


# 수정된 가중치/편향 행렬들을 이용해서 다시 forward propagation
Y = affine1.forward(X)
Y = relu.forward(Y)
Y = affine2.forward(Y)
Y = last_layer.forward(Y, Y_true)
print('loss =', Y) #1.217
print('', last_layer.y_pred) #[0.29602246, 0.25014373, 0.45383381]
                             # 실제값은 [1,0,0] 이므로, 첫번째 원소의 값은 커지고 나머지 두개의 값은 작아지는 방향으로 변화해햐한다

# 미니베치 (mini-batch)
X = np.random.rand(3, 2)
Y_true = np.identity(3)
print('Y_true_identity =', Y_true)
# 컬럼의 갯수는 상관 없다
# 미니베치가 가더라도 동작하는 방식은 같다
# 이 데이터를 가지고 forward, backward -> W/b가 수정이 됨 -> 변경된 값을 가지고 forward, backward 해보기

mini = affine1.forward(X)
mini = relu.forward(mini)
mini = affine2.forward(mini)
mini = last_layer.forward(mini, Y_true)
print('loss', mini)









