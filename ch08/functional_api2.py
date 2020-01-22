"""
GoogleNet p.272 그림 8-11
ResNet p.272 그림 8-12 (residual: 잔차 net: network)
"""

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Add, concatenate

# GoogleNet
# Input Tensor (입력 텐서)
input_tensor = Input(shape=(784,))
x1 = Dense(64, activation = 'relu')(input_tensor)
x2 = Dense(64, activation = 'relu')(input_tensor)
# x1와 x2는 서로 interaction이 없는 레이어
concat = concatenate(([x1,x2])) # 두개의 output 텐서를 연결
# 연결된 텐서를 그 다음 계층으로 전달
x = Dense(32, activation = 'relu')(concat)
# 출력층 생성
output_tensor = Dense(10, activation = 'softmax')(x)

#모델 생성
model = Model(input_tensor, output_tensor)
model.summary()
# ignore the warnings?

print('==================================================================')

#ResidualNet
# 일종의 지름길을 내는 것
# convolution을 연결할 때 (some possible problems: gradient vanishing)
# input(784) -> Dense(32)- relu--> * Dense(32) -relu- -> Dense(32) -relu- -> Dense(10) -softmax-
# 필터를 통과할 수록 이미지의 크기가 작아진다
# ResidualNet은 * 와 같은 지점에서 건너 띄어서 바로 출력층으로 가는 것?
# input(784) -> Dense(32)- relu - + add -> Dense(10) -softmax-
# 지름길로 가버리는 것
# + add 가 일어난다
# + 와 concatenate을 구별해야한다
# + : element, element끼리 더하기 -> 32가 입력으로 들어온다 (더하기의 개념) -> 32개와 32개를 더해줌 i.e. ((1,2),(3,4)) + ((5,6),(7,8)) = ((6, 8), (10, 12))
# concatenate: 더해주는 것 -> 64개가 입력으로 들어오는 것 i.e. ((1,2),(3,4)) + ((5,6),(7,8)) = ((1,2),(3,4),(5,6),(7,8))

input_tensor = Input(shape=(784,))
sc = Dense(32, activation = 'relu')(input_tensor) # 첫번째 Dense를 지나서 나온 출력값을 sc라고 부름
x = Dense(32, activation = 'relu')(sc)
x = Dense(32, activation = 'relu')(x)
x = Add()([x, sc]) # concatenate은 함수라서 () 에 값을 줬지만, Add는 클래스이름이라 먼저 호출 하고 뒤에 붙여줌
output_tensor = Dense(10, 'softmax')(x)
# 개념적으로 구현해 보았다

# Model 생성
model = Model(input_tensor, output_tensor)
model.summary()
# n_input * n_output + n_b
# 784 * 32 + 32 = 25120
# 32 * 32 + 32 = 1056

# 신경망이 깊어지면 깊어질 수록 기울기가 소실된다는 문제가 있었다
# 지름길을 통과하는 네트워크를 하나를 만들어서 기울기 손실이 없는 네트워크 한개를 끝까지 보내면 -> 마지막 단계에서 소실되어도 모든 망을 통과한 값 + 소실되지 않은 값을 더해줘서 계산해주는

