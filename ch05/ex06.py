import numpy as np

np.random.seed(103)
X = np.random.randint(10, size = (2,3))
print('X =', X) #X:(2,3)
W = np.random.randint(10, size = (3,5))
print('W =', W) # W: (3,5)

# forward propagation
Z = X.dot(W) #Z: (2,5)
print('z =', Z)

# backward propagation
# delta는 난수로 만들어서 보내준다 (원래는 누군가? 만들어주는 값)
delta = np.random.randn(2,5) # Z의 모양과 같아야한다
            # randn은 튜플을 주는 것이 아니다. param 선언이: (d0, d1, d2...) -> 그래서 dimension을 직접 준다
            # 튜플을 묶는 경우: size가 파라미터에 선언
print('delta =', delta)
# 이 델타를 x와 W방향으로 역전파 시킨다 (오차역전파)

# X방향으로 가는 역전파: dX
dX = delta.dot(W.T) # (2,5) @ (3,5).T => (2,5)@(5,3) 행렬의 dot product = (2,3) -> X의 모양
print('dX =', dX)
# x11이 1이 바뀌면, 전체 오차는 12.672.. 만큼 바뀐다
# x12이 1 바뀌면, 전체 오차는 11.65.. 만큼 바뀐다
# 오차가 바뀌는 기울기가 나온다 => 반대 방향으로 오차를 바꾸면 gradient descent를 한다

# W방향으로 가는 역전파: dW
dW = X.T.dot(delta) #(2,3)T @ (2,5) => (3,2)@(2,5) = (3,5) = W.shape
print('dW =', dW)
# W11이 1이 바뀌면, 전체 오차는 -10.71.. 만큼 바뀐다
