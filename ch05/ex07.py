import numpy as np

class Affine:
    # 입력값들이 합쳐지는 부분에서 행렬의 연산으로 사용  # y = X@W + b
    # X는 1차원일 수도 있고, 미니배치인 경우에는 2차원이 될 수도 있다
    def __init__(self, W, b):
        # class를 초기화 시켜주는 def
        self.W = W #가중치/weight 행렬
        self.b = b #bias 행렬
        self.X = None #입력 행렬을 저장할 field (변수)
        self.dW = None # W gradient -> W = W - lr * dW (미니배치에서 값에 변화를 줄 때 사용)
        self.db = None # b행렬 gradient -> b = b - lr * db

    def forward(self, X):
        self.X = X
        out = X.dot(self.W) + self.b # X도 backward propagation을 할 때 필요하다 (W에서 역전파 =? X.T.dot(Z))
                                     # 그러므로, X는 계산만 할 뿐만 아니라 저장까지 해야한다
        return out

    def backward(self, dout):
        """
        # dout: 이전 계층에서 넘어온 오차
        # 마지막: X만 리턴해주면 된다 (중간 프로세스는 no skip 하지만 마지막 출력은 X의 변화값)
        """
        # b 행렬 방향으로 gradient
        self.db = np.sum(dout, axis = 0)
        # z행렬 방향으로의 gradient -> W 방향과 X방향으로의 gradient
        self.dW = self.X.T.dot(dout)
        #self. 이 들어가면 -> 저장하고 있을꺼임 *-*!! 이라는 뜻
        dX = dout.dot(self.W.T) #W를 계속 변화 시키려고 하는 것 -> W와 b의 변화율을 사용해서 하니까 => dw와 dx의 값을 저장하고 있어야한다
                                # GD를 사용해서 W, b를 fitting 시킬 때 사용하기 위해서
        return dX

    # 왜 db를 계산할 때에 sum()을 사용했을까?
    # Y(2,5) -> b(1,5) => 2개를 가진 아이가 두 갈래로 갈라져와서 => 합침
    # Y + b 는 원래 수학에서는 연산이 되지 않는다. 하지만, 넘파이라 가능
    # b(1,5) 를 I(2,1) 와 dot product을 시켜주면 =? I.dot(b) = (2,1) @ (1,5) = (2,5) => 이렇게 하면 b와 Y의 행과 열의 갯수가 맞아서 계산할 수 있는 배열이 된다
    


if __name__ == '__main__':
    np.random.seed(103)

    X = np.random.randint(10, size = (2,3)) #입력 행렬
    # print('X =', X)

    W = np.random.randint(10, size = (3,5)) # 가중치 행렬
    # print('W =', W)
    b = np.random.randint(10, size = 5) # bias 행렬
    # print('b =', b)

    # Forward transformation
    affine = Affine(W, b) #Affine 클래스의 객체 생성
    Y = affine.forward(X) #Affine의 출력값
    print('Y =', Y)

    dout = np.random.randn(2,5)
    dX = affine.backward(dout)
    print('dX =', dX)
    # dW와 db처럼 self에 있는 필드들을 외부에서 호출할 때에는 -> 위치한 객체에서 revoke 할 수 있다
    print('dW =', affine.dW) #(3,5) # dW와 db는 affine 클래스의 field 이므로, affine.field 형식으로 호출 할 수 있다
    print('db =', affine.db) #(1,5)
    # db1 = 1이 변화할때의 b값은 -1.19.. 만큼 변화하고, 변화 시킬 때 사용 ??























































