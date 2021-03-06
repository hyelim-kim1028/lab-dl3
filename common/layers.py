# coding: utf-8
import numpy as np
from common.functions import *
from common.util import im2col, col2im

# neural network 에서 사용할 각각의 계층을 모아둔

class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = sigmoid(x)
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx


class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        
        self.x = None
        self.original_x_shape = None
        # 가중치와 편향 매개변수의 미분
        self.dW = None
        self.db = None

    def forward(self, x):
        # 텐서 대응
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        
        dx = dx.reshape(*self.original_x_shape)  # 입력 데이터 모양 변경(텐서 대응)
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None # 손실함수
        self.y = None    # softmax의 출력
        self.t = None    # 정답 레이블(원-핫 인코딩 형태)
        
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size: # 정답 레이블이 원-핫 인코딩 형태일 때
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        
        return dx

# 비슷한 기능들만 모은 파일 -> 모듈 => set of modules classified in folders => 패키지

class Dropout:
    """
    http://arxiv.org/abs/1207.0580
    """
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
                    # train_flg: 학습중인가 아닌가를 나타내는 파라미터 # 학습단계인가 아닌가
                    # 학습중이다: gradient를 구하는 중이다 => 메소드들에서 gradient를 구할 때에는 해당 파라미터가 참이였고, accuracy를 구하는 메소드에서는 해당 파라미터가 거짓이였다.
                    # 학습중일 때에만 제거를 한다 -> 학습이 다 끝난 다음에 정확도나 그런것들을 계산할 때에는 모두 이어주는,,,

        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
                        # (100, 784) = 100 * 784
                        # not randn (0 중심에 있는 것들이 많이 뽑힌다) but rand = 균등분포
                        # 균등분포 -> 0부터 1사이의 숫자들이 균일하게 뽑힌다

                        # dropout_ratio -> 숫자가 크면 클수록 self.mask 에서 50%에 가까운 값들이 뽑힌다...?
                        # i.e. 0.5 라면 (1 - 0.5)의 값들이 뽑힌다,,,

            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio) # 들어온것을 bypass 하는 것,,, 그냥 보내겠다 / 안내보내겠다 하면 dropout과 같은 효과를 주는 것 -
                                                  # dropout_ratio = 0 이라면 아무것도 제거하지 않은 것 => x 그대로
                                                  # dropout_ratio = 0.8 이라면 -> 원래 값의 0.2만 내보낸다
                                                  # 곱할수도, 안 곱할수도 있다

    def backward(self, dout):
        return dout * self.mask # 활성화되어있으면 보내고, 아니면 보내지않겠다 (0을 보냄)

    # drop out이라고 하는 클래스는 보내고, 안 보내고를 결정한다

# 신경망을 조립할 때 사용하는 아이들은 모두 forward와 backward 메소드가 있어요


class BatchNormalization:
    """
    http://arxiv.org/abs/1502.03167
    """
    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None # 합성곱 계층은 4차원, 완전연결 계층은 2차원  

        # 시험할 때 사용할 평균과 분산
        self.running_mean = running_mean
        self.running_var = running_var  
        
        # backward 시에 사용할 중간 데이터
        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, x, train_flg=True):
        self.input_shape = x.shape
        if x.ndim != 2:
            N, C, H, W = x.shape
            # C: channel -> 흑백 -> binary/2 ; 컬러 -> 3 (RGB) + 투명도 -> 4
            # H, W: Height * Width
            x = x.reshape(N, -1)
            # 2차원으로의 flatten

        out = self.__forward(x, train_flg)
        
        return out.reshape(*self.input_shape) # 원래대로 되돌려서 리턴
            
    def __forward(self, x, train_flg):
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)
                        
        if train_flg:
            mu = x.mean(axis=0) # 변수/컬럼별로
            xc = x - mu # x -평균 -> 분자 계산
            var = np.mean(xc**2, axis=0)
            std = np.sqrt(var + 10e-7) # 표준편차가 numpy에 있는데 저자는 굳이 자기가 만들어서 사용하고 있다 ^^; # epsilon을 더해주기 위해서 -> 분모가 0이 될수도 있으니까
            xn = xc / std
            
            self.batch_size = x.shape[0]
            self.xc = xc # 나중에 back propagation에서 사용되기 위해서 저장
            self.xn = xn
            self.std = std
            self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1-self.momentum) * var            
        else:
            xc = x - self.running_mean
            xn = xc / ((np.sqrt(self.running_var + 10e-7)))
            
        out = self.gamma * xn + self.beta 
        return out

    def backward(self, dout):
        if dout.ndim != 2: # 2차원이 아니라면!
            N, C, H, W = dout.shape
            dout = dout.reshape(N, -1)

        dx = self.__backward(dout) # 미분 계산

        dx = dx.reshape(*self.input_shape)
        return dx

    def __backward(self, dout): # field에 저장하고 있던 아이들을 사용해서 계산해준다
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xn * dout, axis=0)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
                # xc: X - 평균 (분자부분)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size
        # 실제로 계산 그래프를 그려보고, 미분을 계산해봐야 모두 comprehensively understand 할 수 있다
        # 더하기 곱하기 나누기로 이루어져 있음

        self.dgamma = dgamma
        self.dbeta = dbeta
        # 궁극적으로 알고 싶은 것: 감마와 베타가 어떻게 변화하는가

        return dx


class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad
        
        # 중간 데이터（backward 시 사용）
        self.x = None   
        self.col = None
        self.col_W = None
        
        # 가중치와 편향 매개변수의 기울기
        self.dW = None
        self.db = None

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = 1 + int((H + 2*self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2*self.pad - FW) / self.stride)

        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T

        out = np.dot(col, col_W) + self.b
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0,2,3,1).reshape(-1, FN)
        # 선 transpose -> 후 reshape # forward와 반대로 function
        # transpose에서 축의 인덱스도 forward (0, 3, 1, 2)와 반대로 해준다  -> (0,2,3,1)
        # out에서 같은 모양으로 나온다

        self.db = np.sum(dout, axis=0) # 독립변수인 b, 그냥 숫자 한개로써(?) 더해주면 된다
        self.dW = np.dot(self.col.T, dout) #2차원 dot 연산
        # col: 4차원 데이터를 2차원으로 펼쳐준 것
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)
                    # 2차원을 다시 4차원으로 # x -> (convolution layer) -> Y
                    # X(input) and Y(output) must have the same dimensions

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)
        # col2im: 2d image to 4d

        return dx


class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad
        
        self.x = None
        self.arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h*self.pool_w)

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max

        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)
        
        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        # poolsize = window의 크기?
        # 큰 것이 작아져서 나갔기 때문에, 작은것이 큰것으로 들어 올 때 -> 빈 공간들을 0으로 채워준다
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,)) 
        
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        
        return dx # 들어온 값 그대로 아니면 0으로 리턴하겠다
