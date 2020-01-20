from common.util import im2col
import numpy as np 

class pooling:
    def __init__(self, fh, fw, stride = 1, pad =0):
        # in the text book fh is written as pool_h and so on
        self.fh = fh # pooling 윈도우의 높이(height)
        self.fw = fw # pooling 윈도우의 너비(width)
        self.stride = stride # pooling 윈도우의 이동시키는 보폭
        self.pad = pad # 패딩 크기
        # backward에서 사용하게 될 값
        self.x = None # pooling 레이어로 forward 되는 데이터
        self.arg_max = None # 찾은 최댓값의 인덱스

    def forward(self, x):
        """ 순방향 전파 메소드 
            x: (samples, channel, height, width) 모양의 4차원 배열     
        """
       # 1) 구현
       # 2) x --> im2col --> 2차원 변환
       # 3) 채널 별 최댓값을 찾을 수 있는 모양으로 x를 reshape
       # 4)
