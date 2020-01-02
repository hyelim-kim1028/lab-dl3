"""
숙제: sigmoid
# 활성화 함수 중 많이 사용되는 시그모이드 함수
# 시그모이드 함수의 정의: y = 1/ (1 + exp(-x))
    1) dy/dx = y(1-y) 를 증명
    2) 이걸을 이용해서 sigmoid 라는 클래스를 생성
        -> 시그모이드 뉴런작성 with forward & backward 메소드
    # 시그모이드의 입력값은 1개
"""
import math


class sigmoid_function:
    def __init__(self,x):
        x.self = None

    def f_sigmoid(self,x):
        x.self = x
        return 1/ (1+ math.exp(-x))

    def back_sigmoid(self, x):
        x.self = x
        y = 1/ (1+ math.exp(-x))
        return self.y(1- self.y)



if __name__ == '__main__':
    pass







