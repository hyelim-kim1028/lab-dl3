import numpy as np

def and_gate(x):
    # x는 [0,0], [0,1], [1,0], [1,1] 중 하나인 numpy.ndarray 타입
    # w = [w1, w2]인 numpy.ndarray 가중치와 bias b를 찾음

    # my_solution
    # x = np.ndarray(([0,0], [0,1], [1,0], [1,1])) # 일단, ndarray 타입으로 x를 생성
    # w = np.random.randint(0,1,2) # w을 랜덤으로 생성
    # b = 1
    # y = x[0] * w + x[1] * w + b # 식
    # if w == 0: # if 로 b = 0 일 때와 b를 구해야할 때를 구함
    #     return b == 0
    # elif w1 != 0 or w2 != 0:
    #     return b = y - (x[0] * w1 + x[1] * w2)

    # teacher's solution
    w = np.array([1,1])
    b = 1
    test = x.dot(w) + b #np.sum(x*w) + b
    print('test =', test)
    if test > 2:
        return 1
    else:
        return 0

def nand_gate(x):
    w = np.array([1,1])
    b = 1
    test = x.dot(w) + b
    if test <= 2:
        return 1
    else:
        return 2

def or_gate(x):
    w = np.array([1,1])
    b = 1
    test = x.dot(w) + b
    if test >= 2:
        return 1
    else:
        return 0

if __name__ == '__main__':
    for x1 in (0,1):
        for x2 in (0, 1):
            x = np.array([x1, x2])
            result = perceptron(x)