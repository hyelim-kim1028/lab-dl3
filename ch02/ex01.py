"""
Perceptron(퍼셉트론): 다수의 신호를 입력받아서, 하나의 신호를 출력
# 인풋은 multiple이고, output은 한개인 (전기회로의 개념이론을 컴퓨터 소프트웨어에 적용한 것)
AND: 두 입력이 모두 1일 때 출력이 1, 그 외에는 0
NAND: AND 출력의 반대(not AND)
OR : 두 입력 중 적어도 하나가 1이면 출력이 1, 그 외에는 0
XOR: 배타적OR/ exclusiveOR/  두 입력 중 하나는 1, 다른 하나는 0일때문 1을 출력, 그 이외에는 0

"""

def and_gate(x1, x2):
    w1, w2 = 1, 1 # 가중치
    # bias 결정: how we came up with b. write
    b = -1
    y = x1 * w1 + x2 * w2 + b
    # x1과 x2는 0 혹은 1이다 라는 가정 아래에서 하는 계산
    if y > 0:
        return 1
    else:
        return 0

    # 데이터를 가지고 컴퓨터를 학습시킨 후, 신경망이 w1, w2, b 를 직접 만드는 것이 목표

# my solution
# def nand_gate(x1, x2):
#     w1, w2 = 1, 1 #가중치 (weight)
#     b = 0 # 편향(bias)
#     y = x1 * w1 + x2 * w2 + b
#     if x1 == 1 and x2 == 1:
#         return 0
#     else:
#         return 1

# teacher's method
# def nand_gate(x1, x2):
#     if and_gate(x1,x2) > 0:
#         return 0
#     else:
#         return 1

# teacher's method 2
def nand_gate(x1, x2):
    w1, w2 = 0.5, 0.5 #가중치 (weight)
    b = 0 # 편향(bias)
    y = x1 * w1 + x2 * w2 + b
    if y < 1:
        return 1
    else:
        return 0

# def or_gate(x1, x2):
#     w1, w2 = 1, 1
#     b = 0
#     y = x1 * w1 + x2 * w2 + b
#     if x1 == 1 or x2 == 1:
#         return 1
#     else:
#         return 0

# teacher's solution
def or_gate(x1, x2):
    w1, w2 = 0.5, 0.5
    b = 0.5
    y = x1 * w1 + x2 * w2 + b
    if y >= 1:
        return 1
    else:
        return 0

#  my_solution
# def xor_gate(x1, x2):
#     w1, w2 = 1, 1
#     b = 0
#     y = x1 * w1 + x2 * w2 + b
#     if x1 == x2:
#         return 0
#     else:
#         return 1

# XOR_gate
def xor_gate(x1, x2):
    """XOR (Exclusive OR: 배타적 OR)
       선형 관계식 ( y = x1 * w1 + x2 * w2 + b) 하나만 이용해서는 만들 수 없음.
       NAND, OR, AND를 조합해야 가능 """
    z1 = nand_gate(x1, x2) # NAND의 결과
    z2 = or_gate(x1, x2) # OR의 결과
    return and_gate(x1, x2) # forward propagation(순방향 전파)


if __name__ == '__main__':
    # main 하고 enter를 누르면 자동 완성 ^^
    print('========= AND GATE =========')

    for x1 in (0,1):
        for x2 in (0, 1):
            print(f'AND({x1}, {x2}) -> {and_gate(x1, x2)}')

    print('========= NAND GATE =========')

    for x1 in (0, 1):
        for x2 in (0, 1):
            print(f'NAND ({x1}, {x2}) -> {nand_gate(x1, x2)}')

    print('========= OR GATE =========')

    for x1 in (0, 1):
        for x2 in (0, 1):
            print(f'OR ({x1}, {x2}) -> {or_gate(x1, x2)}')

    print('========= OR GATE =========')

    for x1 in (0,1):
        for x2 in (0, 1):
            print(f'OR ({x1}, {x2}) -> {xor_gate(x1, x2)}')