"""
교재 p.160 그림 5-15의 빈칸 채우기
apple = 100원, n_a = 2개
orange = 150원, n_o = 3ro
tax = 1.1
라고 할 때, 전체 과일 구매 금액을 AddLayer와 MultiplyLayer를 사용해서 계산하세요
df/dapple, df/dn_a, df/dorange, df/dn_o, df/dtax 값들도 각각 계산하세요.
"""
from ch05.ex01_basic_layer import MultiplyLayer, AddLayer

# 초기값 설정:
apple, n_a, orange, n_o, tax = 100, 2, 150, 3, 1.1

# FORWARD PROPAGATION: ((apple * n_a) + (orange * n_o)) * tax
# (apple * n_a), (orange * n_o)
# MultiplyLayer 생성
mul_layer = MultiplyLayer()
mul_apple = mul_layer.forward(apple, n_a)
mul_orange = mul_layer.forward(orange, n_o)  # 새로운 뉴런을 생성했어야함

# (apple * n_a) + (orange * n_o)
add_layer = AddLayer()
add_ao = add_layer.forward(mul_apple, mul_orange)

# ((apple * n_a) + (orange * n_o)) * tax
mul_add_ao = mul_layer.forward(add_ao, tax)

# BACKWARD PROPAGATION
# mul_add_ao -> add_ao
ds3, dt = mul_layer.backward(1)
print('dse =', ds3)
print('dt =', dt)

#mul_ad -> add_ao
ds1, ds2 = add_layer.backward(ds3)
print('ds1 =', ds1)
print('ds2 =', ds2) # 뭐가 문제일까 #AttributeError: 'str' object has no attribute 'ds2'


# 여기서 잘못 한 것 같다
# ds1 -> dapple, dn_o
dapple, dn_a = mul_layer.backward(ds1)
print(f'dapple = {dapple}, dn_a = {dn_a}')

# ds2 ->  orange, dn_o
dorange, dn_o = mul_layer.backward(ds2)
print(f'orange = {dorange}, dn_o = {dn_o}')









































