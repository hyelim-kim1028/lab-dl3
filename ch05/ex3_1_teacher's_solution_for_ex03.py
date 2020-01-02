"""
The solution for ch05 ex03
"""
from ch05.ex01_basic_layer import MultiplyLayer, AddLayer

apple, n_a = 100, 2
orange, n_o = 150, 3
tax = 1.1


# 뉴런 생성 및 forward propagation
apple_mul = MultiplyLayer() # 뉴런 생성
apple_price = apple_mul.forward(apple, n_a) #forward propagation
print('apple_price =', apple_price)

orange_mul = MultiplyLayer()
orange_price = orange_mul.forward(orange, n_o) #forward propagation
print('orange_price =', orange_price)

add_gate = AddLayer()
price = add_gate.forward(apple_price, orange_price)
print('price =', price)

tax_multiplier = MultiplyLayer()
total_price = tax_multiplier.forward(price, tax)
print('total_price', total_price)


# Backward propagation
dprice, dtax = tax_multiplier.backward(1)
print('dprice =', dprice)
print('dtax =', dtax)

d_apple_price, d_orange_price = add_gate.backward(dprice)
print('d_apple_price', d_apple_price)
print('d_orange_price', d_orange_price)

d_apple, d_na = apple_mul.backward(d_apple_price)
print('d_apple =', d_apple) # df/d_apple 편미분 계산
print('d_na =', d_na) #df/d_na 편미분 계산
d_orange, d_no = orange_mul.backward(d_orange_price)
print('d_ornage =', d_orange) #df/d_orange 편미분 계산
print('d_no =', d_no) #df/d_no 편미분 계산








