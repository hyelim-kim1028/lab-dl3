"""
MultiplyLayer() & AddLayer() 연습
f(x, y, z) = (x + y) * z
x = -2, y = 5, z = -4  or f( -2, 5, -4) 에서의 df/dx, df/dy, df/dz의 값을 ex01에서 구현한 MultiplyLayer와 AddLayer 클래스를 이용해서 구하세요

q = x + y라 하면, dq/dx = 1, dq/dy = 1
f = q * z 이므로, df/dq = z, df/dz = q
위의 결과를 이용하면
df/dx = (df/dq)(dq/dx) = z
df/dy = (df/dq)(dq/dy) = z

numerical_gradient 함수에서 계산된 결과와 비교
"""
# Imported packages(?)
from ch05.ex01_basic_layer import AddLayer, MultiplyLayer
import numpy as np

# numerical_gradient로 계산 # 계산비용이 크다
def f(x,y,z):
    return (x + y) * z

 # numerical gradient 함수 테스트 # 왜 밑에서는 안되지?
h = 1e-12
dx = (f(-2 + h, 5, -4) - f(-2,5,-4))/h # 미분의 정의가 특정 점에서 x값만 h만큼 늘리고 빼주는
print('df/dx',dx)

dy = (f(-2, 5 + h, -4) - f(-2,5,-4))/h
print('df/dy',dy)

dz = (f(-2, 5, -4 + h) - f(-2,5,-4))/h
print('df/dz',dz)

if __name__ == '__main__':
    # f(-2, 5, -4)
    # forward propagation for f(x,y,z)
    # (x + y)
    x, y, z = -2, 5, -4 #초기값 설정
    add_gate = AddLayer()
    q = add_gate.forward(x, y)
    print('q =', q)

    # (x + y) * z
    mul_gate = MultiplyLayer()
    f = mul_gate.forward(q, z)
    print('f =',f)

    # backward propagation for (x + y) * z
    delta = 1
    dq, dz = mul_gate.backward(delta)
    print(f'dq = {dq}, dz = {dz}')

    t0, t1 = add_gate.backward(dq)
    print(f't0 = {t0}, t1 = {t1}')

    # f가 손실함수가 된다 -> 그 손실함수가 최소화되는 x,y,z를 찾는 것 (미분)
    # 우리는 노드의 값 1개만 알고, 그것이 연결만 되면 된다 (모든 값을 모두 알 필요는 없다)
    # 인풋이 들어왔을 때 아웃풋을 무엇을 내보낼 것인가
    # 백워드로 무언가가 들어왔을 때, 어떤 미분값을 내보낼 것인가
    # 어떠한 값은 들어온다고 치고,,,
    # al/af: 손실함수를 f로 미분한 값 => final?
    # 이 아이를 오차역전파로 보낼 때 operation이 * 일 경우, x,y를 바꿔서 보내주고, 더하기 게이트는 같은 델타?알파? 값을 갖는다
    # 레이어의 갯수는 상관없음!
    # backward propagation은 결국 미분연쇄법칙을 만들어낸 것
    







