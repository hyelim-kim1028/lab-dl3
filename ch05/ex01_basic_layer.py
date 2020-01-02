"""
ex01_basic_layer
-
"""

class MultiplyLayer:
    def __init__(self):
        # forward 메소드가 호출될 때 전달되는 입력값을 저장하기 위한 변수
        # Below/stored values/ to be used when backward method will be revoked
        self.x = None
        self.y = None



    def forward(self, x, y):
        """ 입력값이 들어오면 연산을 해주는 메소드
            - 두개가 들어가서 한개가 아웃풋으로 나온다
            """
        self.x = x
        self.y = y
        return x * y
        # x,y를 저장하고 있는 공간이 없다
        # 단지 리턴만 아니라 저장할 수 있도록 __init__을 변경해준다 (anadir self.x and self.y)


    def backward(self, delta_out):
        """ 아웃풋을 입력값 방향으로 반대로 연산해주는 메소드
          - 한개의 인풋이 있으면, 두개의 아웃풋을 리턴
          - from tha backward propagation => 알파라는 값이 아웃풋 레이어 쪽으로 들어오면
          - a/ax (xy) => 알파 * a/ax = 알파y
          - a/ay (xy) => 알파 * a/ay = 알파x
          - 알파 = delta_out
          """
        dx = delta_out * self.y
        dy = delta_out * self.x
        return dx, dy

class AddLayer():
    def __init__(self):
        pass

    def forward(self, x, y):
        # self.x = x
        # self.y = y
        # x와 y값의 저장 할 때: backward할 때 이 값들이 사용된다면 저장하고, 아니라면 저장하지 않아도 된다.
        # 고로, 여기서는 사용되지 않으므로, 저장하지 않아도 된다
        return x + y

    def backward(self, dout):
        dx, dy = dout, dout
        return dx, dy


if __name__ == '__main__':

    # MultiplyLayer 객체 생성
    apple_layer = MultiplyLayer()

    # 순방향 전파 example
    apple = 100 #사과 1개의 갯수: 100원
    n = 2 # 사과 개수 1개
    apple_price = apple_layer.forward(apple, n) #순방향 전파
    print('apple_price =',apple_price)

    # Aunque este processo se usa la multiplicacion (la misma que del arriba), necessitamos que crear otro 객체
    # tax_layer를 MultiplyLayer 객체로 생성  # 왜 다른 객체를 생성하는가
    # tax = 1.1 설정해서 사과 2개 구매할 때 총 금액을 계산
    tax_layer = MultiplyLayer()
    tax = 1.1
    total_price = tax_layer.forward(apple_price, tax)
    print('total_price =',total_price)

    # x의 크기가 1 만큼 바뀌었을 때의 변화율을 계산 => back propagation
    # 만든 이유: 다음과 같은 것을 계산해보기 위해서
    # f = a * n * t 라고 할 때,
    delta = 1.0
    # 1) tax가 1 증가하면 전체 가격을 얼마가 증가? -> df/dt
    dprice, dtax = tax_layer.backward(delta) # 변수이름을 2개 선언하는 이유: there are two returned values
    print(f'dprice = {dprice}, dtax = {dtax}')
    # what we wanted to know: when the tax is increased by 1, total price is increased by dtax = 200.
    # df/dt: tax 변화율에 대한 전체 가격 변화율

    # 2) 사과 가격이 1 증가하면, 전체 가격은 얼마가 증가? -> df/da
    # 3) 사과 갯수가 1 증가하면, 전체 가격을 얼마나 증가할까요? -> df/dn
    dapple, dn = apple_layer.backward(dprice)
    print(f'dapple = {dapple}, dn = {dn}')
    # dapple: how the total price change when the price of an apple changes (price of an apple incrased by 1)
    # dn: how the total price changes when the number of apples bought changes (number of apples increased by 1)
    # 전체가격의 변화율 accroding to tax, # of apples, price of an apple

    # my solution
    # backward_prop = tax_layer.backward(1)
    # backward_prop0 = apple_layer.backward(1)
    # print(f' df/dt = {backward_prop}, df/dn, df/da = {backward_prop0}')

    # (사과가격 + 귤가격) * tax
    # 우선, add 클래스를 생성한다
    # AddLayer 테스트
    add_layer = AddLayer()
    x = 100
    y = 200
    dout = 1.5
    f = add_layer.forward(x, y)
    dx, dy = add_layer.backward(dout)
    print(f'fruit_forward = {f}, dx = {dx}, dy = {dy}')

    




