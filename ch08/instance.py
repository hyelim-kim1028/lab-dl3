"""
How memory works in Python
                    참조하기...
foo (@주소값 123)     ----- >    field/ member 변수
주소값을 저장하고 있는 값:         + init_val = 1
인스턴스/객체                     method
                                 + __init__()
- 인스턴스는 함수거나 메소드가 아니다, 그저 주소값일 뿐 (호출의 대상이 아니다)
boo (@456)          ----->      field
                                + init_val = 1
                                method
                                + init()
                                + call()

"""


class Foo:
    def __init__(self, init_val = 1):
        self.init_val = init_val

class Boo:
    def __init__(self, init_val = 1):
        print('__init__호출')
        self.init_val = init_val

    def __call__(self, n):
        print('__call__호출')
        self.init_val *= n
        return self.init_val


if __name__ == '__main__':
    # Foo 클래스의 인스턴스를 생성 - 생성자 호출
    foo = Foo()
    print('init_val =', foo.init_val)
    # foo(100) # 인스턴스 객체의 이름을 함수처럼 사용할 수 없다 (TypeError: 'Foo' object is not callable)

    #Boo 클래스의 인스턴스를 생성
    boo = Boo()
    print('init_val =', boo.init_val)
    boo(5) # 방금전에는 가능하지 않았지만, 이제는 가능함
    # 인스턴스 이름을 함수처럼 사용하면서 값을 주면, 그 인스턴스가 가지고 있는 메소드를 호출한다
    # Foo에는 메소드가 없었고, boo에는 메소드가 있었기 때문에 boo(5) 는 call을 호출했고
    print('boo.init_val =', boo.init_val)

    # 인스턴스 호출: 인스턴스 이름을 마치 함수 이름처럼 사용하는 것
    # 클래스에 정의된 __call__메소드를 호출하게 됨
    # 클래스에서 __call__을 작성하지 않은 경우에는 인스턴스 호출을 사용할 수 없음

    #callable: __call__ 메소드를 구현한 객체
    print('foo 호출 가능:', callable(foo)) #False
    print('boo 호출 가능:', callable(boo)) #True

    print()
    boo = Boo(2)
    x = boo(2)
    print('x =', x) #4
    x = boo(x)
    print(x) #16

    print()
    input = Boo(1)(5) #Boo(1) = 생성자 호출 #(5) 함수에 argument 호출
    # is the same as: boo = Boo(1) # init_val = 1
    #               input = boo(5)
    x = Boo(5)(input) # 5*5
    print('x =', x) # x = 25
    x = Boo(5)(x) # 5 * 25
    print('x =', x) # 125
