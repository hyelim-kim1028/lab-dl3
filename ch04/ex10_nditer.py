"""
numpy.nditer 객체: 반복문(for, while)을 쉽게 쓰게 도와주는 객체
"""
import numpy as np

# 반복문 2개를 1개로 줄일 수 있어서, 코드를 줄여줄 수 있다
# 코드쓰기가 쉽지는 않음,,,^^

np.random.seed(1231)
a = np.random.randint(100, size = (2,3)) #(0~100미만, 행렬의 모양(2X3))
print(a)

# a 출력의 결과를 한줄로 써주기 => for or while loop
# 40 21 5 52 84 39 # 반복문을 2번 써주어야한다

for row in a:
    for x in row:
        print(x, end = ' ') # end는 \n 이 기본값
print()

# while 문으로도 가능하다
# while row in a:
#     while x in row:
#         print(x, end = ' ')
# print()

i = 0
while i < a.shape[0]: # nrow  # i는 2보다 작을 때 까지
    j = 0
    while j < a.shape[1]: #ncol # j는 3보다 작을 때 까지
        print(a[i,j], end = ' ')
        j += 1
    i += 1
print()

# 하지만 iterator를 사용하면 for/while 문을 1개만 사용할 수 있다

with np.nditer(a) as iterator: # nditer클래스 객체 생성
    for val in iterator: # for-in 구문에서의 iterator는 배열에서 알아서 꺼내줌
        print(val, end = ' ')
print()

# 보통 컬럼 부터 -> 행을 늘려줌 (c_index => c언어에서 사용하는 인덱스 순서 / f_index => fort라는 언어에서 사용하는 인덱스 순서)
# 이터레이터도 기본값을 이 순서로 감

# for-loop(인덱스 가져올 수 없으니까, 값만 가지고 비교해야한다)과 while-loop(길고, 복잡해지지만 인덱스로 특정 원소로 접근할 수 있다-> slicing 가능)은 장단점이 있다
# for-loop 에서 인덱스를 사용하려고 enumerate을 사용했었다

with np.nditer(a, flags = ['multi_index']) as iterator: # multi_index: a가 2차원이니까
    while not iterator.finished: # iterator 의 반복이 끝나지 않았으면
                                 # finished = True 가 되면, not 이 충족되지 못하니까 loop이 끝난다
        i = iterator.multi_index #iterator 에서 멀티 인덱스를 찾아 주세요
        print(f'{i}: {a[i]}', end=' ')
        iterator.iternext() # c_index 순서대로 뽑아 주었다 (iterator 가 (두개으 쌍인) 인덱스를 줌)
print() # 값 변경이 가능하다

# python의  for - in 구문같은 것: iterator가 내부적으로 있다...?! 호호이ㅣㅣ

with np.nditer(a, flags=['c_index']) as iterator:
    while not iterator.finished:
        i = iterator.index
        print(f'{i}:{iterator[0]}', end = ' ')
        iterator.iternext() # 2차원 배열이 1차원으로 flatten 되면서 0부터 차례대로 인덱스를 붙쳐버린다
print() #값 변경이 불가능하다

a = np.arange(6).reshape((2,3))
print('a =',a)

with np.nditer(a, flags = ['multi_index']) as it:
    while not it.finished:
        a[it.multi_index] *= 2
        it.iternext()
print('after a =',a)

a = np.arange(6).reshape((2,3))
with np.nditer(a, flags = ['c_index'], op_flags = ['readwrite']) as it:
    while not it.finished:
        it[0] *= 2 #error: output array is read-only -> we put: op_flags = ['readwrite']
        it.iternext()
print('after readwrite =',a)
# op_flag = operation flag (operation pertains to write, read, etc)

# iterator 는 1차원이든 2차원이든 같은 방식을 사용한다






























