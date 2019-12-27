"""
만개에서 짤라 쓰는데 왜 에러가 안나죠?
"""
import numpy as np

a = np.arange(10)
print('a =', a) # 원소 10개짜리 배열이다

size = 5
for i in range(0, len(a), size):
    print(a[i:i+size])

# 하지만 나눠지지 않는 숫자를 size에 주면? 그냥 유두리 있게 잘 함
size = 3
for i in range(0, len(a), size):
    print(a[i:i+size])

# 1개를 선택하는 것은 에러가 난다
# print(a[12]) #index 12 is out of bounds for axis 0 with size 10

# 하지만 존재하는 인덱스를 포함하는 이런 식은 있는 것 까지 반환해준다
print(a[9:12])
# 부분집합을 만드는 것이기때문에 에러가 나지 않느다
# ex11에서 batch_size를 만들 때 나누어 떨어지는 것에 연연할 필요는 없다


# append 설명
# python의 list
b = [1,2]
c = [3,4,5]
b.append(c)
print(b) # 2차원 리스트가 반환된다
         # c라는 리스트 한개가 b의 리스트에 원소 1개로 추가가 된다 -> 괄호가 두개로 출력된다
         # 그냥 3번째 원소가 리스트인것~
         # [1, 2, [3, 4, 5]]

# numpy의 array
x = np.array([1,2])
y = np.array([3,4,5])
x = np.append(x, y)
print(x) # [1 2 3 4 5]
# 생긴건 똑같았지만, python의 list와 numpy의 array는 다른 결과를 준다
# numpy의 array는 풀어서 리스트를 리턴해준다


# train data -> 신경망 학습 (가중치와 편향등을 직접 찾는 작업)





