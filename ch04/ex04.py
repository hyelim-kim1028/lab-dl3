import numpy as np

# size에 대한 설명/활용
# [[1,2,3], [4,5,6]] #ndim = 2, shape = (2,3), size = 6 # size = 전체 원소의 갯수
# [1,2,3] #ndim = 1, shape(3,), size =3

a = np.array([1,2,3])
print('dim:', a.ndim)
print('shape:', a.shape)
print('size:', a.size)
print('len:', len(a))

print('===============')

A = np.array([[1,2,3], [4,5,6]])
print('dim:', A.ndim)
print('shape:', A.shape)
print('size:', A.size) #6 # shape[0] * shape[1]
print('len:', len(A))