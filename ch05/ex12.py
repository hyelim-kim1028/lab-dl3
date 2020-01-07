"""
ex11.py 에서 저장한 pickle 파일을 읽어서, 파라미터(가중치 행렬/ 편향 행렬)들을 화면에 출력
"""
import pickle
# 파일 -> 객체: deserialization(역 직렬화)
with open('ex11_1.pickle', mode='rb') as f:  # r: read, b: binary
    params = pickle.load(f)

for key, param in params.items():
    print(key, ':', param.shape)

# 3D그래프 & 등보선 그래프
