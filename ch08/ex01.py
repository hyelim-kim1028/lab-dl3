"""
cmd code
pip list  --outdated  # 현재 버전보다 설치된 버전이 낮은 패키지를 찾을 때
pip install --upgrade 패키지이름1, 패키지이름2, 패키지이름3 # 설치된 패키지를 업데이트 할 때 (여러개의 이름을 한번에 줄 수 있다)
pip install tensorflow # TensorFlow 설치
pip install keras # Keras 설치
"""

import tensorflow as tf
import keras

print('TensorFlow version:',tf.__version__)
print('Keras Version:',keras.__version__)
# Using TensorFlow backend. # Keras가 Tensorflow를 사용할것이라 보여준 것







