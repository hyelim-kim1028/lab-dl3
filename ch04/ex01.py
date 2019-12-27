"""
Machine Learning (기계학습)
- 그 중 한 종류: deep learning (심층학습)

- what people wanted to do: make an AI (Artificial Intelligence)
- A computer that thinks and decides
 1) 사람들이 직접 프로그램을 짜는 방법
 2) 기계에게 학습을 시켜서 직접 하게하는 방법 -> machine learning  -> 신경망안에 넣으면 기계가 확률을 준다
    - 이 머신러닝의 알고리즘이 다양하게 있다
    - ex. kNN (지도학습, 정답을 리턴, 정답이 있어야한다), SVM (비지도학습, 정답을 리턴하기 보다는 분류, 정답이 꼭 있어야하지 않는다)
    - 그 중 Deep learning (dl) or 신경망
    - 음성인식, 이미지 인식과 같은 것

- deep learning도 기계학습. 그럼 어떻게 기계를 학습 시킬 것 인가?
- training data set (학습 세트)/ test data set (검증 세트)
- 정답지를 가지고 있는가, 없는가?
- 정답지를 가지고 모델을 학습 시키면 -> 지도 학습
- 정답지 없이 데이터만 가지고 학습을 시키면 -> 비지도 학습

- 신경망 층들을 지나갈 때 -> 가중치, 편향도 넣어보고,,, 하는 과정을 반복
- 이러한 단순무식한 작업을 컴퓨터/ 머신에게 시켜보는 것,,, 그것이 머신 러닝!!
- 신경망들이 층들을 지나갈 때 사용되는 weight, bias, 행렬들을 찾는 것이 목적
- 오차를 최소화하는 가중치 행렬들을 찾음
- 그 것을 찾을 때 기준으로하는 근거: 손실함수, 비용함수 (loss function, cost function)
- We find the minimum X of loss/cost function (in short, we are looking for the minimum) => 최솟값
- 손실/비용 함수의 값을 최소화하는 가중치 행렬을 찾음

# 손실 함수 :
    - 평균 제곱 오차 (MSE: Mean Squared Error)
    - 교차 엔트로피 (cross-entropy)
    -
"""
# 오차제곱합: 책과는 다를 수도 있다/ 선생님의 견해: 책에서는 평균은 내지 않고 오차 제곱 합 까지만 계산
from dataset.mnist import load_mnist
import numpy as np
import math

if __name__ == '__main__':
    (X_train, y_train), (X_test, y_true) = load_mnist()

    # 10개 테스트 이미지들의 실제 값
    print('y[:10] =', y_true[:10])

    #10개 테스트 데이터 이미지들의 예측 값
    y_pred = np.array([7, 2, 1, 6, 4, 1, 4, 9, 6, 9])
             # y_true에서 값 2개를 강제로 바꿔주었다
    print('y_pred =', y_pred)

    #오차
    error = y_pred - y_true[:10]
    print('error =',error)
    # 부호의 문제 때문에 오차를 제곱해서 사용한다

    # 오차 제곱 = squared error
    sq_err = error**2
    print('squared_error =', sq_err)

    # 평균 제곱 오차 (mean squared error)
    mse = np.mean(sq_err)
    print('MSE =', mse)
    # 단위를 맞춰주기 위해서 루트를 사용 Root Mean Squared Error (RMSE)
    print('RMSE =', np.sqrt(mse))
    # np.sqrt 는 배열을 계산해주고, math.sqrt는 배열을 계산해주지 않는다

    






