import pickle
import numpy as np
from ch03.ex11 import forward
from dataset.mnist import load_mnist

# We followed what's in the book
# 선생님의 견해로 이것은 맞지 않는 코드 인 것 같아서 다른 코드 (ex01)로 사용



if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = load_mnist(one_hot_label= True)

    y_true = y_test[:10] #(10,10)
    print('y_true =', y_true) # 같은 숫자를 one_hot_label 을 사용해서 [00001000] 처럼 표시한다

    with open('../ch03/sample_weight.pkl', 'rb') as file:
        network = pickle.load(file)

    y_pred = forward(network, X_test[:10]) #(10,10)
    print('y_pred =', y_pred)

    print(y_true[0])
    print(y_pred[0])
    # 어떻게 에러를 계산할 것인가? 매우 애매애매,,,, ㅎㅎ

    error = y_pred[0] - y_true[0]
    print(error) # 애매해,,, 왜냐하면 0에서 자기 자신을 빼니까 결국엔 -자기자신이 되어버렸어
                 # 그리고 이 인덱스를 숫자 1개에 해당해서만 값을 주기 때문에, 다른 값들도 오차값을 구해줘야한다
    print(error**2)
    print(np.sum(error**2))  # 0.26713013243767775

    int('y_true[8] =', y_true[8])
    print('y_pred[8] =', y_pred[8])
    print(np.sum((y_true[8] - y_pred[8]) ** 2))