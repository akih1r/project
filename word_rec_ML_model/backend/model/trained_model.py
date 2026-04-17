


import numpy as np
from ML_model import CNN
import random

data = np.load("processed_data_split.npz")
x_test = data['x_test']
t_test = data["t_test"]
classes = data['classes']



if x_test.ndim == 3:
    x_test = x_test.reshape(-1, 1, 128, 32)


network = CNN(
    input_dim=(1, 128, 32), 
    hidden_size=100, 
    output_size=len(classes)
)


network.load_params("best_params.pkl")
print("学習済みパラメータを読み込み。")
sum = []
for _ in range(1000):
    ls = []
    for _ in range(50):
        rd = random.randint(0,37)
        ls.append(rd)


    cnt = 0
    for target_index in ls:
        x = x_test[target_index]
        ans = t_test[target_index]

        x = x.reshape(1, 1, 128, 32)

        y = network.predict(x, train_flg=False)


        predicted_class_idx = np.argmax(y)

        if classes[predicted_class_idx] == classes[ans]:
            cnt += 1
    sum.append(cnt / 50)
sum = np.array(sum)
ans = sum.mean()
print(ans)
