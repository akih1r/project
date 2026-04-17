"""
import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from ML_model import CNN

# ====================================================
# 1. データの読み込みと整形
# ====================================================
# 保存したファイルを読み込む
data = np.load("processed_data_split.npz")

x_train = data['x_train']
x_test = data['x_test']
t_train = data['t_train']
t_test = data['t_test']
classes = data['classes']


if x_train.ndim == 3:
    x_train = x_train.reshape(-1, 1, 128, 32)
    x_test = x_test.reshape(-1, 1, 128, 32)

print(f"学習データ: {x_train.shape}")
print(f"テストデータ: {x_test.shape}")
print(f"クラス数: {len(classes)}")

# ====================================================
# 2. ネットワークの生成
# ====================================================
network = CNN(
    input_dim=(1, 128, 32), 
    hidden_size=100, 
    output_size=len(classes)
)

# ==========================
# 3. ハイパーパラメータの設定 
# ==========================
learning_rate = 0.01
iters_num = 20000
batch_size = 100
train_size = x_train.shape[0]

iter_per_epoch = max(train_size / batch_size, 1) #データ量全体にたいしてサンプルの大きさ何個ぶんか

train_loss_list = []
train_acc_list = []
test_acc_list = []

# Momentum用の速度(v)を初期化
velocity = {}
for key in network.params.keys():
    velocity[key] = np.zeros_like(network.params[key])

print(f"--- 学習開始 (LR={learning_rate}, Iters={iters_num}) ---")
best_acc = 0.0
# ====================================================
# 4. 学習ループ
# ====================================================
for i in range(iters_num):
    
    if i == 10000:
        learning_rate *= 0.1
        print(f"学習率変更")
        
    # ミニバッチの取得
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    # ホワイトノイズの追加
    noise = np.random.randn(*x_batch.shape) * 0.01
    x_batch = x_batch + noise

    # SpecAugment
    # 周波数マスキング)
    # 128ピクセル（高さ）のうち、ランダムに8ピクセル分を「無音」にする
    f_mask_width = 4
    f0 = np.random.randint(0, 128 - f_mask_width)
    x_batch[:, :, f0:f0+f_mask_width, :] = 0

    # 時間マスキング (Time Masking)
    # 32ピクセル（幅）のうち、ランダムに4ピクセル分を「無音」にする
    t_mask_width = 2
    t0 = np.random.randint(0, 32 - t_mask_width)
    x_batch[:, :, :, t0:t0+t_mask_width] = 0

    grad = network.gradient(x_batch, t_batch)

    # Momentum SGD による更新
    momentum = 0.9
    for key in network.params.keys():
        velocity[key] = momentum * velocity[key] - learning_rate * grad[key]
        network.params[key] += velocity[key]

    loss = network.loss(x_batch, t_batch, train_flg=True)
    train_loss_list.append(loss)

    if i % int(iter_per_epoch) == 0:
        train_acc = network.accuracy(x_train[:2000], t_train[:2000], batch_size=batch_size)
        test_acc = network.accuracy(x_test[:2000], t_test[:2000], batch_size=batch_size)
        
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        
        print(f"Epoch {int(i/iter_per_epoch)} | Loss: {loss:.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
        
        if test_acc > best_acc:
            best_acc = test_acc
            network.save_params("best_params.pkl")  # ファイルに保存
            print(f"最高記録更新")




★ 最高記録更新！保存しました (Acc: 0.0400)
Epoch 0 | Loss: 1.6565 | Train Acc: 0.5655 | Test Acc: 0.5770
★ 最高記録更新！保存しました (Acc: 0.5770)
Epoch 1 | Loss: 1.4662 | Train Acc: 0.6855 | Test Acc: 0.6735
★ 最高記録更新！保存しました (Acc: 0.6735)
Epoch 2 | Loss: 1.3791 | Train Acc: 0.7650 | Test Acc: 0.7260
★ 最高記録更新！保存しました (Acc: 0.7260)
Epoch 3 | Loss: 1.0341 | Train Acc: 0.7575 | Test Acc: 0.7275
★ 最高記録更新！保存しました (Acc: 0.7275)
Epoch 4 | Loss: 1.0483 | Train Acc: 0.8170 | Test Acc: 0.7855
★ 最高記録更新！保存しました (Acc: 0.7855)
Epoch 5 | Loss: 0.7607 | Train Acc: 0.8595 | Test Acc: 0.8165
★ 最高記録更新！保存しました (Acc: 0.8165)
Epoch 6 | Loss: 0.6856 | Train Acc: 0.8325 | Test Acc: 0.7940
Epoch 7 | Loss: 0.8566 | Train Acc: 0.8545 | Test Acc: 0.8145
Epoch 8 | Loss: 0.6166 | Train Acc: 0.8855 | Test Acc: 0.8320
★ 最高記録更新！保存しました (Acc: 0.8320)
Epoch 9 | Loss: 0.5649 | Train Acc: 0.8965 | Test Acc: 0.8330
★ 最高記録更新！保存しました (Acc: 0.8330)
Epoch 10 | Loss: 0.5290 | Train Acc: 0.9055 | Test Acc: 0.8380
★ 最高記録更新！保存しました (Acc: 0.8380)
学習率変更
Epoch 11 | Loss: 0.6580 | Train Acc: 0.9300 | Test Acc: 0.8645
★ 最高記録更新！保存しました (Acc: 0.8645)
Epoch 12 | Loss: 0.4701 | Train Acc: 0.9340 | Test Acc: 0.8650
★ 最高記録更新！保存しました (Acc: 0.8650)
Epoch 13 | Loss: 0.6296 | Train Acc: 0.9280 | Test Acc: 0.8550
Epoch 14 | Loss: 0.5928 | Train Acc: 0.9385 | Test Acc: 0.8705
★ 最高記録更新！保存しました (Acc: 0.8705)
Epoch 15 | Loss: 0.5175 | Train Acc: 0.9385 | Test Acc: 0.8565
Epoch 16 | Loss: 0.4906 | Train Acc: 0.9445 | Test Acc: 0.8725
★ 最高記録更新！保存しました (Acc: 0.8725)




"""



