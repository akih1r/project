import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from model import CNN
import torch.nn as nn
import numpy as np


# 1. 保存したデータの読み込み
data_dict = torch.load("/app/torchmodel/processed_audio_data.pt")
x = data_dict['x']         # (N, 1, 128, 32)
t = data_dict['t']         # (N,)
classes = data_dict['classes'] #x, tは (N,1,128,32), (N,)のテンソル

# 2. データセットの作成と分割
full_dataset = TensorDataset(x, t) #zipのようなもの
train_size = int(0.8 * len(full_dataset))#full_dataset では　(x[i], t[i])
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])#それぞれi番目のタプルについて８割は学習用、残りはテスト用にわけられる。


#ハイパーパラメータ
batch_size = 32
learning_rate = 0.01
iters_num = 200000
iter_per_epoch = max(train_size / batch_size, 1) #データ量全体にたいしてサンプルの大きさ何個ぶんか



test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
# テストデータは固定（評価用）

test_x, test_t = test_dataset[:] # 小規模なら一気に取り出してTensor化しておくと評価が速い

# 2. モデル・最適化手法の設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN(output_size=len(classes)).to(device)


optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

print(f"学習開始 (Device={device})")

# ====================================================
# 4. 学習ループ
# ====================================================
best_acc = 0.0
iters_counter = 0


while iters_counter < iters_num:
    model.train()
    
    for x_batch, t_batch in train_loader:
        if iters_counter >= iters_num: 
            break
        
        if iters_counter == 10000:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
            print("学習率を変更しました")
        
        
        if iters_counter == 100000:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
            print("学習率をさらに 0.1 倍に下げました")

        x_batch, t_batch = x_batch.to(device), t_batch.to(device)

        # ホワイトノイズ
        noise = torch.randn_like(x_batch) * 0.01
        x_batch = x_batch + noise

        # 周波数マスキング 
        f_mask_width = 4
        f0 = np.random.randint(0, 128 - f_mask_width)
        x_batch[:, :, f0:f0+f_mask_width, :] = 0

        # 時間マスキング
        t_mask_width = 2
        t0 = np.random.randint(0, 32 - t_mask_width)
        x_batch[:, :, :, t0:t0+t_mask_width] = 0

        optimizer.zero_grad()    # 勾配リセット
        outputs = model(x_batch) # 順伝播
        loss = criterion(outputs, t_batch) # 損失計算
        loss.backward()          # 逆伝播（勾配計算）
        optimizer.step()         # パラメータ更新


        if iters_counter % int(iter_per_epoch) == 0:
            model.eval()
            total_correct = 0
            total_samples = 0
            
            with torch.no_grad():
                # test_loader を使って全データを少しずつ回す
                for x_test_batch, t_test_batch in test_loader:
                    x_test_batch = x_test_batch.to(device)
                    t_test_batch = t_test_batch.to(device)
                    
                    outputs = model(x_test_batch)
                    pred = outputs.argmax(dim=1)
                    
                    total_correct += (pred == t_test_batch).sum().item()
                    total_samples += t_test_batch.size(0)
            
            # 全体の平均精度を計算
            test_acc = total_correct / total_samples
            
            print(f"Iter {iters_counter} | Loss: {loss.item():.4f} | Full Test Acc: {test_acc:.4f}")

            #最高値が出た時だけパラメータを保存
            if test_acc > best_acc:
                best_acc = test_acc
                #state_dictの中身はOrderedDict
                torch.save(model.state_dict(), "best_saved_prms.pth")
                print(">> 最高記録更新（全件評価）・モデル保存")

        iters_counter += 1
"""       
#/torchmodel/train.py
--- 学習開始 (Device=cuda) ---
Iter 0 | Loss: 3.6578 | Full Test Acc: 0.0233
>> 最高記録更新（全件評価）・モデル保存
Iter 2645 | Loss: 1.4435 | Full Test Acc: 0.5920
>> 最高記録更新（全件評価）・モデル保存
Iter 5290 | Loss: 1.9880 | Full Test Acc: 0.6784
>> 最高記録更新（全件評価）・モデル保存
Iter 7935 | Loss: 1.1155 | Full Test Acc: 0.7401
>> 最高記録更新（全件評価）・モデル保存
学習率を変更しました
Iter 10580 | Loss: 1.2017 | Full Test Acc: 0.8049
>> 最高記録更新（全件評価）・モデル保存
Iter 13225 | Loss: 1.0960 | Full Test Acc: 0.8300
>> 最高記録更新（全件評価）・モデル保存
Iter 15870 | Loss: 0.8827 | Full Test Acc: 0.8294
Iter 18515 | Loss: 1.0579 | Full Test Acc: 0.8346
>> 最高記録更新（全件評価）・モデル保存
Iter 21160 | Loss: 1.0725 | Full Test Acc: 0.8426
>> 最高記録更新（全件評価）・モデル保存
Iter 23805 | Loss: 0.7675 | Full Test Acc: 0.8305
Iter 26450 | Loss: 0.8549 | Full Test Acc: 0.8431
>> 最高記録更新（全件評価）・モデル保存
Iter 29095 | Loss: 1.1227 | Full Test Acc: 0.8447
>> 最高記録更新（全件評価）・モデル保存
Iter 31740 | Loss: 1.1439 | Full Test Acc: 0.8467
>> 最高記録更新（全件評価）・モデル保存
Iter 34385 | Loss: 0.9844 | Full Test Acc: 0.8469
>> 最高記録更新（全件評価）・モデル保存
Iter 37030 | Loss: 0.9803 | Full Test Acc: 0.8550
>> 最高記録更新（全件評価）・モデル保存
Iter 39675 | Loss: 0.6455 | Full Test Acc: 0.8537
Iter 42320 | Loss: 0.3525 | Full Test Acc: 0.8498
Iter 44965 | Loss: 0.5605 | Full Test Acc: 0.8553
>> 最高記録更新（全件評価）・モデル保存
Iter 47610 | Loss: 0.7907 | Full Test Acc: 0.8559
>> 最高記録更新（全件評価）・モデル保存
Iter 50255 | Loss: 0.9032 | Full Test Acc: 0.8618
>> 最高記録更新（全件評価）・モデル保存
Iter 52900 | Loss: 0.5389 | Full Test Acc: 0.8633
>> 最高記録更新（全件評価）・モデル保存
Iter 55545 | Loss: 0.8221 | Full Test Acc: 0.8675
>> 最高記録更新（全件評価）・モデル保存
Iter 58190 | Loss: 0.8019 | Full Test Acc: 0.8566
Iter 60835 | Loss: 0.6385 | Full Test Acc: 0.8655
Iter 63480 | Loss: 0.4222 | Full Test Acc: 0.8666
Iter 66125 | Loss: 0.7659 | Full Test Acc: 0.8730
>> 最高記録更新（全件評価）・モデル保存
Iter 68770 | Loss: 0.8054 | Full Test Acc: 0.8705
Iter 71415 | Loss: 0.7258 | Full Test Acc: 0.8645
Iter 74060 | Loss: 0.4550 | Full Test Acc: 0.8661
Iter 76705 | Loss: 0.5782 | Full Test Acc: 0.8689
Iter 79350 | Loss: 0.5967 | Full Test Acc: 0.8697
Iter 81995 | Loss: 0.3453 | Full Test Acc: 0.8741
>> 最高記録更新（全件評価）・モデル保存
Iter 84640 | Loss: 0.4259 | Full Test Acc: 0.8761
>> 最高記録更新（全件評価）・モデル保存
Iter 87285 | Loss: 0.6437 | Full Test Acc: 0.8762
>> 最高記録更新（全件評価）・モデル保存
Iter 89930 | Loss: 0.6798 | Full Test Acc: 0.8697
Iter 92575 | Loss: 0.5135 | Full Test Acc: 0.8762
>> 最高記録更新（全件評価）・モデル保存
Iter 95220 | Loss: 0.7401 | Full Test Acc: 0.8748
Iter 97865 | Loss: 0.6155 | Full Test Acc: 0.8770
>> 最高記録更新（全件評価）・モデル保存
学習率をさらに 0.1 倍に下げました（最終調整）
Iter 100510 | Loss: 0.5186 | Full Test Acc: 0.8759
Iter 103155 | Loss: 0.6800 | Full Test Acc: 0.8755
Iter 105800 | Loss: 0.3570 | Full Test Acc: 0.8801
>> 最高記録更新（全件評価）・モデル保存
Iter 108445 | Loss: 0.2080 | Full Test Acc: 0.8799
Iter 111090 | Loss: 0.5896 | Full Test Acc: 0.8807
>> 最高記録更新（全件評価）・モデル保存
Iter 113735 | Loss: 0.2456 | Full Test Acc: 0.8785
Iter 116380 | Loss: 0.1577 | Full Test Acc: 0.8779
Iter 119025 | Loss: 0.3292 | Full Test Acc: 0.8803
Iter 121670 | Loss: 0.2884 | Full Test Acc: 0.8747
Iter 124315 | Loss: 0.7888 | Full Test Acc: 0.8822
>> 最高記録更新（全件評価）・モデル保存
Iter 126960 | Loss: 0.5112 | Full Test Acc: 0.8808
Iter 129605 | Loss: 0.6550 | Full Test Acc: 0.8815
Iter 132250 | Loss: 0.3873 | Full Test Acc: 0.8827
>> 最高記録更新（全件評価）・モデル保存
Iter 134895 | Loss: 0.8924 | Full Test Acc: 0.8797
Iter 137540 | Loss: 0.5368 | Full Test Acc: 0.8792
Iter 140185 | Loss: 0.4073 | Full Test Acc: 0.8792
Iter 142830 | Loss: 0.7528 | Full Test Acc: 0.8774
Iter 145475 | Loss: 0.7673 | Full Test Acc: 0.8783
Iter 148120 | Loss: 0.3082 | Full Test Acc: 0.8819
Iter 150765 | Loss: 0.5295 | Full Test Acc: 0.8810
Iter 153410 | Loss: 0.3928 | Full Test Acc: 0.8790
Iter 156055 | Loss: 0.5078 | Full Test Acc: 0.8813
Iter 158700 | Loss: 0.1875 | Full Test Acc: 0.8791
Iter 161345 | Loss: 0.4424 | Full Test Acc: 0.8809
Iter 163990 | Loss: 0.6537 | Full Test Acc: 0.8821
Iter 166635 | Loss: 0.5498 | Full Test Acc: 0.8817
Iter 169280 | Loss: 0.3328 | Full Test Acc: 0.8814
Iter 171925 | Loss: 0.5049 | Full Test Acc: 0.8823
Iter 174570 | Loss: 0.4545 | Full Test Acc: 0.8806
Iter 177215 | Loss: 0.4164 | Full Test Acc: 0.8799
Iter 179860 | Loss: 0.5347 | Full Test Acc: 0.8840
>> 最高記録更新（全件評価）・モデル保存
Iter 182505 | Loss: 0.5254 | Full Test Acc: 0.8805
Iter 185150 | Loss: 0.5999 | Full Test Acc: 0.8827
Iter 187795 | Loss: 0.3978 | Full Test Acc: 0.8797
Iter 190440 | Loss: 0.4246 | Full Test Acc: 0.8816
Iter 193085 | Loss: 0.6706 | Full Test Acc: 0.8796
Iter 195730 | Loss: 0.4012 | Full Test Acc: 0.8779
Iter 198375 | Loss: 0.3536 | Full Test Acc: 0.8812
"""