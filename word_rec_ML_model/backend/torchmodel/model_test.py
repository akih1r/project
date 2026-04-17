import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from model import CNN
import torch.nn as nn
import numpy as np

torch.serialization.add_safe_globals([CNN])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load("trained_model.pth", map_location=device, weights_only=False)



checkpoint = torch.load("/app/torchmodel/processed_audio_data.pt")
x = checkpoint['x']         # (N, 1, 128, 32)
t = checkpoint['t']         # (N,)
classes = checkpoint['classes']

full_dataset = TensorDataset(x, t)
print(full_dataset.tensors[0].shape)


# 推論（テスト）フェーズ

# データローダーを作成（1個ずつ取り出す設定）
test_loader = DataLoader(full_dataset, batch_size=1, shuffle=False)

# 正解数をカウントする変数
correct = 0

# 勾配計算をオフにする
with torch.no_grad():
    for data, target in test_loader:
        # データをデバイス（CPU/GPU）に送る
        data = data.to(device)
        target = target.to(device)

        # モデルにデータを入力して予測
        output = model(data)

        # outputの中で最大値のインデックスが「予測したクラス」になる
        # outputの形は (1, クラス数) なので dim=1 で最大を探す
        prediction = output.argmax(dim=1, keepdim=True)

        # 予測が当たっているか確認
        correct += prediction.eq(target.view_as(prediction)).sum().item()

        #最初の数件だけ中身を表示してみる
        print(f"正解: {classes[target.item()]} | 予測: {classes[prediction.item()]}")

# 全体の正解率を表示
accuracy = 100. * correct / len(full_dataset)
print(f"\nテスト完了！ 正解率: {accuracy:.2f}%")

