#実行時docker exec -it my-ml-container python3 /app/torchmodel/test.py


import os
import librosa
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import json

DATASET_PATH = "/app/dataset" 
SR = 16000
N_MELS = 128
TARGET_WIDTH = 32



class CustomAudioDataset(Dataset):
    def __init__(self, dataset_dir):
        self.file_paths = []
        self.labels = []
        
        self.classes = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        for class_name in self.classes:
            class_dir = os.path.join(dataset_dir, class_name)
            label_id = self.class_to_idx[class_name]
            
            for file_name in os.listdir(class_dir):
                if file_name.endswith(('.wav', '.mp3')):
                    self.file_paths.append(os.path.join(class_dir, file_name))
                    self.labels.append(label_id)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        
        # 音声読み込み(numpy.ndarray)
        y, _ = librosa.load(file_path, sr=SR)
        
        #正規化
        max_val = np.max(np.abs(y))
        if max_val > 0:
            y = y / max_val
            
        #メルスペクトログラム変換(numpy.ndarray (2次元)縦128固定の画像に変換)
        S = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS, fmax=8000)
        S_dB = librosa.power_to_db(S, ref=np.max)
        
        #幅の調整 (TARGET_WIDTHにする)
        current_width = S_dB.shape[1]
        if current_width < TARGET_WIDTH:
            pad_width = TARGET_WIDTH - current_width
            S_fixed = np.pad(S_dB, ((0, 0), (0, pad_width)), mode='constant')
        else:
            S_fixed = S_dB[:, :TARGET_WIDTH]
            
        #テンソルに変換 (Channel, Height, Width) の形にする
        #unsqueeze(0)で（123,32）-> (1,128,32)
        tensor_data = torch.from_numpy(S_fixed).float().unsqueeze(0)
        tensor_label = torch.tensor(label).long()
        
        return tensor_data, tensor_label






def save_preprocessed_data():
    dataset = CustomAudioDataset(DATASET_PATH)
    
    all_features = []
    all_labels = []

    for i in range(len(dataset)):
        data, label = dataset[i]
        all_features.append(data)
        all_labels.append(label)
        
        if i % 50 == 0:  # 50件ごとに進捗を表示
            print(f"現在 {i} 件目を処理中... (全体: {len(dataset)}件)")

    #リストから一つのテンソルに
    x_tensor = torch.stack(all_features)
    t_tensor = torch.stack(all_labels)

    save_path = "/app/torchmodel/processed_audio_data.pt"
    torch.save({
        'x': x_tensor,
        't': t_tensor,
        'classes': dataset.classes
    }, save_path)

    print(f"保存完了！ ファイル名: {save_path}")
    print(f"データの形: {x_tensor.shape}")
    
    
    json_path = "/app/torchmodel/classes.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(dataset.classes, f, ensure_ascii=False)
    print(f"クラス名の軽量ファイルを作成しました！ ファイル名: {json_path}")

if __name__ == "__main__":
    save_preprocessed_data()