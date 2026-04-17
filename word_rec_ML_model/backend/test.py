import torch
from torch.utils.data import Dataset
import os

dataset_path = "/app/dataset"
class Mydataset(Dataset):
    def __init__(self, dir_path):
        
        #データセットのクラス名をリストにする
        self.class_name = []
        for d in os.listdir(dir_path):
            full_d_path = os.path.join(dir_path, d)
            if os.path.isdir(full_d_path):
                self.class_name.append(d)
        self.class_name.sort()
        #クラス名とインデックスを対応づけ
        self.name_to_idx = {v:i for i, v in enumerate(self.class_name)}
        
        #all_files = [(ファイルパス, そのクラス名)]をつくる
        self.file_paths = []
        self.labels = []
        for name in self.class_name:
            idx = self.name_to_idx[name]
            name_dir_path = os.path.join(dir_path, name)
            for file in os.listdir(name_dir_path):
                if file[-4:] == ".wav":
                    file_fullpath = os.path.join(name_dir_path, file)
                    self.file_paths.append(file_fullpath)
                    self.labels.append(idx)
    
    
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

    
    
    

                    
        
        
                    
        
        