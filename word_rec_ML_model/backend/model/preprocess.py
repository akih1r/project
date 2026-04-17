import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split

# ==========================================
# 設定とパス
# ==========================================
DATASET_PATH = "../dataset" 
SAVE_PATH = "processed_data_split.npz"
TARGET_WIDTH = 32
SR = 16000
N_MELS = 128

# ==========================================
# 関数定義
# ==========================================
def normalize_audio(y):
    max_val = np.max(np.abs(y))
    if max_val == 0:
        return y
    return y / max_val

def adjust_width(spectrogram, target_width):
    current_width = spectrogram.shape[1]
    if current_width < target_width:
        pad_width = target_width - current_width
        return np.pad(spectrogram, ((0, 0), (0, pad_width)), mode='constant')
    else:
        return spectrogram[:, :target_width]

# ==========================================
# 音声ファイルを画像ファイルに
# ==========================================
print("データを読み込んでいます...")

X_list = []
y_list = []

classes = sorted([d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))])
label_map = {label: i for i, label in enumerate(classes)}

total_files = 0

for label_name in classes:
    class_dir = os.path.join(DATASET_PATH, label_name)
    label_id = label_map[label_name]
    
    with os.scandir(class_dir) as entries:
        for entry in entries:
            if entry.is_file() and entry.name.endswith(('.wav', '.mp3')):
                try:
                    y, _ = librosa.load(entry.path, sr=SR)
                    y_norm = normalize_audio(y)
                    S = librosa.feature.melspectrogram(y=y_norm, sr=SR, n_mels=N_MELS, fmax=8000)
                    S_dB = librosa.power_to_db(S, ref=np.max)
                    S_fixed = adjust_width(S_dB, TARGET_WIDTH)
                    
                    X_list.append(S_fixed)
                    y_list.append(label_id)
                    total_files += 1
                    
                    if total_files % 100 == 0:
                        print(f"{total_files} ファイル処理完了...")
                except Exception as e:
                    print(f"エラースキップ: {entry.name} - {e}")


X_data = np.array(X_list).astype(np.float32)
y_label = np.array(y_list)

# CNN用にチャンネル次元を追加 (N, 1, 128, 32)
X_data = X_data[:, np.newaxis, :, :]

# ==========================================
# 事前に学習データにかたちにしておく
# ==========================================
print("データを分割しています...")
x_train, x_test, t_train, t_test = train_test_split(
    X_data, y_label, test_size=0.2, random_state=42
)

# ==========================================
# データの保存
# ==========================================
np.savez_compressed(
    SAVE_PATH, 
    x_train=x_train, 
    x_test=x_test, 
    t_train=t_train, 
    t_test=t_test, 
    classes=np.array(classes)
)

print("-" * 30)
print(f"分割保存完了: {SAVE_PATH}")
print(f"Train: {x_train.shape}, Test: {x_test.shape}")
print("-" * 30)