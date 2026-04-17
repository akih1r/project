import torch
import torch.nn.functional as F # 確率計算用
import sys
import os
import io
import librosa
import numpy as np
from fastapi import FastAPI, File, UploadFile


try:
    import torchmodel.model
    sys.modules['model'] = torchmodel.model
except: pass


app = FastAPI()


import json
import os

import tempfile
import subprocess

# このファイル(main.py)が置かれているディレクトリを基準にパスを解決
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 軽いJSONファイルだけを読み込む
with open(os.path.join(_BASE_DIR, "torchmodel", "classes.json"), "r", encoding="utf-8") as f:
    CLASS_NAMES = json.load(f)


SR = 16000
N_MELS = 128
TARGET_WIDTH = 32 
MODEL_PATH = os.path.join(_BASE_DIR, "torchmodel", "trained_model.pth")


# webmファイルをwavファイルに変換する関数
def convert_webm_to_wav(webm_bytes: bytes) -> bytes:
    """
    webmファイルのバイト列をwavファイルのバイト列に変換して返す
    """
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=True) as webm_tmp, \
         tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as wav_tmp:
        webm_tmp.write(webm_bytes)
        webm_tmp.flush()
        # ffmpegでwebm→wav変換
        cmd = [
            "ffmpeg", "-y", "-i", webm_tmp.name,
            "-ar", str(SR), "-ac", "1", wav_tmp.name
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        wav_tmp.seek(0)
        return wav_tmp.read()


# モデルロード
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
model.to(device).eval()

def preprocess_audio(file_bytes: bytes):
    y, _ = librosa.load(io.BytesIO(file_bytes), sr=SR)
    
    # トリミングを弱くする (20 -> 40)
    y, _ = librosa.effects.trim(y, top_db=40) 
    
    #音量を最大化
    max_val = np.max(np.abs(y))
    if max_val > 0:
        y = y / max_val
            
    # 3. スペクトログラム変換
    S = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)
    
    # 学習コードと「完全に」同じパディングにする
    current_width = S_dB.shape[1]
    if current_width < TARGET_WIDTH:
        pad_width = TARGET_WIDTH - current_width
        # 学習時と同じく、あえて「0 (最大音量の白)」で埋める！！
        S_fixed = np.pad(S_dB, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
    else:
        S_fixed = S_dB[:, :TARGET_WIDTH]
    
    return torch.from_numpy(S_fixed).float().unsqueeze(0).unsqueeze(0).to(device)

@app.post("/upload-audio")
async def upload_audio(file: UploadFile = File(...)):
    content = await file.read()          # UploadFile.read() は非同期
    wav_content = convert_webm_to_wav(content)  # 通常関数なのでawait不要
    input_tensor = preprocess_audio(wav_content)
    
    with torch.no_grad():
        output = model(input_tensor)
        # 確率(%)に変換
        probs = F.softmax(output, dim=1)[0]
        # 上位3つを取得
        top_probs, top_indices = torch.topk(probs, 3)

    results = []
    for i in range(3):
        idx = top_indices[i].item()
        results.append({
            "label": CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else "???",
            "prob": f"{top_probs[i].item()*100:.1f}%"
        })

    return {
        "filename": file.filename,
        "top_result": results[0],
        "other_candidates": results[1:],
        "message": f"【判定】{results[0]['label']} ({results[0]['prob']})"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)