
FROM nvidia/cuda:12.6.3-base-ubuntu24.04

# 環境変数でインタラクティブな入力を無効化（インストール中の停止を防ぐ）
ENV DEBIAN_FRONTEND=noninteractive

# OSのパッケージ更新と、Python、音声処理に必要なライブラリをインストール
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3-pip \
    python3-venv \
    libsndfile1 \
    ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# コンテナ内での python コマンドを python3.12 に紐付け
RUN ln -s /usr/bin/python3.12 /usr/bin/python

WORKDIR /app

# 依存関係のインストール
# ※Ubuntu 24.04のpipは「EXTERNALLY-MANAGED」エラーが出ることがあるため
# --break-system-packages をつけるか、venvを使うのが一般的です
COPY requirements.txt .
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

COPY . .

# FastAPIを起動
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
#docker run -d -p 8000:8000 --gpus all --name my-ml-container word-rec-app
#コンテナの情報まとめ
#名前: my-ml-container

#コンテナID: 69f740c3119f... （IDの先頭数文字でも操作可能です）

#イメージ名: word-rec-app