# 1. ベースとなるイメージ
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

WORKDIR /workspace

# ★ココを追加！「対話形式の質問を無視する」という魔法の命令です
ENV DEBIAN_FRONTEND=noninteractive

# 3. システムに必要なツールを入れる
# タイムゾーンの質問を避けるために tzdata も先回りして設定します
RUN apt-get update && apt-get install -y \
    tzdata \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# 4. ライブラリをインストール
RUN pip install --no-cache-dir \
    diffusers \
    transformers \
    accelerate \
    xformers \
    pillow \
    torchvision \
    datasets \
    sentencepiece