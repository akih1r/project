# 1. NVIDIA公式の最新開発用イメージ (2026年最新版)
# 通常の PyTorch イメージより重いですが、最新GPUへの対応が早いです
FROM nvcr.io/nvidia/pytorch:26.01-py3

# 2. 作業場所を固定
WORKDIR /workspace

# 3. 必要なライブラリを追加
# イメージが巨大なので、キャッシュを消して軽量化
RUN pip install --no-cache-dir transformers tweepy

# 4. ファイルをコピー
COPY . /workspace/

# 5. プログラム内で確実に GPU を使う設定になっているか確認し、起動！
CMD ["python3", "ai_bot_post.py"]