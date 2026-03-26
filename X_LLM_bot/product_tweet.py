from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 1. 学習が終わって保存された「自分専用モデル」を指定
model_dir = "./my_tweet_model"

# 2. モデルとトークナイザー（辞書）の読み込み
print("モデルを読み込んでいます...")
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir)

# GPUが使えるならGPUにモデルを移動させる（爆速になります）
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# 3. AIに投げる「書き出し」の言葉
# 何も思いつかなければ "今日は" とか "最近" にしてみてください
prompt = ""

# 文章をAI用の数字に変換
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

# 4. 文章生成の実行
print(f"「{prompt}」から続く文章を生成中...")
with torch.no_grad():
    outputs = model.generate(
        input_ids,
        max_length=10,          # 最大の長さ
        do_sample=True,         # ランダム性を入れる（毎回違う結果になる）
        top_k=50,               # 上位50個の候補から選ぶ
        top_p=0.95,             # 確率の低い変な単語を削る
        temperature=1.2,        # 1.0より高いとカオスに、低いと無難になります
        repetition_penalty=1.2, # 同じ言葉の繰り返しを防ぐ
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

# 5. 結果を表示
# rinnaモデル特有の [SEP] や [CLS] などの特殊記号を綺麗に消して表示します
decoded_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
# 改行コードなどが含まれる場合があるので整形
clean_text = decoded_text.replace(" ", "")

print("\n" + "="*30)
print("【生成されたツイート】")
print(clean_text)
print("="*30)