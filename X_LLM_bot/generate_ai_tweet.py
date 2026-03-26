from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 保存した自分専用モデルを読み込む
model_dir = "./my_tweet_model"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir)

# AIに与える「書き出し」の言葉（空文字でもOK）
prompt = ""
inputs = tokenizer(prompt, return_tensors="pt")

# ツイートを生成する
outputs = model.generate(
    **inputs,
    max_length=50,          # 生成する最大文字数
    do_sample=True,         # ランダム性を持たせる
    top_p=0.95,             # 突拍子もない単語を弾く
    top_k=50,
    repetition_penalty=1.2, # 同じ言葉の繰り返しを防ぐ
    pad_token_id=tokenizer.pad_token_id
)

# 生成された数値を日本語テキストに戻して表示
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("【生成されたツイート】")
print(generated_text)