import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import tweepy
import time # 時間を測るための道具
# ==========================================
# 1. 手に入れた「4つのカギ」をここに貼り付けてください
# ==========================================
API_KEY = "qxWi4sMsUxcblboiBjo69lbeM"
API_SECRET = "ORviXwY4ce9goBBQiOomrJWyvnvPTTwmzALUAjw3TBWZjvVsek"
ACCESS_TOKEN = "2035402526116450304-SZbgrWaKfMok880R8X11S5EzMth4Rd"
ACCESS_TOKEN_SECRET = "pnWpXDbMkl7t5teUy9GLjkWxtQiH8Kk6BxmwCm5HWjdQ3"

# ==========================================
# 2. AIモデルの準備
# ==========================================
model_dir = "./my_tweet_model"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# ==========================================
# 3. 無限ループ（ずっと繰り返す）
# ==========================================
while True:
    print("-" * 30)
    print(f"実行時刻: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # ツイート内容の生成
    prompt = "今日は"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    attention_mask = torch.ones(input_ids.shape, device=device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids, 
            attention_mask=attention_mask,
            max_length=10, 
            do_sample=True, 
            temperature=1.2, 
            pad_token_id=tokenizer.pad_token_id, 
            bos_token_id=tokenizer.bos_token_id, 
            eos_token_id=tokenizer.eos_token_id
        )

    tweet_text = tokenizer.decode(outputs[0], skip_special_tokens=True).replace(" ", "")
    print(f"生成内容: {tweet_text}")

    # Xへの投稿
    try:
        client = tweepy.Client(
            consumer_key=API_KEY,
            consumer_secret=API_SECRET,
            access_token=ACCESS_TOKEN,
            access_token_secret=ACCESS_TOKEN_SECRET
        )
        client.create_tweet(text=tweet_text)
        print("投稿成功！")
    except Exception as e:
        print(f"投稿エラー（たぶん5ドル分終わったか通信ミス）: {e}")

    # 次の投稿まで待機（秒単位で指定）
    # 例：3600秒 = 1時間
    wait_time = 14400
    print(f"{wait_time // 60}分後にまた投稿します。このまま待機...")
    time.sleep(wait_time)