import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import tweepy
import time 

# 1. APIキーの読み込み
with open("tokens.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

API_KEY = lines[0].strip()
API_SECRET = lines[1].strip()
ACCESS_TOKEN = lines[2].strip()
ACCESS_TOKEN_SECRET = lines[3].strip()


#ツイートさせたいアカウントを選ぶ
user = tweepy.Client(
                consumer_key=API_KEY,
                consumer_secret=API_SECRET,
                access_token=ACCESS_TOKEN,
                access_token_secret=ACCESS_TOKEN_SECRET
            )


model_dir = "./my_tweet_model"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用デバイス: {device}")
model.to(device)

while True:
    print("-" * 30)
    print(f"実行時刻: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # ツイート内容の生成
    prompt = ""
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    attention_mask = torch.ones(input_ids.shape, device=device)
    

    with torch.no_grad():
        outputs = model.generate(
            input_ids, 
            attention_mask=attention_mask,
            max_length=50,          
            do_sample=True, 
            temperature=0.8, 
            repetition_penalty=1.2, 
            pad_token_id=tokenizer.pad_token_id, 
            bos_token_id=tokenizer.bos_token_id, 
            eos_token_id=tokenizer.eos_token_id
        )

    # テキストの復元と整形
    raw_text = tokenizer.decode(outputs[0], skip_special_tokens=True).replace(" ", "")
    
    #改行で区切って最初の1行目だけを取得
    final_tweet = raw_text.split('\n')[0]
    
    print(f"生成内容: {final_tweet}")

    # Xへの投稿
    try:
        if final_tweet:
            
            user.create_tweet(text=final_tweet)
            print("投稿成功！")
        else:
            print("生成されたテキストが空だったので投稿をスキップしました。")
            
    except Exception as e:
        print(f"投稿エラー: {e}")

    wait_time = 14400
    print(f"{wait_time // 60}分後にまた投稿します。このまま待機...")
    time.sleep(wait_time)