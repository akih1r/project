import json
import re

# 読み込むファイルと保存するファイルの名前
input_file = "tweets.js"
output_file = "extracted_tweets.txt"

with open(input_file, "r", encoding="utf-8") as f:
    raw_data = f.read()

#一番最初に現れる[のインデックス
start_index = raw_data.find('[')
json_data = raw_data[start_index:]

tweets = json.loads(json_data)

with open(output_file, "w", encoding="utf-8") as f:
    for item in tweets:
        text = item["tweet"]["full_text"]
        num_good = item["tweet"]["favorite_count"]
        
        # 前処理1：URL（http:// または https:// から始まる文字列）を削除
        text = re.sub(r'https?://\S+', '', text)
        
        # 前処理2：@ユーザー名を削除
        text = re.sub(r'@[a-zA-Z0-9_]+\s*', '', text)
        
        # 前処理3：「RT」という文字を削除
        text = re.sub(r'\bRT\s*', '', text)
        
        # おまけ：RTや@を消した後に残りやすい先頭のコロン「:」や、前後の余分な空白を綺麗にする
        text = text.strip(' :')
        text = text.strip()
        
        # 文字が空っぽになっていない場合だけファイルに書き込む
        if text:
            f.write(text + "\n")

print("URLやRTの削除処理が完了しました！ extracted_tweets.txt を確認してください。")