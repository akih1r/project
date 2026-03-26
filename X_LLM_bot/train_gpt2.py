from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling

# 日本語に強いrinnaの小型モデルを使用
model_name = "rinna/japanese-gpt2-small"

# トークナイザー（文章をAIが読める単語に分割するツール）の準備
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# 前処理したツイートテキストを読み込む
dataset = load_dataset("text", data_files={"train": "extracted_tweets.txt"})

# テキストをトークン（数値）に変換する処理
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# 学習前のベースとなるAIモデルを読み込む
model = AutoModelForCausalLM.from_pretrained(model_name)

# 学習の設定（GPUのメモリに合わせてbatch_sizeなどを調整可能）
training_args = TrainingArguments(
    output_dir="./my_tweet_model",
    num_train_epochs=5,             # 学習を繰り返す回数（増やしすぎると過学習になります）
    per_device_train_batch_size=4,  # 1回に処理するデータ量
    save_steps=500,
    save_total_limit=1,
)

# データをモデルに渡すための準備
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 学習の実行役（Trainer）をセットアップ
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    data_collator=data_collator,
)

print("学習を開始します...")
trainer.train()

# 学習が終わった「あなた専用モデル」を保存
trainer.save_model("./my_tweet_model")
tokenizer.save_pretrained("./my_tweet_model")
print("学習完了！ ./my_tweet_model にAIが保存されました。")