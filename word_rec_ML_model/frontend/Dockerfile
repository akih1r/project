# Node.js 20 の軽量イメージ（alpine = Alpine Linux ベース）を使用
FROM node:20-alpine

WORKDIR /app

# package.json と package-lock.json だけ先にコピー
# → ソースコードが変わっても npm install のキャッシュが再利用される
COPY package*.json ./
RUN npm install

# ソースコード全体をコピー
COPY . .

EXPOSE 3000

# 開発サーバーを起動
# 0.0.0.0 を指定しないとコンテナ外からアクセスできない
CMD ["npm", "run", "dev", "--", "-H", "0.0.0.0"]
