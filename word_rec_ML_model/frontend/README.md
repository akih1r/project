# Word Rec ML — Frontend

音声を録音してバックエンドの機械学習モデルに単語認識をさせる Web アプリのフロントエンドです。  
[Next.js](https://nextjs.org) (App Router) + TypeScript + Tailwind CSS v4 で構築されています。

---

## 技術スタック

| 項目 | 内容 |
|------|------|
| フレームワーク | Next.js 16.2.2 (App Router) |
| 言語 | TypeScript |
| スタイリング | Tailwind CSS v4 |
| 音声取得 | Web Audio API (`MediaRecorder`, `AnalyserNode`) |

---

## ディレクトリ構成

```
src/
├── app/
│   ├── page.tsx              # トップページ（録音画面）
│   ├── result/page.tsx       # 推論結果表示ページ
│   ├── retry/page.tsx        # 無音検出時のやり直しページ
│   ├── api/
│   │   └── transcri/route.ts # バックエンドへの音声転送 API Route
│   ├── globals.css           # グローバルスタイル（ダークテーマ）
│   └── layout.tsx            # 全ページ共通レイアウト
├── components/
│   ├── WaveformVisualizer.tsx # 録音中の波形キャンバス描画
│   ├── RecordControls.tsx     # 録音開始ボタン
│   └── InferenceResult.tsx    # 推論結果カード
├── hooks/
│   └── useAudioRecorder.ts   # 録音ロジックをまとめたカスタムフック
└── types/
    └── index.ts              # 共通型定義（InferenceResult など）
```

---

## 画面フロー

```
/ (録音画面)
  ↓ ▶ボタンを押して3秒間録音
  ├─ 無音を検出 → /retry（やり直し促進ページ）→ / に戻る
  └─ 音声あり  → バックエンドに送信 → /result（推論結果ページ）
```

---

## 主要コンポーネント・フック

### `useAudioRecorder` (hooks/useAudioRecorder.ts)

録音に関わるすべてのロジックを集約したカスタムフックです。

- `MediaRecorder` でマイク音声を録音（3秒で自動停止）
- `AnalyserNode` を接続してフレームごとの RMS 振幅を計測
- 録音終了後、平均振幅が閾値 (`0.02`) 未満なら `/retry` へ遷移
- 音声ありの場合は `/api/transcri` 経由でバックエンドに送信し、結果を `sessionStorage` に保存して `/result` へ遷移

### `WaveformVisualizer` (components/WaveformVisualizer.tsx)

録音中の音声波形を `<canvas>` にリアルタイム描画するコンポーネントです。

- プレイヘッド（白い縦棒）が左から右に移動
- プレイヘッド左側：各フレームの RMS 振幅をインディゴの縦バーで表示
- プレイヘッド右側：グレーの水平線（未録音エリア）
- `requestAnimationFrame` で約60fpsで更新

### `api/transcri/route.ts`

Next.js の API Route として動作するバックエンドへのプロキシです。  
フロントエンドから受け取った音声ファイルをそのままバックエンドの `/upload-audio` に転送します。

---

## 環境変数

| 変数名 | デフォルト値 | 説明 |
|--------|-------------|------|
| `BACKEND_URL` | `http://localhost:8000` | バックエンド API の URL。Docker Compose 環境では `http://backend:8000` を設定 |

ローカル開発時は `.env.local` に記載します：

```
BACKEND_URL=http://localhost:8000
```

---

## 起動方法

### Docker Compose（推奨）

リポジトリルートで実行します：

```bash
docker compose up --build
```

`http://localhost:3000` でアクセスできます。

### ローカル単体起動

```bash
npm install
npm run dev
```

---

## 認識できる単語一覧

| カテゴリ | 単語 |
|----------|------|
| コマンド | Yes, No, Up, Down, Left, Right, On, Off, Stop, Go |
| 数字 | Zero, One, Two, Three, Four, Five, Six, Seven, Eight, Nine |
| その他 | Bed, Bird, Cat, Dog, Happy, House, Marvin, Sheila, Tree, Wow |

