# 実行方法（FastAPI & React & Firestore）

## 1. 事前準備
- Google CloudでサービスアカウントJSONを発行し、`backend`ディレクトリに配置
- `backend/.env` を編集（例）
	- GOOGLE_APPLICATION_CREDENTIALS=./db_account.json
	- FIRESTORE_PROJECT_ID=your-gcp-project-id

## 2. 初回セットアップ
```bash
cd frontend
npm install
cd ../backend
pip install -r requirements.txt
```

## 3. Dockerで実行
```bash
cd .. # プロジェクトルート
docker-compose build
docker-compose up
```

## 4. 動作確認
- ブラウザで `http://localhost:3000`（React）
- APIは `http://localhost:8000`（FastAPI）

## 注意
- `node_modules` はGit管理しません。`frontend/.gitignore`に `node_modules/` を記載してください。
# アプリケーション概要
