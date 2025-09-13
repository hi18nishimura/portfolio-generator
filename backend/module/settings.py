import os

class Settings:
    # 環境変数の読み込み
    # Firebaseのサービスアカウントキーのパス
    GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    # FirestoreのプロジェクトID
    FIRESTORE_PROJECT_ID = os.getenv("FIRESTORE_PROJECT_ID")
    # JWTのシークレットキー
    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
    # フロントエンドのURL
    FRONTEND_URL = os.getenv("FRONTEND_URL")
    # APIのバージョン
    API_VERSION = os.getenv("API_VERSION")
    
settings = Settings()