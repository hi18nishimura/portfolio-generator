from fastapi import FastAPI, Request, HTTPException, Depends
import os
from dotenv import load_dotenv
import bcrypt
from module.Model import RegisterRequest, LoginRequest
from module.settings import settings
from fastapi.middleware.cors import CORSMiddleware
import firebase_admin
from firebase_admin import credentials, firestore
from jose import jwt, JWTError
from datetime import datetime, timedelta, timezone


load_dotenv()
app = FastAPI()


# 環境変数の読み込み
API_VERSION = settings.API_VERSION
FRONTEND_URL = settings.FRONTEND_URL
FIRESTORE_PROJECT_ID = settings.FIRESTORE_PROJECT_ID
GOOGLE_APPLICATION_CREDENTIALS = settings.GOOGLE_APPLICATION_CREDENTIALS
# JWT設定
# JWT設定
SECRET_KEY = settings.JWT_SECRET_KEY
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# JWT認証用関数
def get_current_user(token: str = None):
    if token is None:
        raise HTTPException(status_code=401, detail="認証トークンが必要です")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="トークンが不正です")
        return user_id
    except JWTError:
        raise HTTPException(status_code=401, detail="トークンが不正または期限切れです")
# 認証済みユーザーのみアクセス可能なテスト用エンドポイント
from fastapi.security import OAuth2PasswordBearer
oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"/{API_VERSION}/login")

@app.get(f"/{API_VERSION}/protected")
def protected_endpoint(token: str = Depends(oauth2_scheme)):
    if not token:
        raise HTTPException(status_code=401, detail="認証トークンが必要です")
    current_user = get_current_user(token)
    return {"message": "認証済みユーザーのみアクセス可能", "user_id": current_user}

# Firestoreクライアント初期化
cred = credentials.Certificate(GOOGLE_APPLICATION_CREDENTIALS)
firebase_admin.initialize_app(cred)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL],  # 許可するオリジン
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/')
def read_root():
    return {"message": "Hello, this is the root!!!"}

@app.post(f"/{API_VERSION}/register")
async def register_user(request: RegisterRequest):
    try:
        # パスワードバリデーション（例: 8文字以上）
        if not request.password or len(request.password) < 8:
            raise HTTPException(status_code=400, detail="パスワードは8文字以上で入力してください")

        # パスワードハッシュ化
        hashed_pw = bcrypt.hashpw(request.password.encode('utf-8'), bcrypt.gensalt())

        # Firestore保存（例: usersコレクション）
        user_data = {
            "user_id": request.user_id,
            "password": hashed_pw.decode('utf-8'),
            "email": request.email
        }
        # デバッグ用の出力
        print("user_data:", user_data)
        # Firestoreクライアント取得
        db = firebase_admin.firestore.client()
        users_ref = db.collection("users")
        # 既存ユーザー重複チェック（user_id）
        query = users_ref.where("user_id", "==", request.user_id).stream()
        if any(query):
            raise HTTPException(status_code=409, detail="このユーザーIDは既に登録されています")

        # Firestoreにユーザー情報を登録
        users_ref.add(user_data)
        return {"message": "登録成功", "user_id": request.user_id}
    except HTTPException as e:
        # FastAPIのHTTPExceptionはそのままraise
        raise e
    except Exception as e:
        print(f"登録エラー: {e}")
        raise HTTPException(status_code=500, detail="サーバーエラーが発生しました")

@app.post(f"/{API_VERSION}/login")
async def login_user(request: LoginRequest):
    try:
        db = firebase_admin.firestore.client()
        users_ref = db.collection("users")
        query = users_ref.where("user_id", "==", request.user_id).stream()
        user_doc = next(query, None)
        if not user_doc:
            raise HTTPException(status_code=401, detail="ユーザーIDまたはパスワードが間違っています")
        user = user_doc.to_dict()
        # パスワード認証
        if not bcrypt.checkpw(request.password.encode('utf-8'), user["password"].encode('utf-8')):
            raise HTTPException(status_code=401, detail="ユーザーIDまたはパスワードが間違っています")

        # JWT生成
        expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        payload = {
            "sub": user["user_id"],
            "exp": expire
        }
        access_token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
        return {"access_token": access_token, "token_type": "bearer"}
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"ログインエラー: {e}")
        raise HTTPException(status_code=500, detail="サーバーエラーが発生しました")