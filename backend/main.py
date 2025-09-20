from fastapi import FastAPI, Request, HTTPException, Depends
import os
from dotenv import load_dotenv
import bcrypt
from module.model import RegisterRequest, LoginRequest, GeminiPromptRequest
from module.firebase_caller import register_user_firestore, authenticate_user_firestore
from module.settings import settings
from module.github_caller import fetch_github_repo, fetch_github_issues, fetch_github_pull_requests, fetch_github_events, fetch_github_releases
from module.gemini_caller import prompt_github_info,generate_gemini_response
from fastapi.middleware.cors import CORSMiddleware
import firebase_admin
from firebase_admin import credentials, firestore
from jose import jwt,JWTError
from datetime import datetime, timedelta, timezone

# 環境変数の読み込み
API_VERSION = settings.API_VERSION
FRONTEND_URL = settings.FRONTEND_URL
FIRESTORE_PROJECT_ID = settings.FIRESTORE_PROJECT_ID
GOOGLE_APPLICATION_CREDENTIALS = settings.GOOGLE_APPLICATION_CREDENTIALS
# JWT設定
SECRET_KEY = settings.JWT_SECRET_KEY
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60
# API初期化
load_dotenv()
app = FastAPI()
# 認証済みユーザーのみアクセス可能なテスト用エンドポイント
from fastapi.security import OAuth2PasswordBearer
oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"/{API_VERSION}/login")

# JWT認証用関数
def get_current_user(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="トークンが不正です")
        return True
    except JWTError:
        raise HTTPException(status_code=401, detail="トークンが不正または期限切れです")

# JWT生成メソッド
def create_access_token(user_id: str, expires_delta: timedelta = None):
    expire = datetime.now(tz=timezone.utc) + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    payload = {
        "sub": user_id,
        "exp": expire
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

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
        # Firestore登録
        register_user_firestore(request.user_id, request.password, request.email)
        # JWT発行
        access_token = create_access_token(request.user_id)
        return {"access_token": access_token, "token_type": "bearer"}
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"登録エラー: {e}")
        raise HTTPException(status_code=500, detail="サーバーエラーが発生しました")
    
# 認証済みユーザーのみアクセス可能なテスト用エンドポイント
@app.get(f"/{API_VERSION}/protected")
def protected_endpoint(token: str = Depends(oauth2_scheme)):
    if not token:
        raise HTTPException(status_code=401, detail="認証トークンが必要です")
    current_user = get_current_user(token)
    return {"message": "認証済みユーザーのみアクセス可能", "user_id": current_user}

@app.post(f"/{API_VERSION}/login")
async def login_user(request: LoginRequest):
    try:
        user = authenticate_user_firestore(request.user_id, request.password)
        # JWT発行
        access_token = create_access_token(user["user_id"])
        print(f"ユーザーログイン成功: {user['user_id']}")
        print(f"発行トークン: {access_token}")
        return {"access_token": access_token, "token_type": "bearer"}
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"ログインエラー: {e}")
        raise HTTPException(status_code=500, detail="サーバーエラーが発生しました")

# Geminiプロンプト送信API
@app.post(f"/{API_VERSION}/project_generate")
async def gemini_prompt_api(request: GeminiPromptRequest, token: str = Depends(oauth2_scheme)):
    # 認証チェック
    if not get_current_user(token):
        raise HTTPException(status_code=401, detail="認証トークンが不正です")
    
    # GeminiAPI呼び出し
    try:
        file_contents, commit_history = fetch_github_repo(request.githubUser, request.githubRepo)
        if request.selectedOptions:
            if request.selectedOptions.get("issue"):
                issue_info = fetch_github_issues(request.githubUser, request.githubRepo)
            if request.selectedOptions.get("pull_request"):
                pr_info = fetch_github_pull_requests(request.githubUser, request.githubRepo)
            if request.selectedOptions.get("events"):
                event_info = fetch_github_events(request.githubUser, request.githubRepo)
            if request.selectedOptions.get("releases"):
                release_info = fetch_github_releases(request.githubUser, request.githubRepo)
        prompt = prompt_github_info(file_contents, commit_history, issue=issue_info if request.selectedOptions.get("issue") else None,
                           pr=pr_info if request.selectedOptions.get("pull_request") else None,
                           event=event_info if request.selectedOptions.get("events") else None,
                           release=release_info if request.selectedOptions.get("releases") else None,
                           additional_instructions=request.geminiPrompt if request.geminiPrompt else "")
        return generate_gemini_response(prompt)
    except Exception as e:
        print(f"GitHubリポジトリ取得エラー: {e}")
        raise HTTPException(status_code=500, detail="GitHubリポジトリの情報取得に失敗しました")
    