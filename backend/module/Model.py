from pydantic import BaseModel

# 登録情報用のPydanticモデル
class RegisterRequest(BaseModel):
    user_id: str
    password: str
    email: str = None

# ログイン用のPydanticモデル
class LoginRequest(BaseModel):
    user_id: str
    password: str

# Geminiプロンプト送信用リクエストモデル
class GeminiPromptRequest(BaseModel):
    projectName: str
    projectDes: str
    githubUser: str
    githubRepo: str
    selectedOptions: dict = None
    geminiPrompt: str
