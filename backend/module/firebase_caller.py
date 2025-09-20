import firebase_admin
from firebase_admin import firestore
import bcrypt
from fastapi import HTTPException

def register_user_firestore(user_id: str, password: str, email: str):
	db = firebase_admin.firestore.client()
	users_ref = db.collection("users")
	# 既存ユーザー重複チェック
	query = users_ref.where("user_id", "==", user_id).stream()
	if any(query):
		raise HTTPException(status_code=409, detail="このユーザーIDは既に登録されています")
	# パスワードハッシュ化
	hashed_pw = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
	user_data = {
		"user_id": user_id,
		"password": hashed_pw.decode('utf-8'),
		"email": email
	}
	users_ref.add(user_data)
	return True

def authenticate_user_firestore(user_id: str, password: str):
	db = firebase_admin.firestore.client()
	users_ref = db.collection("users")
	query = users_ref.where("user_id", "==", user_id).stream()
	user_doc = next(query, None)
	if not user_doc:
		raise HTTPException(status_code=401, detail="ユーザーIDまたはパスワードが間違っています")
	user = user_doc.to_dict()
	if not bcrypt.checkpw(password.encode('utf-8'), user["password"].encode('utf-8')):
		raise HTTPException(status_code=401, detail="ユーザーIDまたはパスワードが間違っています")
	return user
