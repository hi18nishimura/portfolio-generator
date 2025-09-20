import firebase_admin
from firebase_admin import firestore
import bcrypt
from fastapi import HTTPException
from datetime import datetime
import json

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

def save_project_firestore(user_id: str, prompt: str, output: str, prj_name: str, prj_description: str = ""):
	"""
	ユーザーIDに紐づけてプロンプトと出力結果をFirestoreに保存する関数
	"""
	try:
		output_dict = json.loads(json.dumps(output, default=lambda o: o.__dict__))
		# Firestoreクライアントの取得
		db = firebase_admin.firestore.client()
		users_ref = db.collection("users")
		# user_idが一致するユーザーを検索
		query = users_ref.where("user_id", "==", user_id).stream()
		user_doc = next(query, None)
		if user_doc:
			# ドキュメントIDを取得
			doc_id = user_doc.id
			# サブコレクションにデータを追加
			collection_ref = users_ref.document(doc_id).collection("generated_outputs")
			collection_ref.add({
				"createdAt": firestore.SERVER_TIMESTAMP,
				"prj_name": prj_name,
				"prj_description": prj_description,
				"prompt": prompt,
				"output": output_dict
			})

		print(f"ドキュメントが正常に保存されました。")
		return True
	except Exception as e:
		print(f"データの保存中にエラーが発生しました: {e}")
		return False
	
def load_projects_firestore(user_id: str):
	"""
	ユーザーIDに紐づけてプロンプトと出力結果をFirestoreから取得する関数
	"""
	try:
		# Firestoreクライアントの取得
		db = firebase_admin.firestore.client()
		users_ref = db.collection("users")
		# user_idが一致するユーザーを検索
		query = users_ref.where("user_id", "==", user_id).stream()
		user_doc = next(query, None)
		if user_doc:
			# ドキュメントIDを取得
			doc_id = user_doc.id
			# サブコレクションからデータを取得
			collection_ref = users_ref.document(doc_id).collection("generated_outputs")
			docs = collection_ref.order_by("createdAt", direction=firestore.Query.DESCENDING).stream()
			results = []
			for doc in docs:
				data = doc.to_dict()
				data["id"] = doc.id  # ドキュメントIDを追加
				# FirestoreのタイムスタンプをISOフォーマットに変換
				if "createdAt" in data and isinstance(data["createdAt"], firestore.SERVER_TIMESTAMP.__class__):
					data["createdAt"] = data["createdAt"].isoformat()
				results.append(data)
			return results
		else:
			print(f"ユーザーが見つかりません: {user_id}")
			return []
	except Exception as e:
		print(f"データの取得中にエラーが発生しました: {e}")
		return []
