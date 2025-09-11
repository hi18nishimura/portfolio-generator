
from fastapi import FastAPI
import os
from dotenv import load_dotenv
from google.cloud import firestore

load_dotenv()
app = FastAPI()

# Firestoreクライアント初期化
firestore_project_id = os.getenv("FIRESTORE_PROJECT_ID")
firestore_client = firestore.Client(project=firestore_project_id)

@app.get("/")
def read_root():
    # Firestoreからデータ取得例
    docs = firestore_client.collection("sample").stream()
    items = [{"id": doc.id, **doc.to_dict()} for doc in docs]
    return {"message": "Hello from FastAPI!", "items": items}
