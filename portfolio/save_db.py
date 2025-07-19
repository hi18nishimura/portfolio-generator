
# プロジェクトIDとmarkdownの保存先のパスを受け取り、データベースに保存する処理
import sqlite3
import os

# DB初期化（必要なら一度だけ実行）
def init_db(db_path='projects.db'):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS project_files (
            prj_id TEXT PRIMARY KEY,
            path TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

# DBに保存
def process(path, prj_id, db_path='projects.db'):
    # 初期化（初回または存在しない場合）
    init_db(db_path)

    # 保存処理
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # すでに同じIDがある場合は上書き
    c.execute('''
        INSERT INTO project_files (prj_id, path)
        VALUES (?, ?)
        ON CONFLICT(prj_id) DO UPDATE SET path=excluded.path
    ''', (str(prj_id), path))

    conn.commit()
    conn.close()
