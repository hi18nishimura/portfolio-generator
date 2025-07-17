import sqlite3


def process(prj_id):
    # SQLiteデータベースに接続
    conn = sqlite3.connect("prj.db")
    cursor = conn.cursor()


    # 正しいテーブル名とカラム名に基づいてクエリを実行
    query = "SELECT file_path FROM prj_files WHERE project_id = ?"
    cursor.execute(query, (prj_id,))
   
    # 結果をリストに格納
    file_list = [row[0] for row in cursor.fetchall()]


    # 接続を閉じる
    conn.close()
    print(file_list)
    return file_list
