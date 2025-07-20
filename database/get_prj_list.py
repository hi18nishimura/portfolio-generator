import sqlite3

def process(projects_db_path="projects.db", prj_db_path="prj.db"):
    """project_name と readme パスを対応づけて取得する"""
    try:
        # project.db: prj_id, path
        conn1 = sqlite3.connect(projects_db_path)
        cursor1 = conn1.cursor()
        cursor1.execute("SELECT prj_id, path FROM project_files")
        project_rows = cursor1.fetchall()
        conn1.close()
    except sqlite3.Error as e:
        print(f"[Error] Failed to fetch from projects.db: {e}")
        project_rows = []

    try:
        # prj.db: project_id, project_name
        conn2 = sqlite3.connect(prj_db_path)
        cursor2 = conn2.cursor()
        cursor2.execute("SELECT project_id, project_name FROM prj")
        prj_rows = cursor2.fetchall()
        conn2.close()
    except sqlite3.Error as e:
        print(f"[Error] Failed to fetch from prj.db: {e}")
        prj_rows = []

    # project_id → name の辞書を作る（文字列化して対応）
    id_to_name = {str(pid): name for pid, name in prj_rows}

    # 対応付け
    prj_list = []
    name_list = []
    for prj_id, path in project_rows:
        name_list.append(path)
        name = id_to_name.get(str(prj_id))  # prj_idを文字列として照合
        if name:
            prj_list.append({"project_name": name, "path": path, "project_id": prj_id})

 #   print(prj_list)  # 空でもOK
    return prj_list  # 空でもOK
