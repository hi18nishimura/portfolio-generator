#prj_name:プロジェクトの名前
#zip_file:アップロードされたzipファイル

#この関数は、プロジェクト名とzipファイルを受け取ります。
#zipファイルを解凍します。
#zipファイルから展開したファイルは、prjに保存して欲しいです。
#その後、データベースにプロジェクトIDを作り、プロジェクト名とファイルパスを登録します。
#データベースはdbに保存して欲しいです。（適切な名前でないといけない場合はそちらでお願いします）
#prj_idはプロジェクトIDです。それを返して欲しいです。（どのようにIDを割り当てるか任せます。）

#ここで保存されたプロジェクトIDを参照して、ファイル読み込んで、AIに渡す処理を考えています。
#AIに渡す処理は、別の関数で実装する予定です。

import streamlit as st
import zipfile
import os
import sqlite3

def process(prj_name,zip_file,pros_path = "./prj"):
    filepathlist = []

    with zipfile.ZipFile(zip_file, 'r') as zip_ref: #データベースにいれるプロジェクト名と構成ファイルのリストを取得
        zip_ref.extractall(pros_path)

        for info in zip_ref.infolist():
            if not info.is_dir():
                filepathlist.append(os.path.normpath(os.path.join(pros_path, info.filename)))

    conn = sqlite3.connect('prj.db')
    cursor = conn.cursor()
    #id(プロジェクトid),プロジェクト名，タイムスタンプのデータベース
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS prj (
    project_id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_name TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
     )
    ''')
    #id,プロジェクトid,構成ファイルのデータベース
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS prj_files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER,
    file_path TEXT NOT NULL,
    FOREIGN KEY(project_id) REFERENCES prj(project_id)
     )
    ''')

    # プロジェクトを登録
    cursor.execute("INSERT INTO prj (project_name) VALUES (?)", (prj_name,))
    project_id = cursor.lastrowid  # 自動採番されたIDを取得

    # 構成ファイルを登録
    for filepath in filepathlist:
        cursor.execute("INSERT INTO prj_files (project_id, file_path) VALUES (?, ?)", (project_id, filepath))

    conn.commit()
    conn.close()

