#url:githubのリポジトリURL
#api_key:GitHub APIキー

#GitHubのAPIを使用するためにデータベースにAPIキーとURLを登録します。
#プロジェクトIDと紐づけて保存して欲しいです。
#ただここまで実装できない可能性があるので、作らなくても大丈夫です。
import streamlit as st
import zipfile
import os
import sqlite3
import streamlit as st

def process(url = None, api_key = None, prj_id = None):
    
    if url == None or  api_key == None or prj_id ==None :
        return
    
    conn = sqlite3.connect('prj.db')
    cursor = conn.cursor()   
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS gits (
    project_id INTEGER PRIMARY KEY,
    API_Key TEXT NOT NULL,
    repo_URL TEXT NOT NULL
     )
    ''')
    cursor.execute("INSERT INTO gits (project_id, repo_URL, API_Key) VALUES (?, ?, ?)", (prj_id, url, api_key))

    conn.commit()
    conn.close()

    pass