"""
pages/file_select.py: ファイル選択画面モジュール
"""
import streamlit as st
from services import FileService
from services import DocumentService

doc_svc = DocumentService()
file_svc = FileService()

def render():
    st.header("ファイル選択画面")
    #uploaded = st.file_uploader("プロジェクトファイルを選択", accept_multiple_files=True)
    uploaded = st.file_uploader("フォルダを ZIP に圧縮してアップロードしてください",type=["zip"])
    prj_name = st.text_input("プロジェクト名")
    git_url = st.text_input("Git リポジトリ URL")
    api_key = st.text_input("Git personal access token", type="password")
 
    if st.button('完了'):
        if not uploaded:
            st.error("ファイルが選択されていません。")
        if not prj_name:
            st.error("プロジェクト名が入力されていません。")
        if uploaded and prj_name:
            #zipファイルを処理するプログラムを起動
            st.session_state.work_prj_id = file_svc.upload_files(prj_name,uploaded)
            #Gitから情報を取得するプログラムを起動（実装予定）
            if git_url and api_key:
                file_svc.set_git_info(git_url, api_key, st.session_state.work_prj_id)
                #print("Git情報:", st.session_state.git_info)
            st.session_state.uploaded = uploaded
            # st.session_state.git_url = git_url
            # st.session_state.api_key = api_key
            #出力結果を取得する画面遷移の設定
            st.session_state.page = 'doc_output'
            #AIの出力を生成する
            data = doc_svc.call_api(st.session_state.work_prj_id)
            st.session_state.description = data['description']
            st.session_state.improvements = data['improvements']
            st.session_state.generated_md = data['markdown']
            #プログラムを再実行して、画面遷移する
            st.rerun()