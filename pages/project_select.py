"""
pages/project_select.py: プロジェクト選択画面モジュール
"""
import streamlit as st
from services import ProjectService

proj_svc = ProjectService()

def render():
    st.header("プロジェクト選択画面")
    projects = proj_svc.list_projects()
    cols = st.columns(2)
    # データベースを読み込んで、プロジェクト一覧を出力する
    # for idx, name in enumerate(projects):
    #     if cols[idx % 2].button(name):
    #         st.info(f"'{name}' は未実装です。")
    
    #プロジェクト一覧を出力できたら、-1で指定して一番最後にボタンを設置する。
    #if cols[-1].button('＋ 新規作成'):
    if cols[0].button('＋ 新規作成'):
        st.session_state.page = 'file_select'
        st.rerun()