"""
pages/project_select.py: プロジェクト選択画面モジュール
"""
import streamlit as st
from services import ProjectService

proj_svc = ProjectService()

def render():
    st.header("プロジェクト選択画面")
    projects = proj_svc.list_projects()

    if st.button('＋ 新規作成'):
        st.session_state.page = 'file_select'
        st.rerun()
    
    #過去にプロジェクトがある場合、プロジェクト一覧を表示してほしいです。
    if projects:
        cols = st.columns(2)
        for idx, name in enumerate(projects):
            if cols[idx % 2].button(name):
                st.info(f"'{name}' のページ遷移は未実装です。")
    