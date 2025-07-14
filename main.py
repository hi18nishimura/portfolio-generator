"""
main.py: Streamlit 画面遷移ルーター
"""
import streamlit as st
from pages.project_select import render as render_project_select
from pages.file_select import render as render_file_select
from pages.doc_output import render as render_doc_output
from pages.doc_edit import render as render_doc_edit

# 初期ページ設定
if 'page' not in st.session_state:
    st.session_state.page = 'project_select'

st.set_page_config(page_title="ポートフォリオ自動生成", layout="wide")

# ページごとに render 関数を呼び出し
if st.session_state.page == 'project_select':
    render_project_select()
elif st.session_state.page == 'file_select':
    render_file_select()
elif st.session_state.page == 'doc_output':
    render_doc_output()
elif st.session_state.page == 'doc_edit':
    render_doc_edit()