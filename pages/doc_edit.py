"""
pages/doc_edit.py: ドキュメント編集画面モジュール
"""
import streamlit as st
from services import EditorService

edit_svc = EditorService()

def render():
    st.header("ドキュメント編集画面")
    left, right = st.columns([1,1])
    with left:
        st.subheader("Markdown 編集エリア")
        edited_md = st.text_area("input", value=st.session_state.generated_md, height=300, key='edited_md')

        renew, save = st.columns([1,1])
        with renew:
            if st.button('更新'):
                # 編集内容を更新する処理
                st.session_state.generated_md = edited_md
                st.success("更新しました。")
        with save:
            if st.button('保存'):
                edit_svc.save(st.session_state.work_prj_id,edited_md)
                st.session_state.page = 'project_select'
                st.rerun()
    with right:
        st.subheader("プレビュー")
        st.markdown(st.session_state.edited_md)