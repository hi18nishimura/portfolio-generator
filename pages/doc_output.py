"""
pages/doc_output.py: ドキュメント出力画面モジュール
"""
import streamlit as st
from services import DocumentService

doc_svc = DocumentService()

def render():
    st.header("ドキュメント出力画面")
    left, right = st.columns([1,1])
    with left:
        st.subheader("概要／工夫")
        st.text_area("システム概要", height=100, value=st.session_state.description,key='description_output')
        st.text_area("開発の工夫", height=100, value=st.session_state.improvements,key='improvements_output')
        st.text_input("追加指示", key='correction')
        if st.button('AI'):
            if st.session_state.correction:
                #AIの出力の補正の処理
                data = doc_svc.call_api_add(st.session_state.correction,st.session_state.work_prj_id)
                st.session_state.description = data['description']
                st.session_state.improvements = data['improvements']
                st.session_state.generated_md = data['markdown']
                st.rerun()
            else:
                st.warning("テキストを入力してください。")
    with right:
        st.subheader("AI 生成結果プレビュー")
        st.markdown(st.session_state.generated_md)
    if st.button('次へ'):
        st.session_state.page = 'doc_edit'
        st.rerun()