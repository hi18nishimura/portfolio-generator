import streamlit as st
from services import AuthenticationService

auth_svc = AuthenticationService()
def render():
    st.title("ログイン／新規登録")
    mode = st.radio("選択", ["ログイン", "新規登録"])
    username = st.text_input("ユーザ名")
    password = st.text_input("パスワード", type="password")
    if mode == "ログイン":
        if st.button("ログイン"):
            if auth_svc.login(username, password):
                st.session_state.page = 'project_select'
            else:
                st.error("ログインに失敗しました")
    else:
        if st.button("登録"):
            if auth_svc.register(username, password):
                st.session_state.page = 'login_page'
            else:
                st.error("登録に失敗しました")
            st.session_state.page = 'login_page'
            st.rerun()