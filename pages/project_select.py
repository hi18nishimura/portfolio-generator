"""
pages/project_select.py: プロジェクト選択画面モジュール
"""
import streamlit as st
from services import ProjectService

proj_svc = ProjectService()

def get_load_md(path):
    """Markdownファイルを読み込む"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            md = f.read()
    except FileNotFoundError:
        st.error("Markdownテンプレートファイルが見つかりません。")
        md = ""
    except Exception as e:
        st.error(f"Markdownファイルの読み込みに失敗しました: {e}")
        md = ""
    return md

def render():
    st.markdown("""
        <div style="text-align: center;">
            <h1 style="color: #5C4033; font-size: 3.5em; font-weight: bold;">Portfolio Generator</h1>
        </div>
    """, unsafe_allow_html=True)
    
    # スライド画像を横並びに配置
    # 横に並べたい画像ファイルのリスト
    images = [
        "static/portfolio1.png",
        "static/portfolio2.png",
        "static/portfolio3.png",
        "static/portfolio4.png"
    ]

    # レイアウト中央寄せ（左右に空白を作る）
    left_spacer, content, right_spacer = st.columns([1.5, 5, 1.5])

    # img_cols = st.columns(len(images))  # 画像数だけ列を作成

    with content:
        # 画像を横並び（間隔を調整したい数だけ空列を挟む）
        layout = []
        for i in range(len(images)):
            layout.append(1)
            if i < len(images) - 1:
                layout.append(0.3)  # 間隔（相対比率）

        cols = st.columns(layout)
        for i, img_path in enumerate(images):
            cols[i * 2].image(img_path, width=130)

    st.write("")
    st.write("")
    st.subheader("プロジェクトを選択してください")
        
    if st.button('＋ 新規作成'):
        st.session_state.page = 'file_select'
        st.rerun()
    
    #過去にプロジェクトがある場合、プロジェクト一覧を表示してほしいです。
    projects = proj_svc.list_projects()
    if projects:
        st.subheader("プロジェクト一覧")
        for prj_info in projects:
            col1, col2 = st.columns([4, 1])

            with col1:
                if st.button(prj_info['project_name'], key=f"load_{prj_info['project_id']}"):
                    st.session_state.generated_md = get_load_md(prj_info['path'])
                    st.session_state.page = 'doc_edit'
                    st.session_state.work_prj_id = prj_info['project_id']
                    st.rerun()

            with col2:
                if st.button("🗑️", key=f"delete_{prj_info['project_id']}"):
                    proj_svc.delete_project(prj_info['project_id'])
                    st.success(f"'{prj_info['project_name']}' を削除しました")
                    st.rerun()
