# マークダウンを受け取り、ファイルに保存する処理
# プロジェクト名のディレクトリをportfolioに作成して、その中に保存する
import os

def process(markdown, prj_id):
    # 保存先ディレクトリの作成
    prj_id=str(prj_id)
    base_dir = "portfolio/md"
    prj_dir = os.path.join(base_dir, prj_id)
    os.makedirs(prj_dir, exist_ok=True)
    
    # Markdownファイルの保存
    file_path = os.path.join(prj_dir, "README.md")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(markdown)
    
    print(file_path)
    return file_path
