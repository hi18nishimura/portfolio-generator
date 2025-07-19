"""
services.py: データ操作やAPI呼び出しのスタブクラス
具体的な実装は後で追加します。
"""

class ProjectService:
    """プロジェクト一覧取得／作成"""
    def list_projects(self):
        pass

from database.registar_prj import process as registar_prj
from database.registar_api import process as registar_api

class FileService:
    def upload_files(self, prj_name,zip_file):
        return registar_prj(prj_name,zip_file)
    def set_git_info(self, url, api_key, prj_id):
        registar_api(url, api_key, prj_id)


from api.generate import process as generate
from api.generate_add import process as generate_add

class DocumentService:
    """ドキュメント生成 API 呼び出し"""
    def call_api(self, prj_id) -> dict:
        return generate(prj_id)
    def call_api_add(self, prj_id,prompt) -> dict:
        return generate_add(prj_id,prompt)

from portfolio.save_md import process as save_md
from portfolio.save_db import process as registar_md
class EditorService:
    """ドキュメント保存 API 呼び出し"""
    def save(self,prj_id,markdown):
        path = save_md(markdown,prj_id)
        registar_md(path,prj_id)
