
#file_listにはファイル名のリストが入っています。
#それらを使って、AIにファイルを解析してもらうプロンプトを作成してください。
#AIの出力はプロジェクトの概要（description）と工夫されている点(improvements)、
#それらをまとめたマークダウン形式のドキュメント(markdown)の3つにしてほしいです。

#出力が安定しない場合は、プロジェクトの概要と工夫されている点について生成させます。
#その後それらを入力として、マークダウン形式のドキュメントを生成させてください。

#add_promptがNone出ない場合は、プロンプトにtextを追加してください。

import os
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai import types
# from google import genai
# from google.genai import types


def process(file_list,add_prompt=None):
    output = {
        'description': 'get_api_output:test',
        'improvements': 'get_api_output:test',
        'markdown': '# get_api_output:test'
    }
    return output
    
#     #GEMINIモデルを呼び出す
#     # .envファイルの読み込み
#     load_dotenv()
#     # APIキーの設定
#     GOOGLE_API_KEY = os.getenv('AIzaSyDG2-Vc8ZghD49auiuzRm-gbRONMpR4yFI')
#     genai.configure(api_key=GOOGLE_API_KEY)

#     #モデルを初期化
#     gemini_pro = genai.GenerativeModel("gemini-pro")

#     # 複数ファイルのPartを作成（複数のコードをAIに渡せるように）
#     parts = []
#     for file_path in file_list:
#         pass
#         # with open(file_path, "rb") as f:
#             # file_bytes = f.read() 
#             # parts.append(types.Blob(inline_data=file_bytes))
#         # parts.append(types.Part.from_bytes(data=file_bytes, mime_type="text/plain"))

#     # プロンプト作成
#     prompt_intro = "以下はプロジェクトを構成するPythonコードです。"
#     prompt_desc = "これらのファイルの内容をもとに、プロジェクトの概要（description）を説明してください。"
#     prompt_impv = "これらのファイルの内容から、工夫されている点（improvements）を抽出してください。"
#     prompt_md = "上記のプロジェクト概要と工夫点をもとに、マークダウン形式のドキュメントを作成してください。"

#     try:
#         # 概要生成
#         response1 = gemini_pro.generate_content(
#             contents=[prompt_intro, prompt_desc] + parts
#         )
#         desc_text = response1.text
#         print(f"概要のレスポンス: {response1.text}")

#         # 工夫点生成
#         response2 = gemini_pro.generate_content(
#             contents=[prompt_intro, prompt_impv] + parts
#         )
#         impv_text = response2.text

#         # マークダウン生成（テキスト入力に変換して）
#         response3 = gemini_pro.generate_content(
#             contents=[
#                 prompt_md,
#                 f"## 概要\n{desc_text}\n\n## 工夫点\n{impv_text}"
#             ]
#         )
#         md_text = response3.text

#     except Exception as e:
#         # エラー時には概要・工夫のみ生成して返す
#         return {
#             "description": desc_text, #if 'desc_text' in locals() else None,
#             "improvements": impv_text, #if 'impv_text' in locals() else None,
#             "markdown": None,
#             "error": str(e)
#         }


#     return {
#         "description": desc_text,
#         "improvements": impv_text,
#         "markdown": md_text
#     }


# output = process(["main.py", "app.py"])

# print("=== 概要 ===")
# print(output["description"])
# print("=== 工夫点 ===")
# print(output["improvements"])
# print("=== Markdownドキュメント ===")
# print(output["markdown"])  

