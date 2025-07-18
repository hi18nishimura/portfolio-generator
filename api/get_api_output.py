
#file_listにはファイル名のリストが入っています。
#それらを使って、AIにファイルを解析してもらうプロンプトを作成してください。
#AIの出力はプロジェクトの概要（description）と工夫されている点(improvements)、
#それらをまとめたマークダウン形式のドキュメント(markdown)の3つにしてほしいです。

#出力が安定しない場合は、プロジェクトの概要と工夫されている点について生成させます。
#その後それらを入力として、マークダウン形式のドキュメントを生成させてください。

#add_promptがNone出ない場合は、プロンプトにtextを追加してください。

import sqlite3
import json
import requests
import re

API_KEY = "AIzaSyBMWSF9tAMtzl7M13SjNx4kGoFFBfKnEuY"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"

# ✅ Gemini API呼び出し
def generate_ai_output(prompt: str) -> str:
    headers = {"Content-Type": "application/json"}
    data = {"contents": [{"parts": [{"text": prompt}]}]}
    response = requests.post(API_URL, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()['candidates'][0]['content']['parts'][0]['text']
    else:
        raise Exception(f"APIエラー: {response.status_code}\n{response.text}")

# ✅ データベースからファイルパスを取得
def get_file_list_from_db(prj_id):
    conn = sqlite3.connect("prj.db")
    cursor = conn.cursor()
    cursor.execute("SELECT file_path FROM prj_files WHERE project_id = ?", (prj_id,))
    file_list = [row[0] for row in cursor.fetchall()]
    conn.close()
    return file_list

# ✅ ファイル中身をAIに送って概要と工夫点を取得
def process(file_list, add_prompt=None):
    file_summary = ""
    for fname in file_list:
        try:
            with open(fname, "r", encoding="utf-8") as f:
                content = f.read()
            if len(content) > 1500:
                content = content[:1500] + "\n...（省略）"
            file_summary += f"\n--- {fname} ---\n{content}\n"
        except Exception as e:
            file_summary += f"\n--- {fname} ---\n(読み込みに失敗しました: {e})\n"

    base_prompt = (
        "以下はあるPythonプロジェクトに含まれる複数のソースコードファイルの中身です。\n"
        "出力はすべて日本語でお願いします。\n"
        "これらのコードをもとに、このプロジェクトの概要（description）と、\n"
        "コード上で工夫されている点（improvements）を簡潔に抽出してください。\n"
        "※ このプロンプト自身の説明ではなく、ファイル内容だけに基づいて出力してください。\n\n"
        f"【ファイル一覧と中身】\n{file_summary}\n\n"
        "次の形式で、**JSON形式のみ** を出力してください：\n"
        '{\n  "description": "ここに概要",\n  "improvements": "ここに工夫点"\n}'
    )

    if add_prompt:
        base_prompt += f"\n【補足情報】\n{add_prompt}\n"

    try:
        result1 = generate_ai_output(base_prompt)
        print("=== Gemini Raw Output ===")
        print(result1)
        cleaned_result = re.sub(r"```json|```", "", result1).strip()
        parsed = json.loads(cleaned_result)
        description = parsed.get("description", "").strip()
        improvements = parsed.get("improvements", "").strip()
    except Exception as e:
        print("エラー内容:", e)
        description = "プロジェクト概要の生成に失敗しました。"
        improvements = "工夫されている点の抽出に失敗しました。"

    try:
        prompt2 = (
            "以下の情報をもとに、マークダウン形式でプロジェクトの紹介文を作成してください。\n"
            f"- description: {description}\n"
            f"- improvements: {improvements}\n"
            "構成は、# 概要 → ## 工夫点 の順でわかりやすく出力してください。"
        )
        markdown = generate_ai_output(prompt2)
    except Exception:
        markdown = f"# プロジェクト概要\n{description}\n\n## 工夫されている点\n{improvements}"

    return {
        "description": description,
        "improvements": improvements,
        "markdown": markdown
    }
