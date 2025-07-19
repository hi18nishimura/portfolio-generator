
import sqlite3
import json
import requests
import re

API_KEY = "AIzaSyBMWSF9tAMtzl7M13SjNx4kGoFFBfKnEuY"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"

# Gemini API呼び出し
def generate_ai_output(prompt: str) -> str:
    headers = {"Content-Type": "application/json"}
    data = {"contents": [{"parts": [{"text": prompt}]}]}
    response = requests.post(API_URL, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()['candidates'][0]['content']['parts'][0]['text']
    else:
        raise Exception(f"APIエラー: {response.status_code}\n{response.text}")

# DBからファイルパスを取得
def get_file_list_from_db(prj_id):
    conn = sqlite3.connect("prj.db")
    cursor = conn.cursor()
    cursor.execute("SELECT file_path FROM prj_files WHERE project_id = ?", (prj_id,))
    file_list = [row[0] for row in cursor.fetchall()]
    conn.close()
    return file_list

# ファイル中身をAIに送って概要と工夫点を取得
def process(file_list, add_prompt=None):
    file_summary = ""
    for fname in file_list:
        try:
            with open(fname, "r", encoding="utf-8") as f:
                content = f.read()
            file_summary += f"\n--- {fname} ---\n{content}\n"
        except Exception as e:
            file_summary += f"\n--- {fname} ---\n(読み込みに失敗しました: {e})\n"

    # 🔹1. 最初のプロンプト：要約と工夫点をJSON形式で取得
    base_prompt = (
        "以下はあるプログラムプロジェクトに含まれる複数のソースコードファイルの中身です。\n"
        "出力はすべて日本語でお願いします。\n"
        "これらのコードに書かれている内容をもとに、このプロジェクトの概要（description）と、\n"
        "コード上で工夫されている点（improvements）を簡潔に抽出してください。\n"
        "※ このプロンプト自身の説明ではなく、ファイル内容だけに基づいて出力してください。\n\n"
        f"【ファイル一覧と中身】\n{file_summary}\n\n"
        "次の形式で、**JSON形式のみ** を出力してください：\n"
        '{\n  "description": "ここに概要",\n  "improvements": "ここに工夫点"\n}'
    )

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

    # 🔹2. そのままMarkdown形式を作成（この時点では補足指示は使わない）
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

    # 🔹3. 補足指示がある場合は、それに基づいて再生成（上記のdescription/improvementsを修正）
    if add_prompt:
        try:
            refine_prompt = (
                "以下はあるプログラムプロジェクトに関する情報です。\n"
                f"【ファイル一覧と中身】\n{file_summary}\n\n"
                f"【現在の概要】\n{description}\n\n"
                f"【現在の工夫点】\n{improvements}\n\n"
                f"【現在のMarkdown形式の紹介文】\n{markdown}\n\n"
                f"【補足指示】\n{add_prompt}\n\n"
                "このプロジェクトの【現在の概要】、【現在の工夫点】、【現在のMarkdown形式の紹介文】はAIによって自動生成されたものですが、\n"
                "その改善のための【補足指示】が与えられています。\n"
                "【ファイル一覧と中身】を踏まえて、【補足指示】に従って，より適切な情報に修正してください。\n\n"

                "出力形式：以下の3つのフィールドを含む**JSON形式**でお願いします。\n"
                '{\n  "description": "...",\n  "improvements": "...",\n  "markdown": "..." \n}'
            )

            refined = generate_ai_output(refine_prompt)
            print("=== Gemini Refined Output ===")
            print(refined)
            cleaned_refined = re.sub(r"```json|```", "", refined).strip()
            parsed_refined = json.loads(cleaned_refined)
            description = parsed_refined.get("description", "").strip()
            improvements = parsed_refined.get("improvements", "").strip()
            markdown = parsed_refined.get("markdown", "").strip()
        except Exception as e:
            print("補足指示による再生成に失敗:", e)
            # 元の値を使う

    return {
        "description": description,
        "improvements": improvements,
        "markdown": markdown
    }


# メイン実行部
if __name__ == "__main__":
    prj_id = 1  # 対象プロジェクトID
    file_list = get_file_list_from_db(prj_id)
    result = process(file_list)
    print("\n=== Markdown Output ===\n")
    print(result["markdown"])
