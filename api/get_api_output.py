

import sqlite3
import json
import requests
import re
import os

from dotenv import load_dotenv 

# カレントディレクトリ直下の .env を明示的に指定
dotenv_path = os.path.join(os.getcwd(), ".env")
load_dotenv(dotenv_path, override=True)  # override=True で既存の同名環境変数を上書き

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("環境変数 GEMINI_API_KEY が設定されていません。.env を確認してください。")
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
    
    MAX_LEN = 6000  # または適宜調整
    if len(file_summary) > MAX_LEN:
        file_summary = file_summary[:MAX_LEN] + "\n...(中略: 長すぎるため省略されました)...\n"

    # 1. 最初のプロンプト：要約と工夫点をJSON形式で取得
    base_prompt = (
        "出力はすべて日本語でお願いします。\n"
        "以下【ファイル一覧と中身】はあるプログラムプロジェクトに含まれる複数のソースコードファイルの中身です。\n"
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

    # 2. そのままMarkdown形式を作成（この時点では追加指示は使わない）
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

    # 3. 追加指示がある場合は、それに基づいて再生成
    if add_prompt is not None:
        try:
            refine_prompt = (
                "以下【ファイル内容】はプログラムプロジェクトに含まれる複数のソースコードファイルの中身です。\n"
                "以下【概要(description)】【工夫点(improvements)】【Markdown形式の紹介文】は【ファイル内容】を用いて自動生成された内容です。\n"
                "最初にAIによって自動生成された内容を、【追加指示】に基づいて修正してください。\n\n"
                f"【ファイル内容】\n{file_summary}\n\n"
                f"【概要(description)】\n{description}\n\n"
                f"【工夫点(improvements)】\n{improvements}\n\n"
                f"【Markdown形式の紹介文】\n{markdown}\n\n"
                f"【追加指示】\n{add_prompt}\n\n"
                "これらを考慮し、最終的な以下の出力を**JSON形式**で返してください：\n"
                '{\n  "description": "...",\n  "improvements": "...",\n  "markdown": "..." \n}'
            )
            print("=== file_list ===")
            for f in file_list:
                print(f)
            print("=== file_summary (preview) ===")
            print(file_summary[:500])

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

