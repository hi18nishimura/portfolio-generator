import json
import requests
import re
import os

API_KEY = "AIzaSyBMWSF9tAMtzl7M13SjNx4kGoFFBfKnEuY"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"

def generate_ai_output(prompt: str) -> str:
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "contents": [{"parts": [{"text": prompt}]}]
    }

    response = requests.post(API_URL, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()['candidates'][0]['content']['parts'][0]['text']
    else:
        raise Exception(f"APIエラー: {response.status_code}\n{response.text}")


def process(file_list, add_prompt=None):
    file_summary = ""
    for fname in file_list:
        try:
            # with open(fname, "r", encoding="utf-8") as f:
            with open(fname, "rb") as f:
                file_bytes = f.read()
            try:
                content = file_bytes.decode("utf-8")
            except UnicodeDecodeError:
                content = file_bytes.decode("shift_jis", errors="replace")
            # 長すぎるファイルは省略（1500文字）
            if len(content) > 1500:
                content = content[:1500] + "\n...（省略）"
            file_summary += f"\n--- {fname} ---\n{content}\n"
        except Exception as e:
            file_summary += f"\n--- {fname} ---\n(読み込みに失敗しました: {e})\n"

    base_prompt = (
        "以下はあるソフトウェアプロジェクトに含まれる複数のPythonファイルの中身です。\n"
        "これらのコードを分析して、このソフトウェアが何をするものなのか（description）、\n"
        "またコード上で工夫されている実装ポイント（improvements）を簡潔に抽出してください。\n"
        "※ このプロンプト自身のコードや説明ではなく、以下のファイル内容だけをもとに判断してください。\n\n"
        "【ファイル一覧と中身】\n"
        f"{file_summary}"
        "\n\n以下の形式で、必ずJSON形式だけを出力してください（説明やマークダウンは不要）：\n"
        '{\n  "description": "（概要）",\n  "improvements": "（工夫された点）"\n}'
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
            "見出しを含め、わかりやすい構成にしてください。"
        )
        markdown = generate_ai_output(prompt2)
    except Exception:
        markdown = f"# プロジェクト概要\n{description}\n\n## 工夫されている点\n{improvements}"
    
    return {
        "description": description,
        "improvements": improvements,
        "markdown": markdown
    }


def extract_section(text, keyword):
    keyword = keyword.lower()
    for line in text.splitlines():
        if keyword in line.lower():
            parts = line.split(":", 1)
            if len(parts) == 2:
                return parts[1].strip()
    return ""


# if __name__ == "__main__":
#     with open("file_list.txt", "r") as f:
#         file_pos = [line.strip() for line in f if line.strip()]  # ← ここに解析したいファイルを列挙
#     # print(file_pos)
#     result = process(file_pos)
#     print(result["markdown"])
