import requests
import json
from google import genai

def prompt_github_info(file_contents, commit_history, issue=None, pr=None, event=None, release=None, additional_instructions=""):
    """
    promptファイルの形式で情報を埋め込んだプロンプトを生成
    """
    prompt_template = '''あなたはGitHubリポジトリの専門家であり、同時にIT企業の面接官でもあります。以下のGitHubの情報（contents, commit, issue, pull request, Events, Releases API）を基に、プロジェクトの詳細な分析と、就職活動で面接官にアピールするためのポイントを包括的に整理してください。出力はJSON形式で、以下の項目を必ず含んでください。

# 抽出してほしい情報
## プロジェクトの概要
* **目的と機能**: リポジトリのREADME.mdを基に、プロジェクトの目的と主要な機能を要約してください。
* **技術的な工夫点**: commitメッセージやpull requestの議論から、開発プロセスやコードベースにおける技術的な工夫や独自のアプローチを抽出してください。
* **開発のストーリー**: commit履歴とReleases APIの情報を基に、プロジェクトの開始から現在までの主要なマイルストーンを時系列でまとめてください。

## 面接官向けの分析
* **基礎知識**: このプロジェクトの取り組みを通じて、どのようなコンピュータサイエンスやソフトウェア工学の基礎知識（例：アルゴリズム、データ構造、設計パターン、ネットワークなど）が活かされているか、具体的に指摘してください。
* **アピール方法**: 上記の基礎知識を、面接官が納得する形でどのように説明すれば良いか、具体的なアドバイスを提供してください。
* **想定される質問**: issueやpull requestの内容から、面接官が深く掘り下げそうな質問（例：「このバグ修正に際して、なぜその解決策を選んだのですか？」）を具体的にいくつか提示してください。

# 入力情報
* contents: {contents}
* commit: {commit}
* issue: {issue}
* pull request: {pr}
* Events: {event}
* Releases API: {release}

'''
    #print(f"file_contents: {json.dumps(file_contents, ensure_ascii=False, indent=2)}")
    prompt = prompt_template.format(
        contents=json.dumps(file_contents, ensure_ascii=False, indent=2) if file_contents else "",
        commit=json.dumps(commit_history, ensure_ascii=False, indent=2) if commit_history else "",
        issue=json.dumps(issue, ensure_ascii=False, indent=2) if issue else "",
        pr=json.dumps(pr, ensure_ascii=False, indent=2) if pr else "",
        event=json.dumps(event, ensure_ascii=False, indent=2) if event else "",
        release=json.dumps(release, ensure_ascii=False, indent=2) if release else ""
    )
    if additional_instructions:
        prompt += f"\n追加の指示:\n{additional_instructions}\n"
    return prompt


def generate_gemini_response(prompt: str):
  """
  Gemini APIを使用してプロンプトに基づく応答を生成
  """
  client = genai.Client()
  print("gemini API呼び出し前")
  response = client.models.generate_content(
      model='gemini-2.5-flash',
      contents=prompt,

      config={
      'response_mime_type': 'application/json',
      'response_schema': {
        "type": "object",
        "properties": {
          "projectOverview": {
            "type": "object",
            "properties": {
              "summary": {"type": "string"},
              "technicalInnovations": {"type": "array", "items": {"type": "string"}},
              "developmentStory": {"type": "string"}
            },
            "required": ["summary", "technicalInnovations", "developmentStory"]
          },
          "interviewAnalysis": {
            "type": "object",
            "properties": {
              "foundationalKnowledge": {
                "type": "object",
                "properties": {
                  "knowledge": {
                    "type": "array",
                    "items": {
                      "type": "object",
                      "properties": {
                        "area": {"type": "string"},
                        "justification": {"type": "string"}
                      },
                      "required": ["area", "justification"]
                    }
                  },
                  "explanationStrategy": {"type": "string"}
                },
                "required": ["knowledge", "explanationStrategy"]
              },
              "expectedQuestions": {
                "type": "array",
                "items": {
                  "type": "object",
                  "properties": {
                    "question": {"type": "string"},
                    "answerTips": {"type": "string"}
                  },
                  "required": ["question", "answerTips"]
                }
              }
            },
            "required": ["foundationalKnowledge", "expectedQuestions"]
          }
        },
        "required": ["projectOverview", "interviewAnalysis"]
      }
    },
  )
  print("gemini API呼び出し後")
  print(f"Gemini API Response: {response.text}")
  return response