import sqlite3
import json
import requests
import re

def getapi(url,token, commit_sha=None):
    parts = url.split("/")
    owner = parts[-2]
    repo = parts[-1]

    if not commit_sha:
        header = {
            'Authorization': f'token {token}',
            'Accept': 'application/vnd.github.v3.json'
        }
        list_commits_url = f"https://api.github.com/repos/{owner}/{repo}/commits?per_page=100&page=1"
        # print(f"[DEBUG] 最新コミットSHA取得URL: {list_commits_url}") # デバッグ用

        try:
            list_response = requests.get(list_commits_url, headers=header)
            list_response.raise_for_status()
            commit_data_list = list_response.json()
            if not commit_data_list:
                print("[エラー] 指定されたリポジトリにコミットが見つかりません。")
                return None
            
            commit_sha = [commit["sha"] for commit in commit_data_list]


            print(f"[情報] 対象コミットハッシュ: {commit_sha} (最新のコミット)")
        except (requests.exceptions.RequestException, IndexError, json.JSONDecodeError) as e:
            print(f"[エラー] 最新コミットのSHA取得中にエラーが発生しました: {e}")
            print(f"レスポンス内容: {list_response.text if 'list_response' in locals() else 'N/A'}")
            return None
    else:
        print(f"[情報] 対象コミットハッシュ: {commit_sha} (指定されたコミット)")


    commit_data_list = []

    for sha in commit_sha:
        url = f"https://api.github.com/repos/{owner}/{repo}/commits/{sha}"
        try:
            res = requests.get(url, headers=header)
            res.raise_for_status()
            commit_json = res.json()

            message = commit_json["commit"]["message"]
            # 差分（patch）を含むファイルごとの変更内容
            diffs = []
            for f in commit_json.get("files", []):
                if "patch" in f:
                    diffs.append(f["patch"])
            diff_text = "\n".join(diffs)

            commit_data_list.append({
                "commit_message": message,
                "diff": diff_text
            })

        except requests.exceptions.RequestException as e:
            print(f"[エラー] {sha} の取得失敗: {e}")
    print("commit_data_list",commit_data_list)
    return commit_data_list #コミットメッセージとその時の差分の辞書のリスト