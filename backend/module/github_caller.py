import os
from git import Repo
import shutil
import subprocess
from fastapi import HTTPException

# GitHubリポジトリからコミットを取得する関数
def get_commit_history(repo_path: str):
    """
    GitPythonを使用して、指定されたリポジトリのコミット履歴を取得します。
    """
    try:
        repo = Repo(repo_path)
        commits = []
        for commit in repo.iter_commits():
            commits.append({
                'message': commit.message.strip()
            })
        return commits
    except Exception as e:
        print(f"Error getting commit history: {e}")
        return None

# GitHubリポジトリから指定された拡張子のファイルを取得する関数
def get_file_contents(repo_path: str):
    """
    指定されたリポジトリパスから、特定の拡張子を持つファイルの中身を取得し、辞書で返します。
    """
    # github_caller.pyの除外ファイル名リスト
    exclude_files = {
        'package.json', 'package-lock.json', 'yarn.lock', 'pnpm-lock.yaml',
        'requirements.txt', 'Pipfile', 'Pipfile.lock', 'pyproject.toml',
        'poetry.lock', 'environment.yml', 'env.yaml', 'setup.py', 'setup.cfg',
        'manage.py', 'Makefile', 'Dockerfile', 'Procfile', 'Gemfile', 'Gemfile.lock',
        'composer.json', 'composer.lock', 'go.mod', 'go.sum', 'Cargo.toml', 'Cargo.lock',
        'build.gradle', 'build.gradle.kts', 'pom.xml', 'CMakeLists.txt', 'Rakefile',
    }
    # github_caller.pyの拡張子リスト
    text_exts = (
        '.txt','.md','.py','.js','.ts','.json','.html','.css','.csv','.yml',
        '.yaml','.xml','.ini','.cfg','.env','.sh','.bat','.java','.c','.cpp',
        '.h','.hpp','.go','.rs','.rb','.php','.swift','.kt','.scala','.pl','.tex','.r','.ipynb'
    )

    file_data = {}
    for root, _, files in os.walk(repo_path):
        for file in files:
            # 指定したファイルだけを対象にする
            if file in exclude_files:
                continue
            if not file.lower().endswith(text_exts):
                continue
            file_path = os.path.join(root, file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_data[file] = f.read()
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    return file_data

# GitHubリポジトリをクローンし、ファイル内容とコミット履歴を取得する関数
def fetch_github_repo(username: str, repo_name: str):

    # GitHub APIを使用してリポジトリのファイル情報を取得
    repo_url = f"https://github.com/{username}/{repo_name}.git"
    # クローンしたリポジトリを一時保存するディレクトリ
    temp_dir = f"/tmp/{repo_name}"

    try:
        # 1. GitHubリポジトリをクローン
        subprocess.run(["git", "clone", repo_url, temp_dir], check=True)

        # 2. 指定した拡張子のファイルのデータだけを取り出し、辞書を作成
        file_contents = get_file_contents(temp_dir)

        # 3. コミット履歴を取得して辞書に追加
        commit_history = get_commit_history(temp_dir)

        # 3. 辞書データを返却
        return file_contents, commit_history

    except subprocess.CalledProcessError as e:
        # git cloneが失敗した場合のエラーハンドリング
        raise HTTPException(status_code=500, detail=f"Git clone failed for {repo_url}: {e}")
    except Exception as e:
        # その他のエラーハンドリング
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")
    finally:
        # 処理後、一時ディレクトリを必ずクリーンアップ
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

# 指定したリポジトリからissueを取得する関数
def fetch_github_issues(username: str, repo_name: str):
    """
    指定リポジトリのIssuesをGitHub API経由で取得
    """
    import requests
    url = f"https://api.github.com/repos/{username}/{repo_name}/issues"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching issues: {e}")
        return None

# 指定したリポジトリからpull requestを取得する関数
def fetch_github_pull_requests(username: str, repo_name: str):
    """
    指定リポジトリのPull RequestsをGitHub API経由で取得
    """
    import requests
    url = f"https://api.github.com/repos/{username}/{repo_name}/pulls"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching pull requests: {e}")
        return None

# 指定したリポジトリからEventsを取得する関数
def fetch_github_events(username: str, repo_name: str):
    """
    指定リポジトリのEventsをGitHub API経由で取得
    """
    import requests
    url = f"https://api.github.com/repos/{username}/{repo_name}/events"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching events: {e}")
        return None

# 指定したリポジトリからReleases APIを取得する関数
def fetch_github_releases(username: str, repo_name: str):
    """
    指定リポジトリのReleasesをGitHub API経由で取得
    """
    import requests
    url = f"https://api.github.com/repos/{username}/{repo_name}/releases"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching releases: {e}")
        return None
