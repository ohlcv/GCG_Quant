# upload_to_github.py
import os
from git import Repo
import questionary

def upload_file(repo_url=None, directory=None, filename=None, content=None):
    """Upload a Markdown file to GitHub repository."""
    # 默认仓库 URL
    default_repo_url = "https://github.com/ohlcv/GCG_Quant.git"
    repo_url = repo_url or questionary.text(
        "请输入 GitHub 仓库 URL (回车使用默认):", default=default_repo_url
    ).ask()

    # 本地仓库路径
    repo_path = os.path.join(os.getcwd(), "GCG_Quant")
    if not os.path.exists(repo_path):
        print("克隆仓库...")
        Repo.clone_from(repo_url, repo_path)
    repo = Repo(repo_path)

    # 输入目录和文件名
    directory = directory or questionary.text("请输入目录 (如 docs/communications):").ask()
    filename = filename or questionary.text("请输入文件名 (如 20250228_1030_XXX.md):").ask()

    # 构建文件路径
    if directory:
        file_path = os.path.join(repo_path, directory, filename)
        os.makedirs(os.path.join(repo_path, directory), exist_ok=True)
    else:
        file_path = os.path.join(repo_path, filename)

    # 输入内容
    content = content or questionary.text("请输入内容 (多行输入 Ctrl+D 或 Ctrl+Z 结束):", multiline=True).ask()
    
    # 写入文件
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    # Git 提交和推送
    repo.index.add([file_path])
    repo.index.commit(f"Add {filename} by GYD")
    origin = repo.remote(name='origin')
    origin.push()
    print(f"✅ 已上传到 {repo_url}/{directory}/{filename}")

def main():
    print("欢迎使用 GitHub 上传脚本！")
    while True:
        upload_file()
        if not questionary.confirm("继续上传另一个文件？").ask():
            break

if __name__ == "__main__":
    main()