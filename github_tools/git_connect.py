from pathlib import Path
import os
import subprocess
import questionary  # 需要安装 pip install questionary
from git import Repo  # 需要安装 pip install GitPython


class GitWizard:
    def __init__(self, path):
        self.path = Path(path)
        self.repo = None
        
    def detect_env(self):
        """环境状态检测"""
        if not self.path.exists():
            return "not_exist"
            
        try:
            self.repo = Repo(self.path)
            return "existing_repo"
        except:
            return "no_repo"
            
    def handle_new_repo(self):
        """处理全新仓库"""
        self.repo = Repo.init(self.path)
        self._create_essential_files()
        self.repo.index.commit("Initial commit")
        
    def _create_essential_files(self):
        """创建必要文件"""
        (self.path / "README.md").write_text("# GCG Quantitative Trading System\n")
        (self.path / ".gitignore").write_text("\n".join([
            "# Python", "__pycache__/", "*.pyc", "*.pyd", "*.pyo",
            "# Data", "*.csv", "*.hdf5", "*.parquet",
            "# Environments", ".env", ".venv/"
        ]))
        (self.path / "LICENSE").write_text("MIT License")
        
    def connect_remote(self, url):
        """配置远程仓库"""
        if "origin" in self.repo.remotes:
            self.repo.delete_remote("origin")
        self.repo.create_remote("origin", url)
        
    def sync_repo(self):
        """同步代码"""
        if not self.repo.head.is_valid():
            raise ValueError("无效的本地仓库")
            
        self.repo.git.branch("-M", "main")
        origin = self.repo.remote("origin")
        try:
            origin.push(refspec="main:main", force=True)
        except Exception as e:
            if "rejected" in str(e):
                self._handle_push_conflict(origin)
            else:
                raise
                
    def _handle_push_conflict(self, origin):
        """处理推送冲突"""
        choice = questionary.select(
            "远程仓库已有内容，请选择操作：",
            choices=["强制覆盖", "拉取合并", "取消"]
        ).ask()
        
        if choice == "强制覆盖":
            origin.push(refspec="main:main", force=True)
        elif choice == "拉取合并":
            origin.pull("main", rebase=True)
            origin.push("main")
            
def main():
    path = questionary.path("请输入项目路径:").ask()
    remote_url = questionary.text(
        "请输入GitHub仓库URL:",
        default="git@github.com:ohlcv/GCG_Quant.git"
    ).ask()
    
    wizard = GitWizard(path)
    state = wizard.detect_env()
    
    if state == "not_exist":
        if questionary.confirm("路径不存在，是否创建?").ask():
            os.makedirs(path, exist_ok=True)
        else:
            return
            
    if state == "no_repo":
        wizard.handle_new_repo()
        
    wizard.connect_remote(remote_url)
    
    try:
        wizard.sync_repo()
        print("✅ 同步成功！访问仓库:")
        print(f"https://github.com/ohlcv/GCG_Quant")
    except Exception as e:
        print(f"❌ 错误: {str(e)}")
        if "Permission denied" in str(e):
            print("请执行以下命令配置SSH密钥:")
            print("ssh-keygen -t ed25519 -C 'your_email@example.com'")
            print("cat ~/.ssh/id_ed25519.pub")

if __name__ == "__main__":
    main()