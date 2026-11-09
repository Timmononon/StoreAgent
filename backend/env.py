"""从项目根目录加载 .env；由应用入口或独立脚本在 import 其它 backend 模块前调用一次。"""

from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_LOADED = False


def load_env() -> None:
    global _LOADED
    if _LOADED:
        return
    # 使用项目根路径而非当前工作目录，让独立入口加载同一份环境文件。
    load_dotenv(PROJECT_ROOT / ".env")
    _LOADED = True
