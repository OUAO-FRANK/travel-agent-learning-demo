"""支持使用 ``python -m app`` 启动 CLI。"""

from app.cli import main


if __name__ == "__main__":
    # 将 main 的返回码交给 Python 进程。
    # 这样命令行调用者可以通过退出码判断程序是否正常结束。
    raise SystemExit(main())
