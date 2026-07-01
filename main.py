"""LyricWhisper 程序入口。

启动顺序：
1. 配置 sys.path（开发环境 + PyInstaller onedir 环境）
2. 若 cuda_libs/ 完整，加入 PATH（供 CUDA DLL 加载）
3. 创建 QApplication，应用样式，显示主窗口
"""
import os
import sys


def _setup_paths() -> None:
    """确保 src 包可被导入，且 cuda_libs 在 DLL 搜索路径上。"""
    if getattr(sys, "frozen", False):
        # PyInstaller onedir：程序根 = exe 所在目录
        app_root = os.path.dirname(os.path.abspath(sys.executable))
        # _internal 下包含 src（被 collect 到此处）
        internal = os.path.join(app_root, "_internal")
        for p in (app_root, internal):
            if p not in sys.path:
                sys.path.insert(0, p)
    else:
        app_root = os.path.dirname(os.path.abspath(__file__))
        if app_root not in sys.path:
            sys.path.insert(0, app_root)

    # cuda_libs 加入 PATH（仅在 DLL 完整时，见 gpu_detector）
    try:
        from src.core.gpu_detector import ensure_cuda_libs_on_path
        ensure_cuda_libs_on_path()
    except Exception:
        pass


def main() -> int:
    _setup_paths()

    # 设置环境变量，避免某些库输出冗余日志
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # 配置 HuggingFace 下载使用系统证书库，避免企业 TLS 拦截导致的
    # 证书验证失败（CERTIFICATE_VERIFY_FAILED）。
    try:
        from src.utils.hf_session import configure_hf_session
        configure_hf_session()
    except Exception:
        pass

    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import Qt
    from src.ui.main_window import MainWindow
    from src.ui.constants import QSS

    # 高 DPI
    try:
        QApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )
    except Exception:
        pass

    app = QApplication(sys.argv)
    app.setApplicationName("LyricWhisper")
    app.setStyleSheet(QSS)

    # 设置窗口图标（如有）
    icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icon.ico")
    if os.path.isfile(icon_path):
        from PySide6.QtGui import QIcon
        app.setWindowIcon(QIcon(icon_path))

    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
