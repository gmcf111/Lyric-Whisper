"""配置管理：模型路径、目录路径，全部基于程序根目录，可配置。"""
import json
import os
import sys


def get_app_root() -> str:
    """获取程序根目录。

    - 打包后（PyInstaller onedir）：sys.executable 所在目录
    - 开发运行：项目根目录（本文件上溯两级）
    """
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    # 开发环境：src/utils/config.py -> 上溯两级到项目根
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class AppConfig:
    """应用配置，所有路径基于程序根目录，避免写死。"""

    def __init__(self) -> None:
        self.root = get_app_root()
        # 子目录
        self.ffmpeg_dir = os.path.join(self.root, "ffmpeg")
        self.models_dir = os.path.join(self.root, "models")
        self.temp_dir = os.path.join(self.root, "temp")
        self.output_dir = os.path.join(self.root, "output")

        # ffmpeg 可执行文件
        self.ffmpeg_exe = os.path.join(self.ffmpeg_dir, "ffmpeg.exe")
        self.ffprobe_exe = os.path.join(self.ffmpeg_dir, "ffprobe.exe")

        # 模型路径（可配置）
        self.whisper_model_dir = os.path.join(self.models_dir, "whisper")
        # Demucs 默认从 torch hub 缓存加载；如需离线可指向本地目录
        self.demucs_repo_dir = os.path.join(self.models_dir, "demucs")

        # 设置文件
        self.settings_file = os.path.join(self.root, "settings.json")

        # 让 torch hub 缓存与 HuggingFace 缓存默认落到程序目录下 models/，
        # 路径可配置、便于离线复用，避免散落在用户目录。
        self._setup_model_cache_dirs()

        # 默认运行参数
        self.whisper_model_size = "medium"  # tiny/base/small/medium/large-v3
        self.demucs_model_name = "htdemucs"
        self.demucs_segment_length = 7.8  # demucs 默认

        self._ensure_dirs()
        self.settings = self._load_settings()

    def _ensure_dirs(self) -> None:
        for d in (self.ffmpeg_dir, self.models_dir, self.temp_dir, self.output_dir):
            os.makedirs(d, exist_ok=True)

    def _setup_model_cache_dirs(self) -> None:
        """把 torch hub / HF 缓存目录指向程序目录下 models/，路径可配置。

        - TORCH_HOME -> models/hub（demucs 权重缓存）
        - HF_HOME    -> models/hf（faster-whisper 模型缓存，本程序使用 local_files_only）
        必须在 import torch 之前设置，故在本模块初始化时完成。
        """
        os.environ["TORCH_HOME"] = os.path.join(self.models_dir, "hub")
        os.environ["HF_HOME"] = os.path.join(self.models_dir, "hf")
        os.environ["HF_HUB_CACHE"] = os.path.join(self.models_dir, "hf")

    def _load_settings(self) -> dict:
        if os.path.isfile(self.settings_file):
            try:
                with open(self.settings_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def save_settings(self) -> None:
        try:
            with open(self.settings_file, "w", encoding="utf-8") as f:
                json.dump(self.settings, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def get(self, key: str, default=None):
        return self.settings.get(key, default)

    def set(self, key: str, value) -> None:
        self.settings[key] = value
        self.save_settings()


# 全局单例
_config: AppConfig | None = None


def get_config() -> AppConfig:
    global _config
    if _config is None:
        _config = AppConfig()
    return _config
