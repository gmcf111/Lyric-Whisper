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


def setup_hf_env() -> None:
    """设置 HuggingFace / torch hub 缓存目录环境变量，必须在 import huggingface_hub 之前调用。

    huggingface_hub 在首次 import 时会把 HF_HUB_CACHE / HF_HOME 读取为模块常量
    （HUGGINGFACE_HUB_CACHE），后续 os.environ 修改无效。因此本函数需在
    任何会间接 import huggingface_hub 的模块（如 faster_whisper / whisperx / transformers）
    之前执行。src/__init__.py 顶部会调用本函数，确保最早期生效。

    路径与 AppConfig._setup_model_cache_dirs 保持一致：
      - TORCH_HOME  -> <root>/models/hub  （demucs 权重）
      - HF_HOME     -> <root>/models/hf   （HF 总目录）
      - HF_HUB_CACHE -> <root>/models/hf/hub （HF hub 缓存，huggingface_hub 实际下载到此）
    """
    root = get_app_root()
    models_dir = os.path.join(root, "models")
    os.environ["TORCH_HOME"] = os.path.join(models_dir, "hub")
    os.environ["HF_HOME"] = os.path.join(models_dir, "hf")
    # 注意：HF_HUB_CACHE 必须指向 $HF_HOME/hub，否则 huggingface_hub 会下载到 $HF_HOME 根目录
    os.environ["HF_HUB_CACHE"] = os.path.join(models_dir, "hf", "hub")
    # 禁用 hf-xet 协议：xet 直接连 HuggingFace 官方 Xet 服务器，不走镜像 endpoint，
    # 国内网络会卡死。设为 1 后 huggingface_hub 回退到普通 HTTP 下载，走镜像源。
    os.environ["HF_HUB_DISABLE_XET"] = "1"

    # MKL 内存管理：解决 demucs (PyTorch/MKL) 释放后转写 (faster-whisper/ctranslate2)
    # 加载时出现 "mkl malloc:failed to allocate memory" 的问题。
    # 原因：MKL 内存池（fast memory manager）保留已释放的内存不归还系统，
    # 导致后续大块连续内存分配失败。以下变量必须在 import torch/numpy 之前设置。
    # MKL_DISABLE_FAST_MM=1：禁用快速内存管理，让 MKL 直接用 malloc/free，
    #   每次 free 即归还系统，避免内存池累积（轻微性能损耗，换取内存可用性）。
    os.environ.setdefault("MKL_DISABLE_FAST_MM", "1")
    # 限制 MKL/OpenMP 线程数，避免多线程分配大块内存导致瞬时峰值过高。
    # 默认 4 线程，对 demucs/faster-whisper 的 CPU 推理足够。
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    os.environ.setdefault("OMP_NUM_THREADS", "4")


class AppConfig:
    """应用配置，所有路径基于程序根目录，避免写死。"""

    def __init__(self) -> None:
        self.root = get_app_root()
        # 子目录
        self.ffmpeg_dir = os.path.join(self.root, "ffmpeg")
        self.models_dir = os.path.join(self.root, "models")
        self.temp_dir = os.path.join(self.root, "temp")
        self.output_dir = os.path.join(self.root, "output")

        # ffmpeg 可执行文件：优先外置 ffmpeg/（便于用户升级替换），
        # 其次打包内置的 _internal/ffmpeg/。
        self.ffmpeg_exe = self._resolve_bundled("ffmpeg", "ffmpeg.exe")
        self.ffprobe_exe = self._resolve_bundled("ffmpeg", "ffprobe.exe")

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

    def _resolve_bundled(self, subdir: str, name: str) -> str:
        """解析随程序分发的资源路径。

        查找顺序：
        1. 程序根目录下 <subdir>/<name>（外置，便于用户替换升级）
        2. 打包内置 _internal/<subdir>/<name>（PyInstaller onedir 收集到此）
        均不存在时返回外置路径（供后续按 PATH 兜底或提示用户）。
        """
        external = os.path.join(self.root, subdir, name)
        if os.path.isfile(external):
            return external
        internal = os.path.join(self.root, "_internal", subdir, name)
        if os.path.isfile(internal):
            return internal
        return external

    def _setup_model_cache_dirs(self) -> None:
        """把 torch hub / HF 缓存目录指向程序目录下 models/。

        实际环境变量设置委托给模块级 setup_hf_env()（在 src/__init__.py
        顶部已调用，确保在 huggingface_hub 首次 import 之前生效）。
        此处仅做幂等保护：若已被 setup_hf_env 设置则跳过。
        """
        setup_hf_env()

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
