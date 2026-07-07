"""LyricWhisper - 歌词自动生成工具"""
__version__ = "1.0.0"

# 必须在任何子模块 import 之前设置 HuggingFace / torch hub 缓存环境变量。
# 原因：huggingface_hub 在首次 import 时会把 HF_HUB_CACHE 读取为模块常量
# （HUGGINGFACE_HUB_CACHE），后续 os.environ 修改无效。faster_whisper /
# whisperx / transformers 会间接 import huggingface_hub，若环境变量未先设置，
# 模型会下载到用户家目录 ~/.cache/huggingface 而非项目目录 models/hf/。
# 放在 src/__init__.py 顶部可保证任何 `from src.xxx import ...` 都先触发此设置。
from .utils.config import setup_hf_env as _setup_hf_env

_setup_hf_env()
