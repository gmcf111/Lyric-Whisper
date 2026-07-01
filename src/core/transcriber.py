"""语音转写：使用 faster-whisper（ctranslate2 后端）转写人声轨，开启词级时间戳。

手动指定语言时跳过自动检测（language 参数直接传入 ISO 639-1 代码）。
默认 CPU：compute_type=int8；GPU：compute_type=float16。
"""
import os
from typing import Callable

from ..utils.config import get_config


class TranscriptionError(Exception):
    pass


class TranscriptionCancelled(TranscriptionError):
    pass


# 已知 Whisper 模型档位（按从小到大排序）
KNOWN_WHISPER_MODELS = ["tiny", "base", "small", "medium", "large-v3"]


def detect_available_whisper_models(model_dir: str) -> list[str]:
    """检测 models/whisper/ 目录下可用的 whisper 模型。

    仅返回实际存在且包含模型文件的模型名称（按预定义顺序）。
    CTranslate2 格式模型至少包含 model.bin 文件。
    """
    if not os.path.isdir(model_dir):
        return []

    available: list[str] = []
    for name in KNOWN_WHISPER_MODELS:
        path = os.path.join(model_dir, name)
        if _is_valid_model_dir(path):
            available.append(name)
    return available


def _is_valid_model_dir(path: str) -> bool:
    """判断目录是否包含完整的 CTranslate2 whisper 模型文件。

    CTranslate2 格式的 faster-whisper 模型至少需要三个组件：
      - model.bin（模型权重）
      - config.json（模型配置）
      - vocabulary.txt 或 tokenizer.json 或 vocab.json（词表）
    仅存在 model.bin 而缺少其他文件（如下载被取消）不算有效。
    """
    if not os.path.isdir(path):
        return False
    try:
        files = {f.lower() for f in os.listdir(path)}
    except OSError:
        return False
    has_bin = any(f.endswith(".bin") for f in files)
    has_config = "config.json" in files
    has_vocab = (
        "vocabulary.txt" in files
        or "tokenizer.json" in files
        or "vocab.json" in files
    )
    return has_bin and has_config and has_vocab


def _to_device(use_gpu: bool) -> str:
    return "cuda" if use_gpu else "cpu"


def _compute_type(use_gpu: bool) -> str:
    # CPU 用 int8（速度/体积平衡）；GPU 用 float16
    return "float16" if use_gpu else "int8"


class Transcriber:
    """封装 faster-whisper 转写逻辑。"""

    def __init__(self, use_gpu: bool = False, model_size: str | None = None) -> None:
        self.use_gpu = use_gpu
        self.device = _to_device(use_gpu)
        self.compute_type = _compute_type(use_gpu)
        cfg = get_config()
        self.model_size = model_size or cfg.whisper_model_size or "medium"
        self.model_path = os.path.join(cfg.whisper_model_dir, self.model_size)
        self._model = None
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def _load_model(self):
        if self._model is not None:
            return self._model
        from faster_whisper import WhisperModel  # 延迟导入

        path = self.model_path
        if not os.path.isdir(path):
            # 兼容直接放在 whisper 根目录的情况
            alt = get_config().whisper_model_dir
            if os.path.isdir(alt) and any(f.endswith(".bin") or os.path.isdir(os.path.join(alt, f)) for f in os.listdir(alt)):
                path = alt
            else:
                raise TranscriptionError(
                    f"未找到本地 Whisper 模型：{self.model_path}\n"
                    f"请将 faster-whisper 模型（CTranslate2 格式）放入程序目录下的 "
                    f"models/whisper/{self.model_size}/ 文件夹。\n"
                    f"可从 HuggingFace 下载 Systran/faster-whisper-{self.model_size} 转换格式。"
                )
        try:
            self._model = WhisperModel(
                model_size_or_path=path,
                device=self.device,
                compute_type=self.compute_type,
                local_files_only=True,  # 严禁联网自动拉取
            )
        except TranscriptionError:
            raise
        except Exception as e:
            raise TranscriptionError(f"加载 Whisper 模型失败（{self.device}）：{e}") from e
        return self._model

    def transcribe(
        self,
        audio_path: str,
        language: str | None = None,
        total_duration: float | None = None,
        progress_cb: Callable[[float], None] | None = None,
    ) -> list:
        """转写音频，返回 segments 列表（已物化，含词级时间戳）。

        language: None=自动检测；ISO 639-1 代码=手动指定（跳过自动检测）。
        """
        if not os.path.isfile(audio_path):
            raise TranscriptionError(f"音频文件不存在：{audio_path}")

        self._cancelled = False
        model = self._load_model()

        try:
            segments_iter, info = model.transcribe(
                audio_path,
                language=language,  # None=自动检测；手动指定时跳过检测
                word_timestamps=True,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 500},
                beam_size=5,
                # 自动检测语言时 info.language 为检测结果
            )
        except Exception as e:
            raise TranscriptionError(f"启动转写失败：{e}") from e

        detected_language = getattr(info, "language", None) if language is None else language
        duration = total_duration or getattr(info, "duration", None) or 0.0

        # 物化 segments（生成器），同时报告进度
        materialized: list = []
        try:
            for seg in segments_iter:
                if self._cancelled:
                    raise TranscriptionCancelled("用户取消转写任务")
                materialized.append(seg)
                if progress_cb and duration > 0:
                    ratio = min(seg.end / duration, 1.0)
                    progress_cb(ratio)
        except TranscriptionCancelled:
            raise
        except Exception as e:
            raise TranscriptionError(f"转写出错：{e}") from e

        if progress_cb:
            progress_cb(1.0)

        # 附带元数据
        class Result:
            pass
        res = Result()
        res.segments = materialized
        res.language = detected_language
        res.detected = (language is None)
        res.duration = duration
        return res

    def release(self) -> None:
        self._model = None
        if self.use_gpu:
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
