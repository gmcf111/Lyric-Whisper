"""后台线程任务：分离、转写、完整流水线，全部可取消，不阻塞 UI。

使用 QThread 子类 + 信号。每个 worker 持有对应的 core 对象，
通过 cancel() 触发取消，处理对象在回调中抛异常中止。
"""
import os

from PySide6.QtCore import QThread, Signal

from ..utils.config import get_config
from ..core.separator import VocalSeparator, SeparationCancelled, SeparationError
from ..core.transcriber import Transcriber, TranscriptionCancelled, TranscriptionError
from ..core.lyric_writer import build_lines
from ..utils.ffmpeg import convert_to_wav, probe_duration, FFmpegError


class BaseWorker(QThread):
    progress = Signal(float, str)  # (0~1, 状态文本)
    finished_ok = Signal(object)   # 结果对象
    failed = Signal(str)           # 错误信息
    cancelled = Signal()

    def __init__(self, use_gpu: bool, parent=None) -> None:
        super().__init__(parent)
        self.use_gpu = use_gpu
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True
        if hasattr(self, "_engine") and self._engine is not None:
            try:
                self._engine.cancel()
            except Exception:
                pass

    def is_cancelled(self) -> bool:
        return self._cancelled

    def _emit(self, ratio: float, text: str) -> None:
        self.progress.emit(float(ratio), text)


class PipelineWorker(BaseWorker):
    """完整流水线：转码 -> 人声分离 -> 转写 -> 构建 LyricLine。

    进度权重：转码 5% + 分离 45% + 转写 50%。
    """

    def __init__(
        self,
        input_path: str,
        language: str | None,
        model_size: str | None,
        use_gpu: bool,
        parent=None,
    ) -> None:
        super().__init__(use_gpu, parent)
        self.input_path = input_path
        self.language = language
        self.model_size = model_size
        self._engine = None

    def run(self) -> None:
        try:
            cfg = get_config()
            base = os.path.splitext(os.path.basename(self.input_path))[0]
            temp_wav = os.path.join(cfg.temp_dir, f"{base}_input.wav")
            vocals_wav = os.path.join(cfg.temp_dir, f"{base}_vocals.wav")

            # 1) 转码为标准 WAV
            self._emit(0.0, "正在转码音视频为 WAV…")
            duration = probe_duration(self.input_path) or 0.0
            ext = os.path.splitext(self.input_path)[1].lower()
            if ext == ".wav":
                # 直接使用，避免重复转码
                temp_wav = self.input_path
            else:
                convert_to_wav(
                    self.input_path, temp_wav,
                    progress_cb=lambda r: self._emit(r * 0.05, "正在转码为 WAV…"),
                    cancel_flag=self.is_cancelled,
                )
            if self._cancelled:
                self.cancelled.emit()
                return

            # 2) 人声分离
            self._emit(0.05, f"正在分离人声（{'GPU' if self.use_gpu else 'CPU'}）…")
            self._engine = VocalSeparator(use_gpu=self.use_gpu)
            try:
                # 模型缺失时自动下载缓存（进度占 5%~15%）
                self._engine.ensure_model(
                    progress_cb=lambda r: self._emit(
                        0.05 + r * 0.10, "正在下载人声分离模型…"
                    ),
                )
                if self._cancelled:
                    self.cancelled.emit()
                    return
                self._engine.separate(
                    temp_wav, vocals_wav,
                    progress_cb=lambda r: self._emit(0.15 + r * 0.35, "正在分离人声…"),
                )
            except SeparationCancelled:
                self.cancelled.emit()
                return
            finally:
                if self._engine is not None:
                    self._engine.release()
            if self._cancelled:
                self.cancelled.emit()
                return

            # 3) 转写
            self._emit(0.50, f"正在转写歌词（{'GPU' if self.use_gpu else 'CPU'}）…")
            self._engine = Transcriber(use_gpu=self.use_gpu, model_size=self.model_size)
            try:
                result = self._engine.transcribe(
                    vocals_wav,
                    language=self.language,
                    total_duration=duration,
                    progress_cb=lambda r: self._emit(0.50 + r * 0.50, "正在转写歌词…"),
                )
            except TranscriptionCancelled:
                self.cancelled.emit()
                return
            finally:
                if self._engine is not None:
                    self._engine.release()
            if self._cancelled:
                self.cancelled.emit()
                return

            # 4) 构建歌词行
            lines = build_lines(result.segments)

            class PipelineResult:
                pass
            res = PipelineResult()
            res.lines = lines
            res.language = result.language
            res.detected = result.detected
            res.duration = result.duration
            res.vocals_path = vocals_wav
            res.source_path = self.input_path
            self._emit(1.0, "完成")
            self.finished_ok.emit(res)

        except (FFmpegError, SeparationError, TranscriptionError) as e:
            # 取消导致的异常归类为取消，避免误报失败
            if self._cancelled:
                self.cancelled.emit()
            else:
                self.failed.emit(str(e))
        except Exception as e:
            if self._cancelled:
                self.cancelled.emit()
            else:
                self.failed.emit(f"未知错误：{e}")
