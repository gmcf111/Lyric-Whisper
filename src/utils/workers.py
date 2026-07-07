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
from ..core.aligner import Aligner, AlignmentError, AlignmentCancelled
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
        use_separation: bool = True,
        parent=None,
    ) -> None:
        super().__init__(use_gpu, parent)
        self.input_path = input_path
        self.language = language
        self.model_size = model_size
        self.use_separation = use_separation
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

            # 2) 人声分离（可选）
            if self.use_separation:
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
                # 关键：demucs (PyTorch/MKL) 释放后，MKL 内存池仍可能保留缓冲区，
                # 导致加载 faster-whisper 时出现 "mkl malloc:failed to allocate memory"。
                # 这里在 release 之后、转写之前再清理一次 MKL 内存池。
                try:
                    from ..utils.memory import free_mkl_buffers
                    free_mkl_buffers()
                except Exception:
                    pass
                if self._cancelled:
                    self.cancelled.emit()
                    return
                transcribe_input = vocals_wav
            else:
                # 不分离人声：直接转写原始音频，转写进度占满剩余区间
                self._emit(0.05, "已跳过人声分离，直接转写原始音频…")
                transcribe_input = temp_wav

            # 3) 转写
            self._emit(0.50, f"正在转写歌词（{'GPU' if self.use_gpu else 'CPU'}）…")
            self._engine = Transcriber(use_gpu=self.use_gpu, model_size=self.model_size)
            try:
                result = self._engine.transcribe(
                    transcribe_input,
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

            # 繁体自动转为简体（中文 zh / 粤语 yue，繁体多见于粤语与港台中文）
            lang = (result.language or "").lower()
            if cfg.get("simplified_chinese", False) and (
                lang.startswith("zh") or lang.startswith("yue")
            ):
                try:
                    from zhconv import convert as _zh_convert
                    for line in lines:
                        line.text = _zh_convert(line.text, "zh-cn")
                        for w in line.words:
                            w.word = _zh_convert(w.word, "zh-cn")
                except Exception:
                    pass  # zhconv 转换失败时静默跳过，不影响主流程

            class PipelineResult:
                pass
            res = PipelineResult()
            res.lines = lines
            res.language = result.language
            res.detected = result.detected
            res.duration = result.duration
            res.vocals_path = vocals_wav if self.use_separation else transcribe_input
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


class AlignPipelineWorker(BaseWorker):
    """已有歌词文本 → 对齐时间轴 流水线。

    流程：转码 WAV →（可选）人声分离 → 加载对齐模型 → 强制对齐 → 构建 LyricLine。
    不重新转写文本，只用 whisperx.align 把用户提供的文本对齐到音频。

    进度权重：转码 5% + 分离 30% + 模型加载 10% + 对齐 55%。
    """

    def __init__(
        self,
        input_path: str,
        language: str,
        lyric_lines: list[str],
        use_gpu: bool,
        use_separation: bool = True,
        mirror_endpoint: str = "",
        parent=None,
    ) -> None:
        super().__init__(use_gpu, parent)
        self.input_path = input_path
        self.language = language
        self.lyric_lines = lyric_lines
        self.use_separation = use_separation
        self.mirror_endpoint = mirror_endpoint
        self._engine = None  # 复用 BaseWorker.cancel 的 _engine.cancel() 约定

    def run(self) -> None:
        try:
            cfg = get_config()
            base = os.path.splitext(os.path.basename(self.input_path))[0]
            temp_wav = os.path.join(cfg.temp_dir, f"{base}_input.wav")
            vocals_wav = os.path.join(cfg.temp_dir, f"{base}_vocals.wav")

            # 1) 转码为标准 WAV（5%）
            self._emit(0.0, "正在转码音视频为 WAV…")
            duration = probe_duration(self.input_path) or 0.0
            ext = os.path.splitext(self.input_path)[1].lower()
            if ext == ".wav":
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

            # 2) 人声分离（可选，30%）
            if self.use_separation:
                self._emit(0.05, f"正在分离人声（{'GPU' if self.use_gpu else 'CPU'}）…")
                self._engine = VocalSeparator(use_gpu=self.use_gpu)
                try:
                    self._engine.ensure_model(
                        progress_cb=lambda r: self._emit(
                            0.05 + r * 0.05, "正在下载人声分离模型…"
                        ),
                    )
                    if self._cancelled:
                        self.cancelled.emit()
                        return
                    self._engine.separate(
                        temp_wav, vocals_wav,
                        progress_cb=lambda r: self._emit(0.10 + r * 0.25, "正在分离人声…"),
                    )
                except SeparationCancelled:
                    self.cancelled.emit()
                    return
                finally:
                    if self._engine is not None:
                        self._engine.release()
                # 同 PipelineWorker：demucs 释放后、加载对齐模型前清理 MKL 内存池，
                # 避免 "mkl malloc:failed to allocate memory"。
                try:
                    from ..utils.memory import free_mkl_buffers
                    free_mkl_buffers()
                except Exception:
                    pass
                if self._cancelled:
                    self.cancelled.emit()
                    return
                align_input = vocals_wav
            else:
                self._emit(0.05, "已跳过人声分离，直接对齐原始音频…")
                align_input = temp_wav

            # 3) 加载对齐模型（10%）
            self._emit(0.35, f"正在加载对齐模型（{'GPU' if self.use_gpu else 'CPU'}）…")
            self._engine = Aligner(use_gpu=self.use_gpu, language=self.language,
                                   mirror_endpoint=self.mirror_endpoint)
            try:
                self._engine.ensure_model(
                    progress_cb=lambda r: self._emit(0.35 + r * 0.10, "正在加载对齐模型…"),
                )
                if self._cancelled:
                    self.cancelled.emit()
                    return

                # 4) 强制对齐（55%）
                self._emit(0.45, "正在对齐歌词时间轴…")
                lines = self._engine.align_lyrics(
                    align_input, self.lyric_lines, duration,
                    progress_cb=lambda r: self._emit(0.45 + r * 0.55, "正在对齐歌词时间轴…"),
                )
            except AlignmentCancelled:
                self.cancelled.emit()
                return
            finally:
                if self._engine is not None:
                    self._engine.release()
            if self._cancelled:
                self.cancelled.emit()
                return

            # 5) 繁简转换（与转写模式一致：中文 zh / 粤语 yue）
            lang = (self.language or "").lower()
            if cfg.get("simplified_chinese", False) and (
                lang.startswith("zh") or lang.startswith("yue")
            ):
                try:
                    from zhconv import convert as _zh_convert
                    for line in lines:
                        line.text = _zh_convert(line.text, "zh-cn")
                        for w in line.words:
                            w.word = _zh_convert(w.word, "zh-cn")
                except Exception:
                    pass

            class AlignResult:
                pass
            res = AlignResult()
            res.lines = lines
            res.language = self.language
            res.detected = False  # 对齐模式语言为用户手动指定
            res.duration = duration
            res.vocals_path = vocals_wav if self.use_separation else align_input
            res.source_path = self.input_path
            self._emit(1.0, "完成")
            self.finished_ok.emit(res)

        except (FFmpegError, SeparationError, AlignmentError) as e:
            if self._cancelled:
                self.cancelled.emit()
            else:
                self.failed.emit(str(e))
        except Exception as e:
            if self._cancelled:
                self.cancelled.emit()
            else:
                self.failed.emit(f"未知错误：{e}")
