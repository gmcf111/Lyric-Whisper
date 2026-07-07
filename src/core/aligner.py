"""歌词文本强制对齐：使用 whisperx 的 wav2vec2 对齐模型，
将用户提供的歌词文本对齐到人声轨，生成带时间轴的 LyricLine。

与 transcriber.py 解耦：transcriber 从零转写音频得到文本；
本模块只做"已有文本 → 时间戳对齐"，不重新转写文本内容。

依赖说明：
- 仅使用 whisperx.alignment（load_align_model / align），不导入 whisperx.diarize，
  因此不依赖 pyannote.audio（说话人分离），打包时可排除以减小体积。
- 对齐模型首次使用时从 HuggingFace 下载（中文/日语约 1GB，英法德等 torchaudio 自带无需下载），
  缓存到 models/hf/（由 config.py 的 HF_HOME 环境变量控制）。

中日文分词：
- whisperx 对 ja/zh 整行作为一个 word（LANGUAGES_WITHOUT_SPACES），
  导致整行只有一个词级时间戳。
- 本模块在构造输入时按行传入，align 返回行级精细 start/end；
  对中日文按字等分行时间范围生成字级 WordTS，满足"按字粒度切分"。
"""
import os
from typing import Callable

import torch

from .lyric_writer import LyricLine, WordTS


class AlignmentError(Exception):
    pass


class AlignmentCancelled(AlignmentError):
    pass


# 下载时跳过的无用文件（其他框架权重）
_IGNORE_EXTS = (".h5", ".msgpack", ".tflite", ".ot")
_IGNORE_NAMES = {"tf_model.h5", "flax_model.msgpack", "tf_model.h5"}


def _get_align_model_dir(language: str) -> str:
    """获取对齐模型的本地存储目录：<app_root>/models/align_models/<lang>/"""
    from ..utils.config import get_app_root
    return os.path.join(get_app_root(), "models", "align_models", language)


# whisperx 内置支持的对齐模型语言（DEFAULT_ALIGN_MODELS_TORCH + DEFAULT_ALIGN_MODELS_HF）
SUPPORTED_ALIGN_LANGS = {
    # torchaudio 自带（英法德西意）
    "en", "fr", "de", "es", "it",
    # HuggingFace wav2vec2
    "ja", "zh", "nl", "uk", "pt", "ar", "cs", "ru", "pl", "hu", "fi",
    "fa", "el", "tr", "da", "he", "vi", "ko", "ur", "te", "hi", "ca",
    "ml", "no", "nn", "sk", "sl", "hr", "ro", "eu", "gl", "ka", "lv",
    "tl", "sv", "id",
}

# 粤语 yue 无独立对齐模型，复用中文 zh 对齐模型
_ALIGN_LANG_MAP = {"yue": "zh"}

# 对齐模型名映射（与 whisperx.alignment.DEFAULT_ALIGN_MODELS_TORCH/_HF 保持一致）。
# 硬编码在此避免每次状态刷新都触发 `from whisperx.alignment import ...`，
# 那会首次加载 torch/torchaudio/transformers/nltk 等重模块导致 UI 卡顿。
_ALIGN_MODELS_TORCH = {
    "en": "WAV2VEC2_ASR_BASE_960H",
    "fr": "VOXPOPULI_ASR_BASE_10K_FR",
    "de": "VOXPOPULI_ASR_BASE_10K_DE",
    "es": "VOXPOPULI_ASR_BASE_10K_ES",
    "it": "VOXPOPULI_ASR_BASE_10K_IT",
}
_ALIGN_MODELS_HF = {
    "ja": "jonatasgrosman/wav2vec2-large-xlsr-53-japanese",
    "zh": "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
    "nl": "jonatasgrosman/wav2vec2-large-xlsr-53-dutch",
    "uk": "Yehor/wav2vec2-xls-r-300m-uk-with-small-lm",
    "pt": "jonatasgrosman/wav2vec2-large-xlsr-53-portuguese",
    "ar": "jonatasgrosman/wav2vec2-large-xlsr-53-arabic",
    "cs": "comodoro/wav2vec2-xls-r-300m-cs-250",
    "ru": "jonatasgrosman/wav2vec2-large-xlsr-53-russian",
    "pl": "jonatasgrosman/wav2vec2-large-xlsr-53-polish",
    "hu": "jonatasgrosman/wav2vec2-large-xlsr-53-hungarian",
    "fi": "jonatasgrosman/wav2vec2-large-xlsr-53-finnish",
    "fa": "jonatasgrosman/wav2vec2-large-xlsr-53-persian",
    "el": "jonatasgrosman/wav2vec2-large-xlsr-53-greek",
    "tr": "mpoyraz/wav2vec2-xls-r-300m-cv7-turkish",
    "da": "saattrupdan/wav2vec2-xls-r-300m-ftspeech",
    "he": "imvladikon/wav2vec2-xls-r-300m-hebrew",
    "vi": "nguyenvulebinh/wav2vec2-base-vi-vlsp2020",
    "ko": "kresnik/wav2vec2-large-xlsr-korean",
    "ur": "kingabzpro/wav2vec2-large-xls-r-300m-Urdu",
    "te": "anuragshas/wav2vec2-xlsr-53-telugu",
    "hi": "theainerd/Wav2Vec2-large-xlsr-hindi",
    "ca": "softcatala/wav2vec2-xls-r-catala",
    "ml": "gvs/wav2vec2-xls-r-malayalam",
    "no": "NbAiLab/nb-wav2vec2-1b-bokmaal-v2",
    "nn": "NbAiLab/nb-wav2vec2-1b-nynorsk",
    "sk": "comodoro/wav2vec2-xls-r-300m-sk-cv8",
    "sl": "anton-l/wav2vec2-large-xlsr-53-slovenian",
    "hr": "classla/wav2vec2-xls-r-parlaspeech-hr",
    "ro": "gigant/romanian-wav2vec2",
    "eu": "stefan-it/wav2vec2-large-xlsr-53-basque",
    "gl": "ifrz/wav2vec2-xlsr-galician",
    "ka": "xsway/wav2vec2-xlsr-georgian",
    "lv": "jimregan/wav2vec2-xlsr-latvian-cv",
    "tl": "Khalsuu/filipino-wav2vec2-l-xls-r-300m-official",
    "sv": "KBLab/wav2vec2-large-voxrex-swedish",
    "id": "cahya/wav2vec2-large-xlsr-indonesian",
}

# whisperx 内部对 ja/zh 整行处理（LANGUAGES_WITHOUT_SPACES）
_LANGS_WITHOUT_SPACES = {"ja", "zh"}


def _to_device(use_gpu: bool) -> str:
    return "cuda" if use_gpu else "cpu"


def _is_oom(e: Exception) -> bool:
    msg = str(e).lower()
    return "out of memory" in msg or "oom" in msg or ("cuda" in msg and "memory" in msg)


class Aligner:
    """封装 whisperx 强制对齐逻辑，不依赖 pyannote.audio。

    使用流程：
        aligner = Aligner(use_gpu, language)
        aligner.ensure_model(progress_cb)          # 加载/下载对齐模型
        lines = aligner.align_lyrics(vocals_path, lyric_lines, duration, progress_cb)
        aligner.release()
    """

    def __init__(self, use_gpu: bool = False, language: str = "en",
                 mirror_endpoint: str = "") -> None:
        self.use_gpu = use_gpu
        self.device = _to_device(use_gpu)
        # 粤语等映射到对齐模型支持的语言
        self.language = _ALIGN_LANG_MAP.get(language, language)
        # HF 镜像 endpoint（如 https://hf-mirror.com），空则用默认 huggingface.co
        self.mirror_endpoint = (mirror_endpoint or "").strip().rstrip("/")
        self._model = None
        self._metadata = None
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    # ---------------- 模型加载 ----------------

    def ensure_model(
        self,
        progress_cb: Callable[[float], None] | None = None,
    ) -> None:
        """加载对齐模型；缺失则从 HuggingFace 下载（支持镜像 + 实时进度）。

        - torchaudio 自带语言（en/fr/de/es/it）：直接加载，torchaudio 自管下载
        - HuggingFace wav2vec2 模型：用 httpx 直接流式下载到 models/align_models/<lang>/，
          再用 load_align_model(model_name=本地路径) 从本地加载。
          不依赖 huggingface_hub.snapshot_download（镜像对大文件的 HEAD 请求不返回
          X-Repo-Commit header，会导致 FileMetadataError）。
        """
        if self._model is not None:
            if progress_cb:
                progress_cb(1.0)
            return
        if self.language not in SUPPORTED_ALIGN_LANGS:
            raise AlignmentError(
                f"语言 '{self.language}' 暂无对齐模型支持。"
                f"支持的语言：中文、英语、日语、韩语、粤语、法语、德语、西班牙语、"
                f"俄语、意大利语、葡萄牙语、越南语、印尼语、阿拉伯语等。"
            )

        # nltk punkt 分句器（alignment.py 依赖）
        self._ensure_nltk_punkt()

        # 1. torchaudio 自带语言：直接加载
        if self.language in _ALIGN_MODELS_TORCH:
            self._load_align_model_local(progress_cb)
            return

        # 2. HuggingFace 模型：先下载（带镜像 + 进度），再从本地加载
        model_name = _ALIGN_MODELS_HF.get(self.language)
        if not model_name:
            raise AlignmentError(f"语言 '{self.language}' 无对齐模型映射")

        # 已缓存则跳过下载
        if not is_align_model_cached(self.language):
            if progress_cb:
                progress_cb(0.0)
            self._download_model_files(model_name, progress_cb)

        # 从本地目录加载
        self._load_align_model_local(progress_cb)

    def _download_model_files(
        self,
        model_name: str,
        progress_cb: Callable[[float], None] | None,
    ) -> None:
        """用 httpx 直接流式下载模型文件到 models/align_models/<lang>/。

        绕过 huggingface_hub.snapshot_download，因为镜像对大文件的 HEAD 请求
        不返回 X-Repo-Commit header，导致 FileMetadataError。

        - 用 HfApi.model_info 获取文件列表（API 调用走镜像 endpoint）
        - 用 httpx HEAD 请求获取各文件大小（model_info 通过镜像返回的 size=None）
        - 用 httpx.stream() 逐文件下载，支持真实字节级进度与即时取消
        - 跳过已下载的完整文件（断点续传能力）
        - 排除其他框架权重（tf/flax），减少下载体积
        """
        try:
            from huggingface_hub import HfApi
        except ImportError as e:
            raise AlignmentError(
                f"缺少 huggingface_hub 依赖，无法下载对齐模型：{e}"
            ) from e

        import httpx

        endpoint = self.mirror_endpoint or "https://huggingface.co"
        endpoint = endpoint.rstrip("/")

        # 获取文件列表
        try:
            api = HfApi(endpoint=endpoint)
            info = api.model_info(model_name)
        except Exception as e:
            raise AlignmentError(
                f"获取模型信息失败（{model_name}）：{e}\n"
                f"可尝试更换镜像源后重试。"
            ) from e

        # 筛选需要下载的文件名
        filenames: list[str] = []
        for sib in info.siblings or []:
            name = sib.rfilename if hasattr(sib, "rfilename") else getattr(sib, "name", "")
            if name in _IGNORE_NAMES or name.endswith(_IGNORE_EXTS):
                continue
            if not name or name.startswith("."):
                continue
            filenames.append(name)

        local_dir = _get_align_model_dir(self.language)
        os.makedirs(local_dir, exist_ok=True)

        # 用 HEAD 请求获取各文件实际大小（model_info 通过镜像返回 size=None）
        file_sizes: dict[str, int] = {}
        for fname in filenames:
            if self._cancelled:
                raise AlignmentCancelled("用户取消对齐任务")
            url = f"{endpoint}/{model_name}/resolve/main/{fname}"
            try:
                r = httpx.head(url, follow_redirects=True, timeout=30)
                cl = r.headers.get("Content-Length")
                file_sizes[fname] = int(cl) if cl else 0
            except Exception:
                file_sizes[fname] = 0

        total_bytes = sum(file_sizes.values()) or 1
        downloaded_bytes = 0

        for filename in filenames:
            if self._cancelled:
                raise AlignmentCancelled("用户取消对齐任务")

            file_size = file_sizes.get(filename, 0)
            local_path = os.path.join(local_dir, filename)

            # 跳过已完整下载的文件
            if file_size > 0 and os.path.isfile(local_path) and os.path.getsize(local_path) == file_size:
                downloaded_bytes += file_size
                if progress_cb:
                    progress_cb(min(1.0, downloaded_bytes / total_bytes))
                continue

            url = f"{endpoint}/{model_name}/resolve/main/{filename}"

            try:
                with httpx.stream("GET", url, follow_redirects=True, timeout=300) as resp:
                    resp.raise_for_status()

                    # 写入临时文件，下载完成后重命名
                    tmp_path = local_path + ".tmp"
                    with open(tmp_path, "wb") as f:
                        for chunk in resp.iter_bytes(chunk_size=65536):
                            if self._cancelled:
                                f.close()
                                try:
                                    os.remove(tmp_path)
                                except OSError:
                                    pass
                                raise AlignmentCancelled("用户取消对齐任务")
                            f.write(chunk)
                            downloaded_bytes += len(chunk)
                            if progress_cb:
                                ratio = min(1.0, downloaded_bytes / total_bytes)
                                try:
                                    progress_cb(float(ratio))
                                except Exception:
                                    pass

                    # 下载完成，重命名临时文件
                    if os.path.exists(local_path):
                        os.remove(local_path)
                    os.rename(tmp_path, local_path)

            except AlignmentCancelled:
                raise
            except httpx.HTTPStatusError as e:
                raise AlignmentError(
                    f"下载文件失败（{filename}）：HTTP {e.response.status_code}\n"
                    f"可尝试更换镜像源后重试。"
                ) from e
            except Exception as e:
                if self._cancelled:
                    raise AlignmentCancelled("用户取消对齐任务") from e
                raise AlignmentError(
                    f"下载文件失败（{filename}）：{e}\n"
                    f"可尝试更换镜像源后重试。"
                ) from e

        # 下载完成
        if progress_cb:
            progress_cb(1.0)

    def _load_align_model_local(
        self,
        progress_cb: Callable[[float], None] | None,
    ) -> None:
        """从本地目录加载对齐模型。

        - torchaudio 语言：用 load_align_model 默认逻辑（torchaudio 自管下载）
        - HuggingFace 语言：传本地目录路径作为 model_name，from_pretrained 直接从本地加载
        """
        from whisperx.alignment import load_align_model

        # torchaudio 语言不需要本地目录
        if self.language in _ALIGN_MODELS_TORCH:
            cache_dir = _get_hf_hub_cache_dir()
            model_name = None
        else:
            # 传本地目录路径，from_pretrained 会直接从该目录加载
            model_name = _get_align_model_dir(self.language)
            cache_dir = None

        try:
            self._model, self._metadata = load_align_model(
                language_code=self.language,
                device=self.device,
                model_name=model_name,
                model_dir=cache_dir,
                model_cache_only=True if model_name is None else False,
            )
        except Exception as e:
            if self.use_gpu and _is_oom(e):
                # GPU 显存不足，降级到 CPU 重试
                self._fallback_to_cpu()
                self._model, self._metadata = load_align_model(
                    language_code=self.language,
                    device=self.device,
                    model_name=model_name,
                    model_dir=cache_dir,
                    model_cache_only=True if model_name is None else False,
                )
            else:
                raise AlignmentError(
                    f"加载对齐模型失败（{self.language}）：{e}\n"
                    f"首次使用需联网下载约 1GB 的 wav2vec2 对齐模型。"
                ) from e
        if progress_cb:
            progress_cb(1.0)

    @staticmethod
    def _ensure_nltk_punkt() -> None:
        """确保 nltk punkt 分句器可用，缺失则尝试下载。"""
        try:
            import nltk
            for pkg in ("punkt_tab", "punkt"):
                try:
                    nltk.data.find(f"tokenizers/{pkg}")
                except LookupError:
                    try:
                        nltk.download(pkg, quiet=True)
                    except Exception:
                        pass  # 离线环境静默跳过，alignment 内部有兜底
        except Exception:
            pass

    def _fallback_to_cpu(self) -> None:
        self.use_gpu = False
        self.device = "cpu"
        self._model = None
        try:
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # ---------------- 对齐 ----------------

    def align_lyrics(
        self,
        vocals_path: str,
        lyric_lines: list[str],
        total_duration: float,
        progress_cb: Callable[[float], None] | None = None,
    ) -> list[LyricLine]:
        """将歌词文本对齐到人声轨，返回 LyricLine 列表。

        参数：
            vocals_path: 人声轨 wav 路径（分离后或原始转码后）
            lyric_lines: 歌词文本列表，每行一句（已按行分隔）
            total_duration: 音频总时长（秒），用于 segment 范围
            progress_cb: 对齐进度回调，参数 0~1

        对齐策略（关键）：
        - whisperx.align 只会在 segment 的 [start, end] 范围内切片音频做对齐，
          不会在音频中"搜索"歌词位置。
        - 旧实现按行数等分总时长作为每行的 segment 范围，导致前奏段被强行对齐
          到第一行歌词，时间轴完全错位（人声还没出现歌词就开始）。
        - 新实现把所有歌词拼成一个完整文本，作为一个覆盖整个音频的大 segment，
          让 whisperx 在整个音频上用 CTC 强制对齐找最佳路径，再按每行字/词数
          从返回的字/词级时间戳中分组重建每行。
        """
        if not os.path.isfile(vocals_path):
            raise AlignmentError(f"音频文件不存在：{vocals_path}")
        if self._model is None or self._metadata is None:
            raise AlignmentError("对齐模型未加载，请先调用 ensure_model()")

        # 过滤空行，保留非空歌词
        texts = [t.strip() for t in lyric_lines if t and t.strip()]
        if not texts:
            raise AlignmentError("歌词文本为空，请输入或导入歌词后再对齐")

        # 把所有歌词拼成一个完整文本，让 whisperx 在整个音频上对齐
        # 中日文（无空格语言）直接拼接，其他语言用空格分隔
        if self.language in _LANGS_WITHOUT_SPACES:
            full_text = "".join(texts)
        else:
            full_text = " ".join(texts)

        # 作为一个覆盖整个音频的大 segment
        # 注意：whisperx.align 会在 [0, total_duration] 范围内对整个文本做 CTC 对齐，
        # 找出每个字/词在音频中的最佳位置，从而避免等分时长导致的错位
        end_time = float(total_duration) if total_duration > 0 else 30.0
        segments: list[dict] = [{
            "start": 0.0,
            "end": end_time,
            "text": full_text,
        }]

        from whisperx.alignment import align as wx_align

        def _progress(p: float) -> None:
            if self._cancelled:
                raise AlignmentCancelled("用户取消对齐任务")
            if progress_cb:
                try:
                    progress_cb(float(p))
                except Exception:
                    pass

        try:
            result = wx_align(
                transcript=segments,
                model=self._model,
                align_model_metadata=self._metadata,
                audio=vocals_path,
                device=self.device,
                progress_callback=_progress,
            )
        except AlignmentCancelled:
            raise
        except Exception as e:
            if self._cancelled:
                raise AlignmentCancelled("用户取消对齐任务")
            raise AlignmentError(f"对齐失败：{e}") from e

        if self._cancelled:
            raise AlignmentCancelled("用户取消对齐任务")

        return self._build_lines(texts, result, total_duration)

    def _build_lines(
        self,
        texts: list[str],
        result,
        total_duration: float,
    ) -> list[LyricLine]:
        """从 align 返回结果构建 LyricLine 列表。

        新策略：所有歌词拼成一个 segment 对齐，whisperx 返回所有字/词的时间戳。
        按"每行的字/词数"从返回的时间戳序列中顺序切片，重建每行的时间戳。
        - 中日文（LANGUAGES_WITHOUT_SPACES）：whisperx 对每个字符生成一个 word，
          按每行的非空格字符数切片
        - 其他语言：whisperx 按空格切分 word，按每行的词数切片
        """
        aligned_segs = result.get("segments", []) if result else []
        # 取第一个（也是唯一一个）segment 的所有 word 级时间戳
        all_words: list[dict] = []
        if aligned_segs:
            all_words = aligned_segs[0].get("words", []) or []

        lines: list[LyricLine] = []
        word_idx = 0
        n = len(texts)
        # fallback：对齐失败的行用等分时长兜底
        per_line = (total_duration / n) if total_duration > 0 and n > 0 else 30.0

        for i, text in enumerate(texts):
            # 计算这一行应该对应多少个 word
            if self.language in _LANGS_WITHOUT_SPACES:
                # 中日文：按非空格字符数（whisperx 把每个字符作为一个 word）
                word_count = len([c for c in text if not c.isspace()])
            else:
                # 其他语言：按空格切分的词数
                word_count = len(text.split())

            # 从 all_words 中顺序切片
            line_words = all_words[word_idx:word_idx + word_count] if word_count > 0 else []
            word_idx += word_count

            # 计算行级时间戳
            starts = [float(w["start"]) for w in line_words
                      if w.get("start") is not None and w["start"] == w["start"]]
            ends = [float(w["end"]) for w in line_words
                    if w.get("end") is not None and w["end"] == w["end"]]

            if starts and ends:
                start = min(starts)
                end = max(ends)
            else:
                # fallback：等分时长
                start = float(i * per_line)
                end = float((i + 1) * per_line)

            if end < start:
                end = start + 1.0  # 至少 1 秒，避免 0 长度行

            # 构建 word 级时间戳
            words = self._build_words(text, start, end, line_words)
            lines.append(LyricLine(
                text=text,
                start=start,
                end=end,
                words=words,
            ))
        return lines

    def _build_words(self, text: str, start: float, end: float, line_words) -> list[WordTS]:
        """构建词级时间戳列表。

        - 有对齐结果：直接用 align 返回的 word 级时间戳（中日文为字级，其他为词级）
        - 无对齐结果（fallback）：中日文按字等分行时间范围；其他语言整行一个 word
        """
        if line_words:
            out: list[WordTS] = []
            for w in line_words:
                wtext = (w.get("word") or "").strip()
                if not wtext:
                    continue
                ws = float(w.get("start") or start)
                we = float(w.get("end") or end)
                if ws != ws:  # NaN
                    ws = start
                if we != we:
                    we = end
                out.append(WordTS(wtext, ws, we))
            return out

        # fallback：对齐失败的行，按字等分
        if self.language in _LANGS_WITHOUT_SPACES:
            chars = [c for c in text if not c.isspace()]
            if not chars or end <= start:
                return []
            dur = (end - start) / len(chars)
            return [
                WordTS(ch, start + k * dur, start + (k + 1) * dur)
                for k, ch in enumerate(chars)
            ]
        # 其他语言 fallback：整行作为一个 word
        return [WordTS(text, start, end)] if text else []

    # ---------------- 资源释放 ----------------

    def release(self) -> None:
        """释放模型显存/内存（顺序与 VocalSeparator.release 一致）。"""
        model = self._model
        self._model = None
        try:
            if model is not None:
                model.to("cpu")
        except Exception:
            pass
        del model
        try:
            from ..utils.memory import release_engine_memory
            release_engine_memory(use_gpu=self.use_gpu)
        except Exception:
            try:
                import gc
                gc.collect()
                if self.use_gpu and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except Exception:
                pass


def is_align_supported(language: str | None) -> bool:
    """检查语言是否支持对齐（用于 UI 提示）。"""
    if not language:
        return False
    lang = _ALIGN_LANG_MAP.get(language, language)
    return lang in SUPPORTED_ALIGN_LANGS


def get_align_model_name(language: str) -> str | None:
    """返回语言对应的对齐模型名（用于 UI 显示状态）。

    使用本模块硬编码的 _ALIGN_MODELS_TORCH / _ALIGN_MODELS_HF 映射，
    避免每次调用都 `from whisperx.alignment import ...` 触发重模块加载导致 UI 卡顿。
    """
    lang = _ALIGN_LANG_MAP.get(language, language)
    if lang in _ALIGN_MODELS_TORCH:
        return _ALIGN_MODELS_TORCH[lang]
    if lang in _ALIGN_MODELS_HF:
        return _ALIGN_MODELS_HF[lang]
    return None


def is_align_model_cached(language: str) -> bool:
    """检测当前语言的对齐模型是否已缓存到本地。

    - torchaudio 自带语言（en/fr/de/es/it）始终视为已缓存（无需下载）
    - HuggingFace 模型检测关键文件（config.json + 权重文件）是否在本地目录
      models/align_models/<lang>/
    """
    lang = _ALIGN_LANG_MAP.get(language, language)
    if lang in _ALIGN_MODELS_TORCH:
        return True  # torchaudio 自带
    if lang not in _ALIGN_MODELS_HF:
        return False  # 不支持的语言
    local_dir = _get_align_model_dir(lang)
    if not os.path.isdir(local_dir):
        return False
    has_config = os.path.isfile(os.path.join(local_dir, "config.json"))
    # 权重文件可能是 pytorch_model.bin / model.safetensors 等
    weight_files = [
        f for f in os.listdir(local_dir)
        if f.endswith((".safetensors", ".bin")) or "model" in f.lower()
    ]
    return has_config and bool(weight_files)


def _get_hf_hub_cache_dir() -> str | None:
    """获取 huggingface_hub 实际使用的缓存目录。

    优先用环境变量（src/__init__.py 顶部的 setup_hf_env() 已设置为项目目录
    models/hf/hub），避免首次 import huggingface_hub.constants 导致的卡顿。
    仅当环境变量缺失时才 fallback 到 huggingface_hub 常量。
    """
    env = os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if env:
        return env
    try:
        import huggingface_hub.constants as _c
        const = getattr(_c, "HUGGINGFACE_HUB_CACHE", None)
        if const:
            return const
    except Exception:
        pass
    return None
