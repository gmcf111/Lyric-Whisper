"""人声分离：调用 demucs (htdemucs) 分离出人声轨。

使用稳定的低层 API：demucs.pretrained.get_model + demucs.apply.apply_model，
避免 separate_audio_file 在不同版本间返回签名不一致的问题。
"""
import os
from typing import Callable

import torch

from ..utils.config import get_config


class SeparationError(Exception):
    pass


class SeparationCancelled(SeparationError):
    pass


def _to_device(use_gpu: bool) -> str:
    return "cuda" if use_gpu else "cpu"


# Demucs htdemucs 模型（Meta 官方 CDN）
_DEMUCS_CKPT_URL = (
    "https://dl.fbaipublicfiles.com/demucs/"
    "hybrid_transformer/955717e8-8726e21a.th"
)
_DEMUCS_CKPT_NAME = "955717e8-8726e21a.th"


def _demucs_checkpoint_path() -> str:
    """返回 demucs 检查点缓存路径（与界面下载逻辑一致）。"""
    ckpt_dir = os.path.join(torch.hub.get_dir(), "checkpoints")
    return os.path.join(ckpt_dir, _DEMUCS_CKPT_NAME)



class VocalSeparator:
    """封装 demucs 分离逻辑，支持进度回调与取消。"""

    def __init__(self, use_gpu: bool = False) -> None:
        self.use_gpu = use_gpu
        self.device = _to_device(use_gpu)
        self._model_name = get_config().demucs_model_name or "htdemucs"
        self._model = None
        self._cancelled = False
        self._progress_cb: Callable[[float], None] | None = None

    def cancel(self) -> None:
        self._cancelled = True

    def ensure_model(
        self,
        progress_cb: Callable[[float], None] | None = None,
    ) -> None:
        """确保 demucs 模型已缓存，缺失则自动下载。

        progress_cb: 下载进度回调，参数为 0~1。
        下载中若 self._cancelled 置位则抛出 SeparationCancelled。
        """
        ckpt = _demucs_checkpoint_path()
        # 已存在且大小合理则视为可用
        if os.path.isfile(ckpt) and os.path.getsize(ckpt) >= 1_000_000:
            return

        # 清理可能残留的损坏/临时文件
        self._cleanup_corrupt_checkpoint()

        os.makedirs(os.path.dirname(ckpt), exist_ok=True)
        tmp = ckpt + ".tmp"
        try:
            import httpx
            from ..utils.hf_session import make_hf_client
        except Exception as e:
            raise SeparationError(f"初始化下载组件失败：{e}") from e

        client = make_hf_client()
        try:
            stream_timeout = httpx.Timeout(connect=10, read=30, write=10, pool=10)
            with client.stream(
                "GET", _DEMUCS_CKPT_URL,
                timeout=stream_timeout, follow_redirects=True,
            ) as resp:
                resp.raise_for_status()
                total = int(resp.headers.get("Content-Length", 0) or 0)
                downloaded = 0
                with open(tmp, "wb") as f:
                    for chunk in resp.iter_bytes(1024 * 256):
                        if self._cancelled:
                            raise SeparationCancelled("用户取消分离任务")
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total > 0 and progress_cb:
                            progress_cb(downloaded / total)
            os.replace(tmp, ckpt)
        except SeparationCancelled:
            self._safe_remove(tmp)
            raise
        except Exception as e:
            self._safe_remove(tmp)
            raise SeparationError(
                f"人声分离模型自动下载失败，请检查网络连接后重试。\n原始错误：{e}"
            ) from e
        finally:
            try:
                client.close()
            except Exception:
                pass

    @staticmethod
    def _safe_remove(path: str) -> None:
        try:
            if os.path.isfile(path):
                os.remove(path)
        except Exception:
            pass

    def _load_model(self):
        if self._model is not None:
            return self._model
        from demucs.pretrained import get_model
        try:
            model = get_model(self._model_name)
        except Exception as e:
            # 加载失败可能是缓存文件损坏（如自动下载中断），
            # 清理损坏的检查点后给出明确提示，避免后续永久报错。
            self._cleanup_corrupt_checkpoint()
            raise SeparationError(
                f"加载 Demucs 模型失败（缓存可能已损坏并已清理）。\n"
                f"请通过界面「下载模型」按钮重新下载。\n原始错误：{e}"
            ) from e
        model.to(self.device)
        model.eval()
        self._model = model
        return model

    def _cleanup_corrupt_checkpoint(self) -> None:
        """清理可能损坏的 demucs 模型缓存文件。"""
        try:
            import torch
            hub_dir = torch.hub.get_dir()
            ckpt_dir = os.path.join(hub_dir, "checkpoints")
            if os.path.isdir(ckpt_dir):
                for name in os.listdir(ckpt_dir):
                    if name.endswith(".th"):
                        f = os.path.join(ckpt_dir, name)
                        # 过小的文件几乎肯定是损坏的（完整模型约 80MB）
                        if os.path.getsize(f) < 1_000_000:
                            os.remove(f)
        except Exception:
            pass

    def _make_callback(self):
        """构造传给 apply_model 的回调。不同 demucs 版本签名兼容。"""
        def cb(*args, **kwargs):
            if self._cancelled:
                raise SeparationCancelled("用户取消分离任务")
            progress = None
            if args:
                a = args[0]
                if isinstance(a, dict):
                    progress = a.get("progress", a.get("ratio"))
                elif isinstance(a, (int, float)):
                    progress = a
            if progress is None:
                progress = kwargs.get("progress")
            if progress is not None and self._progress_cb:
                try:
                    self._progress_cb(float(progress))
                except Exception:
                    pass
        return cb

    def separate(
        self,
        input_path: str,
        output_vocals: str,
        progress_cb: Callable[[float], None] | None = None,
    ) -> str:
        """分离人声，写入 output_vocals（wav）。返回人声文件路径。"""
        if not os.path.isfile(input_path):
            raise SeparationError(f"输入文件不存在：{input_path}")

        self._progress_cb = progress_cb
        self._cancelled = False

        try:
            import torchaudio
            import inspect
            from demucs.apply import apply_model
        except Exception as e:
            raise SeparationError(f"加载 demucs/torchaudio 失败：{e}") from e

        model = self._load_model()
        target_sr = int(getattr(model, "samplerate", 44100))

        # 检测 apply_model 是否支持 callback 参数（不同 demucs 版本）
        try:
            _supports_cb = "callback" in inspect.signature(apply_model).parameters
        except (ValueError, TypeError):
            _supports_cb = False

        # 1) 读取音频并重采样到模型采样率
        try:
            wav, sr0 = torchaudio.load(input_path)
        except Exception as e:
            # torchaudio 后端失败时尝试 soundfile
            try:
                import soundfile as sf
                import numpy as np
                data, sr0 = sf.read(input_path, always_2d=True)
                wav = torch.from_numpy(data.T).float()
            except Exception as e2:
                raise SeparationError(f"读取音频失败：{e2}") from e2

        if sr0 != target_sr:
            resampler = torchaudio.transforms.Resample(sr0, target_sr)
            wav = resampler(wav)
        wav = wav.float().to(self.device)

        # 2) 归一化（demucs 标准做法）
        ref = wav.mean(0)
        wav_in = (wav - ref.mean()) / ref.std()

        # 3) apply_model: 输入 [batch, channels, samples]
        cb = self._make_callback()
        kwargs = dict(shifts=1, overlap=0.25, split=True, progress=False, device=self.device)
        if _supports_cb:
            kwargs["callback"] = cb
        out = apply_model(model, wav_in.unsqueeze(0), **kwargs)[0]  # [sources, channels, samples]

        if self._cancelled:
            raise SeparationCancelled("用户取消分离任务")

        # 反归一化
        out = out * ref.std() + ref.mean()

        # 4) 取出人声轨
        source_names = getattr(model, "sources", ["drums", "bass", "other", "vocals"])
        if "vocals" not in source_names:
            raise SeparationError(f"模型未包含人声轨，sources={source_names}")
        vocal_idx = source_names.index("vocals")
        vocals = out[vocal_idx]  # [channels, samples]

        # 5) 保存为 wav
        os.makedirs(os.path.dirname(os.path.abspath(output_vocals)), exist_ok=True)
        self._save_wav(vocals, output_vocals, target_sr)

        # 显式释放 GPU 张量与 MKL 内存池，避免后续转写时出现
        # "mkl malloc:failed to allocate memory"（demucs 释放后 MKL 内存未归还系统）
        del out, vocals, wav_in, wav
        try:
            from ..utils.memory import release_engine_memory
            release_engine_memory(use_gpu=self.use_gpu)
        except Exception:
            try:
                import gc
                gc.collect()
                if self.use_gpu:
                    torch.cuda.empty_cache()
            except Exception:
                pass

        if progress_cb:
            progress_cb(1.0)
        return output_vocals

    def _save_wav(self, tensor: "torch.Tensor", path: str, sample_rate: int) -> None:
        import soundfile as sf
        import numpy as np
        wav = tensor.detach().cpu().numpy()
        if wav.ndim == 2:
            wav = wav.T  # to [samples, channels]
        wav = np.ascontiguousarray(wav, dtype="float32")
        sf.write(path, wav, sample_rate, subtype="PCM_16")

    def release(self) -> None:
        """释放模型显存/内存。

        关键顺序（缺一不可，否则显存无法真正归还）：
        1. 把模型搬回 CPU，斩断其在显存上的权重占用
           （self._model = None 只断开一个引用，若模型仍在 GPU 上且被
            内部子模块引用，引用计数不归零，显存不会释放）。
        2. 置空引用后 gc.collect() 强制析构模型对象。
        3. empty_cache() 只能回收「已无 Python 引用」的显存块，
           必须在 gc 之后调用才有效。
        4. synchronize() 确保释放在下一个模型加载前真正完成。
        5. free_mkl_buffers() 释放 MKL 内存池中保留的缓冲区，
           避免 demucs 释放后加载 faster-whisper 时出现
           "mkl malloc:failed to allocate memory"。
        """
        model = self._model
        self._model = None
        try:
            if model is not None:
                model.to("cpu")  # 搬回内存，切断显存权重占用
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
