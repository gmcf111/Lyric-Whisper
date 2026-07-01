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

    def _load_model(self):
        if self._model is not None:
            return self._model
        from demucs.pretrained import get_model
        model = get_model(self._model_name)
        model.to(self.device)
        model.eval()
        self._model = model
        return model

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
        """释放模型显存/内存。"""
        self._model = None
        if self.use_gpu:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
