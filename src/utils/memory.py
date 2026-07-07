"""MKL/PyTorch 内存释放辅助。

解决场景：
- demucs (PyTorch + MKL 后端) 加载并释放后，MKL 内存池仍保留大量缓冲区不归还系统，
  导致后续 faster-whisper (ctranslate2) 加载模型时分配大块连续内存失败：
  "mkl malloc:failed to allocate memory"

提供两个能力：
- free_mkl_buffers(): 通过 ctypes 调用 mkl_free_buffers() 释放 MKL 内存池
- release_engine_memory(): 业务层统一入口，依次调用 gc.collect + empty_cache + free_mkl_buffers
"""
from __future__ import annotations

import ctypes
import os
from typing import Optional


# 缓存已加载的 MKL 共享库句柄（首次成功后复用，避免重复 dlopen）
_mkl_lib: Optional[ctypes.CDLL] = None
_mkl_checked = False


def _find_mkl_library() -> Optional[ctypes.CDLL]:
    """尝试加载 MKL 共享库（mkl_rt.dll / libmkl_rt.so）。

    优先从 torch/lib 目录加载（PyTorch 在 Windows 下静态链接 MKL 到 torch_cpu.dll，
    但部分版本仍提供独立 mkl_rt.dll），其次从系统 PATH 加载。
    """
    global _mkl_lib, _mkl_checked
    if _mkl_checked:
        return _mkl_lib
    _mkl_checked = True

    candidates: list[str] = []

    # 1. torch/lib 目录（GPU 版 torch 自带）
    try:
        import torch as _torch
        torch_lib = os.path.join(os.path.dirname(_torch.__file__), "lib")
        if os.path.isdir(torch_lib):
            for name in ("mkl_rt.2.dll", "mkl_rt.1.dll", "mkl_rt.dll",
                         "libmkl_rt.so.2", "libmkl_rt.so.1", "libmkl_rt.so",
                         "libmkl_intel_lp64.so", "libmkl_intel_lp64.dylib"):
                p = os.path.join(torch_lib, name)
                if os.path.isfile(p):
                    candidates.append(p)
    except Exception:
        pass

    # 2. numpy 内置 MKL（如果是 mkl 版本）
    try:
        import numpy as _np
        np_root = os.path.dirname(_np.__file__)
        for sub in (".", "core", "DLLs"):
            d = os.path.join(np_root, sub) if sub else np_root
            if not os.path.isdir(d):
                continue
            for name in ("mkl_rt.2.dll", "mkl_rt.1.dll", "mkl_rt.dll",
                          "libmkl_rt.so.2", "libmkl_rt.so.1", "libmkl_rt.so"):
                p = os.path.join(d, name)
                if os.path.isfile(p):
                    candidates.append(p)
    except Exception:
        pass

    # 3. 系统搜索（依赖 PATH / LD_LIBRARY_PATH）
    for name in ("mkl_rt.2.dll", "mkl_rt.1.dll", "mkl_rt.dll",
                 "libmkl_rt.so.2", "libmkl_rt.so.1", "libmkl_rt.so",
                 "libmkl_intel_lp64.dylib"):
        candidates.append(name)

    for cand in candidates:
        try:
            lib = ctypes.CDLL(cand)
            # 验证 mkl_free_buffers 符号存在
            if not hasattr(lib, "mkl_free_buffers"):
                continue
            _mkl_lib = lib
            return lib
        except OSError:
            continue
    return None


def free_mkl_buffers() -> None:
    """释放 MKL 内存池中已释放但未归还系统的缓冲区。

    对应 C 接口：void mkl_free_buffers(void);
    幂等：可多次调用，无副作用。无 MKL 库时静默跳过。
    """
    lib = _find_mkl_library()
    if lib is None:
        return
    try:
        # void mkl_free_buffers(void)
        lib.mkl_free_buffers()
    except Exception:
        # 调用失败不抛出，避免影响主流程
        pass


def release_engine_memory(use_gpu: bool = False) -> None:
    """业务层统一内存释放入口。

    顺序（关键，不能调换）：
    1. gc.collect() 触发 Python 对象析构（包括 torch.Tensor）
    2. torch.cuda.empty_cache() 释放 GPU 显存（仅 GPU 模式）
    3. free_mkl_buffers() 释放 MKL 内存池

    通常在 VocalSeparator.release() / Aligner.release() 后调用，
    再加载下一个模型（如 faster-whisper）之前。
    """
    try:
        import gc
        gc.collect()
    except Exception:
        pass

    if use_gpu:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:
            pass

    free_mkl_buffers()
