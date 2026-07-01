"""GPU 环境检测：检测 NVIDIA 显卡 + torch 自带 CUDA DLL 是否齐全。

判定规则：两者都满足才认为 GPU 可用，否则视为不可用。
- CUDA 12 + cuDNN 9 严格对齐 ctranslate2 新版要求。
- CUDA DLL 直接从 torch+cu126 自带的 lib/ 目录加载，无需额外目录。
"""
import os
import subprocess
from typing import NamedTuple


class GpuStatus(NamedTuple):
    available: bool
    has_nvidia: bool
    cuda_libs_ok: bool  # CUDA DLL 是否齐全（保留字段名以兼容 UI 代码）
    gpu_name: str
    driver_version: str
    cuda_version: str
    missing_dlls: list[str]
    reason: str


# ctranslate2 CUDA 12 + cuDNN 9 所需的核心 DLL
# 任一缺失即视为 GPU 不可用，避免运行时崩溃
REQUIRED_CUDA_DLLS = [
    "cudart64_12.dll",        # CUDA Runtime
    "cublas64_12.dll",        # cuBLAS
    "cublasLt64_12.dll",      # cuBLAS Lt
    "cufft64_11.dll",        # cuFFT
    "curand64_10.dll",        # cuRAND
]

REQUIRED_CUDNN_DLLS = [
    # cuDNN 9 系列
    "cudnn64_9.dll",
    "cudnn_engines_precompiled64_9.dll",
    "cudnn_graph64_9.dll",
    "cudnn_heuristic64_9.dll",
    "cudnn_ops64_9.dll",
]


def _detect_nvidia_via_pynvml() -> tuple[bool, str, str, str]:
    """优先用 pynvml 检测 NVIDIA 显卡。返回 (found, name, driver, cuda_ver)。"""
    try:
        import pynvml  # type: ignore
        try:
            pynvml.nvmlInit()
        except Exception:
            return (False, "", "", "")
        try:
            count = pynvml.nvmlDeviceGetCount()
            if count <= 0:
                return (False, "", "", "")
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="ignore")
            driver = pynvml.nvmlSystemGetDriverVersion()
            if isinstance(driver, bytes):
                driver = driver.decode("utf-8", errors="ignore")
            # 通过 driver 版本推断 CUDA 版本上限（粗略，仅作展示）
            cuda_ver = ""
            try:
                cuda_ver = pynvml.nvmlSystemGetCudaDriverVersion()
                # 返回整数，例如 12000 -> 12.0
                if isinstance(cuda_ver, int):
                    cuda_ver = f"{cuda_ver // 1000}.{(cuda_ver // 10) % 100}"
            except Exception:
                cuda_ver = ""
            return (True, str(name), str(driver), str(cuda_ver))
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
    except Exception:
        return (False, "", "", "")


def _detect_nvidia_via_smi() -> tuple[bool, str, str, str]:
    """用 nvidia-smi 命令检测。返回 (found, name, driver, cuda_ver)。"""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version",
             "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW,
            timeout=8,
        )
        text = out.decode("utf-8", errors="ignore").strip()
        if not text:
            return (False, "", "", "")
        name, driver = (text.split(",") + ["", ""])[:2]
        name, driver = name.strip(), driver.strip()
        # 查 CUDA 版本
        cuda_ver = ""
        try:
            out2 = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version",
                 "--format=csv,noheader"],
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW,
                timeout=8,
            )
        except Exception:
            pass
        return (True, name, driver, cuda_ver)
    except Exception:
        return (False, "", "", "")


def _get_torch_cuda_lib_dir() -> str | None:
    """返回 torch CUDA 包的 lib 目录（包含 CUDA 12 + cuDNN 9 DLL）。

    torch+cu126 安装后自带这些 DLL，无需额外目录。
    """
    try:
        import torch
        lib_dir = os.path.join(os.path.dirname(torch.__file__), "lib")
        if os.path.isdir(lib_dir):
            return lib_dir
    except Exception:
        pass
    return None


def _check_cuda_libs() -> tuple[bool, list[str], str]:
    """检查 torch 自带的 CUDA 12 + cuDNN 9 DLL 是否齐全。

    返回 (ok, missing_list, source_dir)。source_dir 为 torch/lib/ 路径。
    """
    torch_lib = _get_torch_cuda_lib_dir()
    if not torch_lib:
        return (False, REQUIRED_CUDA_DLLS + REQUIRED_CUDNN_DLLS, "")

    missing = []
    for dll in REQUIRED_CUDA_DLLS + REQUIRED_CUDNN_DLLS:
        if not os.path.isfile(os.path.join(torch_lib, dll)):
            missing.append(dll)
    if missing:
        return (False, missing, torch_lib)
    return (True, [], torch_lib)


def detect_gpu() -> GpuStatus:
    """主入口：检测 GPU 环境是否可用。"""
    # 1) 检测 NVIDIA 显卡
    found, name, driver, cuda_ver = _detect_nvidia_via_pynvml()
    if not found:
        found, name, driver, cuda_ver = _detect_nvidia_via_smi()

    # 2) 检查 torch 自带 CUDA DLL 是否齐全
    cuda_ok, missing, src_dir = _check_cuda_libs()

    available = found and cuda_ok

    reasons = []
    if not found:
        reasons.append("未检测到 NVIDIA 显卡")
    if not cuda_ok:
        src_label = "torch/lib"
        reasons.append(f"CUDA DLL 不完整（{src_label}）：{', '.join(missing[:5])}")
    if not reasons:
        reasons.append("GPU 环境就绪")

    return GpuStatus(
        available=available,
        has_nvidia=found,
        cuda_libs_ok=cuda_ok,
        gpu_name=name,
        driver_version=driver,
        cuda_version=cuda_ver,
        missing_dlls=missing,
        reason="；".join(reasons),
    )


def ensure_cuda_libs_on_path() -> bool:
    """将 torch 自带的 CUDA DLL 目录加入 PATH，便于 DLL 加载。

    返回是否成功加入。在启动时调用一次。
    """
    ok, _, src_dir = _check_cuda_libs()
    if not ok or not src_dir:
        return False
    sep = os.pathsep
    cur = os.environ.get("PATH", "")
    if src_dir not in cur.split(sep):
        os.environ["PATH"] = src_dir + sep + cur
    return True
