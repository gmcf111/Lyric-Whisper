"""ffmpeg 调用工具。

路径解析顺序：
1. 程序目录下的 ffmpeg/ffmpeg.exe（绿色便携，优先）
2. 系统 PATH 环境变量中的 ffmpeg（用户已安装则直接复用，无需额外下载）
两者都找不到才报错。
"""
import os
import shutil
import subprocess
from typing import Callable

from .config import get_config


class FFmpegError(Exception):
    pass


def _find_in_path(name: str) -> str | None:
    """在系统 PATH 环境变量中查找可执行文件，返回完整路径或 None。"""
    # shutil.which 已跨平台处理扩展名（Windows 上会自动尝试 .exe/.bat 等）
    return shutil.which(name)


def get_ffmpeg_path() -> str:
    """返回 ffmpeg 可执行文件路径。

    优先使用程序目录下的 ffmpeg/ffmpeg.exe；若不存在，则回退到系统 PATH
    中的 ffmpeg（用户已安装环境变量时直接复用，无需下载）。两者都找不到才抛异常。
    """
    # 1) 本地绿色版
    local = get_config().ffmpeg_exe
    if os.path.isfile(local):
        return local
    # 2) 系统 PATH
    found = _find_in_path("ffmpeg")
    if found:
        return found
    raise FFmpegError(
        "未找到 ffmpeg：程序目录下的 ffmpeg/ 文件夹中没有 ffmpeg.exe，"
        "系统 PATH 环境变量中也没有 ffmpeg。\n"
        "请将 ffmpeg.exe 放入 ffmpeg/ 文件夹，或安装 ffmpeg 并加入系统环境变量。"
    )


def get_ffprobe_path() -> str | None:
    """返回 ffprobe 路径（本地优先，其次系统 PATH），均不存在返回 None。"""
    local = get_config().ffprobe_exe
    if os.path.isfile(local):
        return local
    return _find_in_path("ffprobe")


def get_ffmpeg_info() -> tuple[str | None, str]:
    """返回 (ffmpeg路径, 来源)。

    来源：'local' = 程序目录自带；'system' = 系统 PATH；None = 未找到。
    供界面提示用户当前用的是哪个 ffmpeg。
    """
    local = get_config().ffmpeg_exe
    if os.path.isfile(local):
        return local, "local"
    found = _find_in_path("ffmpeg")
    if found:
        return found, "system"
    return None, "none"


def probe_duration(file_path: str) -> float:
    """获取媒体时长（秒）。优先 ffprobe，否则用 ffmpeg 解析 stderr。"""
    ffprobe = get_ffprobe_path()
    if ffprobe:
        try:
            out = subprocess.check_output(
                [
                    ffprobe, "-v", "error", "-show_entries",
                    "format=duration", "-of",
                    "default=noprint_wrappers=1:nokey=1", file_path,
                ],
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            return float(out.decode().strip())
        except Exception:
            pass
    # 降级：解析 ffmpeg 输出
    duration = 0.0
    try:
        proc = subprocess.run(
            [get_ffmpeg_path(), "-i", file_path],
            stderr=subprocess.PIPE, stdout=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        text = proc.stderr.decode("utf-8", errors="ignore")
        import re
        m = re.search(r"Duration:\s(\d+):(\d+):(\d+(?:\.\d+)?)", text)
        if m:
            duration = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    except Exception:
        pass
    return duration


def convert_to_wav(
    input_path: str,
    output_path: str,
    sample_rate: int = 44100,
    progress_cb: Callable[[float], None] | None = None,
    cancel_flag: Callable[[], bool] | None = None,
) -> None:
    """将任意音视频文件转为 WAV（单声道/立体声兼容）。

    progress_cb: 0.0~1.0 进度回调
    cancel_flag: 返回 True 时中止
    """
    ffmpeg = get_ffmpeg_path()
    total = probe_duration(input_path) or 0.0

    cmd = [
        ffmpeg, "-y", "-i", input_path,
        "-vn",  # 去掉视频
        "-ac", "2", "-ar", str(sample_rate),
        "-acodec", "pcm_s16le",
        output_path,
    ]

    proc = subprocess.Popen(
        cmd,
        stderr=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        universal_newlines=True,
        encoding="utf-8",
        errors="ignore",
        creationflags=subprocess.CREATE_NO_WINDOW,
    )

    import re
    time_pat = re.compile(r"time=(\d+):(\d+):(\d+(?:\.\d+)?)")
    try:
        assert proc.stderr is not None
        for line in proc.stderr:
            if cancel_flag and cancel_flag():
                proc.terminate()
                raise FFmpegError("用户取消任务")
            m = time_pat.search(line)
            if m and total > 0 and progress_cb:
                cur = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
                progress_cb(min(cur / total, 1.0))
    finally:
        proc.wait()
    if proc.returncode not in (0, None):
        raise FFmpegError(f"ffmpeg 转码失败，返回码 {proc.returncode}")
    if progress_cb:
        progress_cb(1.0)
