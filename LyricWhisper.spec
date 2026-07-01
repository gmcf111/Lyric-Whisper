# -*- mode: python ; coding: utf-8 -*-
"""LyricWhisper PyInstaller 打包配置（onedir 模式）。

构建命令：
    pyinstaller LyricWhisper.spec --noconfirm

产物：
    dist/LyricWhisper/   解压即用的文件夹

说明：
- 统一打包 GPU 版 torch（CUDA 12.6 + cuDNN 9 DLL 由 collect_all("torch") 自动收集）。
- 没有 NVIDIA 显卡的机器也能运行（程序自动降级为 CPU 模式）。
- ffmpeg.exe / ffprobe.exe 一并打包进 _internal/ffmpeg/，无需用户单独安装。
- 除 Whisper / Demucs 模型权重外的所有运行依赖均已内置。
"""
import os
import sys

from PyInstaller.utils.hooks import collect_all

block_cipher = None

PROJECT_DIR = os.path.dirname(os.path.abspath(SPEC))

# ---- 收集重型依赖（hidden imports + datas + binaries）----
datas = []
binaries = []
hiddenimports = []

HEAVY_PACKAGES = [
    "numpy",
    "torch", "torchaudio",
    "demucs", "julius", "openunmix",
    "faster_whisper", "ctranslate2",
    "soundfile", "tokenizers", "onnxruntime",
    "lameenc",
    "PySide6",
    "truststore", "httpx", "huggingface_hub",
]
for pkg in HEAVY_PACKAGES:
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception as e:
        print(f"[warn] collect_all({pkg}) failed: {e}", file=sys.stderr)

# ---- 源码包 ----
datas.append((os.path.join(PROJECT_DIR, "src"), "src"))

# ---- ffmpeg.exe / ffprobe.exe（打包进 _internal/ffmpeg/）----
ffmpeg_dir = os.path.join(PROJECT_DIR, "ffmpeg")
if os.path.isdir(ffmpeg_dir):
    for name in ("ffmpeg.exe", "ffprobe.exe"):
        p = os.path.join(ffmpeg_dir, name)
        if os.path.isfile(p):
            datas.append((p, "ffmpeg"))
else:
    print("[warn] ffmpeg/ 目录不存在，exe 将缺少内置 ffmpeg", file=sys.stderr)

# ---- CUDA 库由 torch/lib/ 自动收集（collect_all("torch") 已包含）----
# 无需手动收集；torch+cu126 安装时自带 CUDA 12.6 + cuDNN 9 DLL

# ---- 额外隐藏导入 ----
extra_hidden = [
    "soundfile._soundfile",
    "torchaudio.backend",
    "torchaudio.backends",
    "demucs.pretrained",
    "demucs.apply",
    "demucs.separate",
    "demucs.audio",
    "faster_whisper",
    "ctranslate2",
    "idna",
    # numpy 编译子模块（numpy 1.x / 2.x 兼容）
    "numpy.core.multiarray",
    "numpy._core.multiarray",
    "numpy.core._multiarray_umath",
    "numpy._core._multiarray_umath",
]
hiddenimports += extra_hidden

a = Analysis(
    [os.path.join(PROJECT_DIR, "main.py")],
    pathex=[PROJECT_DIR],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # 排除 tkinter，明确不使用
        "tkinter",
        # 排除测试相关
        "pytest", "tests",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="LyricWhisper",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    # upx 压缩 torch/dll 易导致加载问题，默认关闭
    console=False,  # GUI 程序，不显示控制台；调试时可改 True
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # icon=os.path.join(PROJECT_DIR, "icon.ico"),  # 取消注释以使用图标
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="LyricWhisper",
)
