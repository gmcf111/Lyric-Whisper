# LyricWhisper

基于 Whisper + Demucs 的桌面端智能歌词转录工具。

## 功能

- **语音转写**：使用 faster-whisper 将音频文件转录为带时间戳的歌词（SRT/LRC 格式）
- **人声分离**：使用 Demucs (htdemucs) 分离人声和伴奏，提升嘈杂音频的转录质量
- **GPU 加速**：支持 NVIDIA GPU（CUDA 12.6 + cuDNN 9），无 GPU 自动降级 CPU
- **多模型选择**：支持 tiny / base / small / medium / large-v3 等多种 Whisper 模型

## 快速开始

### 安装

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126
```

### 运行

```bash
python main.py
```

## 技术栈

- **GUI**: PySide6 (Qt 6)
- **语音转写**: faster-whisper / ctranslate2
- **人声分离**: Demucs (htdemucs) + PyTorch
- **GPU**: CUDA 12.6 + cuDNN 9
- **打包**: PyInstaller (onedir)

## 许可证

GNU General Public License v3.0
