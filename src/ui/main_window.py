"""LyricWhisper 主窗口。

UI 分层：本模块仅负责界面与用户交互；
分离/转写/歌词生成在 src.core，后台线程在 src.utils.workers。
"""
import os
import shutil

from PySide6.QtCore import Qt, QThread, Signal as _Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QComboBox, QCheckBox, QProgressBar, QFileDialog, QGroupBox,
    QMessageBox, QStatusBar, QApplication, QStyle, QProgressDialog,
)

from ..utils.config import get_config
from ..core.gpu_detector import detect_gpu, GpuStatus
from ..core.lyric_writer import LyricLine, to_standard_lrc, to_plain_text, write_text
from ..core.transcriber import detect_available_whisper_models
from ..utils.workers import PipelineWorker
from ..utils.ffmpeg import get_ffmpeg_info
from .constants import LANGUAGES, WHISPER_MODELS, get_qss, get_theme_colors
from .preview_dialog import PreviewDialog
from .download_dialog import DownloadModelDialog


class FileDropLineEdit(QLineEdit):
    """支持拖放文件的单行输入框。"""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setReadOnly(True)
        self.setPlaceholderText("拖入音频/视频文件，或点击右侧「选择文件」")

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        urls = event.mimeData().urls()
        if not urls:
            return
        path = urls[0].toLocalFile()
        if path:
            self.setText(path)
            win = self.window()
            if isinstance(win, MainWindow):
                win.on_file_changed(path)


class _DemucsDownloadWorker(QThread):
    """后台下载 Demucs 模型，所有 UI 更新通过信号回主线程。

    使用独立 httpx.Client（非共享 get_session()），取消时直接 close()
    可立即中断阻塞中的网络读取。
    """
    progress = _Signal(int)       # 百分比
    finished = _Signal()
    error = _Signal(str)

    def __init__(self, url: str, dest: str, parent=None):
        super().__init__(parent)
        self._url = url
        self._dest = dest
        self._cancelled = False
        self._client = None

    def cancel(self):
        """设置取消标志并立即关闭 client，中断阻塞中的网络读取。"""
        self._cancelled = True
        client = self._client
        if client is not None:
            try:
                client.close()
            except Exception:
                pass

    def run(self):
        import httpx
        from ..utils.hf_session import make_hf_client
        tmp = self._dest + ".tmp"
        incomplete = False
        try:
            self._client = make_hf_client()
            stream_timeout = httpx.Timeout(connect=10, read=10, write=10, pool=10)
            with self._client.stream("GET", self._url, timeout=stream_timeout, follow_redirects=True) as resp:
                resp.raise_for_status()
                total = int(resp.headers.get("Content-Length", 0) or 0)
                downloaded = 0
                chunk_size = 1024 * 256
                with open(tmp, "wb") as f:
                    for chunk in resp.iter_bytes(chunk_size):
                        if self._cancelled:
                            incomplete = True
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total > 0:
                            self.progress.emit(int(downloaded / total * 100))
        except Exception:
            self._safe_remove(tmp)
            if self._cancelled:
                return  # 静默退出，UI 由 canceled 信号处理
            self.error.emit("Demucs 模型下载失败，请检查网络连接")
            return
        finally:
            if self._client is not None:
                try:
                    self._client.close()
                except Exception:
                    pass
                self._client = None

        # 仅在下载未完成（被取消中断）时删除临时文件
        if incomplete:
            self._safe_remove(tmp)
            return
        try:
            os.replace(tmp, self._dest)
            self.finished.emit()
        except Exception as e:
            self._safe_remove(tmp)
            self.error.emit(str(e))

    @staticmethod
    def _safe_remove(path):
        try:
            if os.path.isfile(path):
                os.remove(path)
        except Exception:
            pass


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("LyricWhisper · 歌词自动生成")
        self.resize(980, 760)
        self.setMinimumSize(860, 640)

        self.config = get_config()
        self._gpu_status: GpuStatus = None  # type: ignore
        self._gpu_available = False
        self._use_gpu = False
        self._models_available = False
        self._worker: PipelineWorker | None = None
        self._demucs_worker = None
        self._lines: list[LyricLine] = []
        self._source_name = ""

        self._build_ui()
        # 固定使用浅色主题
        app = QApplication.instance()
        if app is not None:
            app.setStyleSheet(get_qss("light"))
        self._detect_gpu_env()
        self._refresh_gpu_ui(initial=True)
        self._detect_ffmpeg()
        self._populate_model_combo()
        self._refresh_demucs_status()

    # ---------------- UI 构建 ----------------

    def _build_ui(self) -> None:
        central = QWidget()
        central.setObjectName("central")
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        # 标题
        header_row = QHBoxLayout()
        title_col = QVBoxLayout()
        title_col.setSpacing(2)
        title = QLabel("LyricWhisper")
        title.setObjectName("title")
        title_col.addWidget(title)
        sub = QLabel("输入歌曲音频/视频，自动分离人声并生成带时间轴的歌词文件")
        sub.setObjectName("hint")
        title_col.addWidget(sub)
        header_row.addLayout(title_col, 1)
        root.addLayout(header_row)

        # 文件选择
        file_box = QGroupBox("1 · 音视频文件")
        fl = QHBoxLayout(file_box)
        self.file_edit = FileDropLineEdit()
        fl.addWidget(self.file_edit, 1)
        btn_choose = QPushButton("选择文件…")
        btn_choose.setObjectName("secondary")
        btn_choose.clicked.connect(self.on_choose_file)
        fl.addWidget(btn_choose)
        root.addWidget(file_box)

        # 参数设置（两列）
        param_box = QGroupBox("2 · 转写参数")
        pl = QHBoxLayout(param_box)
        pl.setSpacing(12)

        # 语言
        col1 = QVBoxLayout()
        col1.addWidget(QLabel("转写语言"))
        self.lang_combo = QComboBox()
        for name, code in LANGUAGES:
            self.lang_combo.addItem(name, code)
        self.lang_combo.currentIndexChanged.connect(self.on_language_changed)
        col1.addWidget(self.lang_combo)
        pl.addLayout(col1)

        # 人声分离模型 + Whisper 模型
        col2 = QVBoxLayout()
        demucs_row = QHBoxLayout()
        demucs_row.setSpacing(4)
        demucs_row.addWidget(QLabel("人声分离模型 (Demucs)："))
        self._demucs_status_label = QLabel("")
        self._demucs_status_label.setObjectName("hint")
        demucs_row.addWidget(self._demucs_status_label, 1)
        self.btn_download_demucs = QPushButton("下载模型")
        self.btn_download_demucs.setObjectName("secondary")
        self.btn_download_demucs.setToolTip("从 Meta 官方 CDN 下载 Demucs 人声分离模型（约 80 MB）")
        self.btn_download_demucs.clicked.connect(self._download_demucs_model)
        demucs_row.addWidget(self.btn_download_demucs)
        col2.addLayout(demucs_row)

        col2.addWidget(QLabel("Whisper 模型"))
        model_row = QHBoxLayout()
        model_row.setSpacing(4)
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(160)
        self.model_combo.currentIndexChanged.connect(self.on_model_changed)
        model_row.addWidget(self.model_combo, 1)
        self.btn_model_dropdown = QPushButton("▾")
        self.btn_model_dropdown.setFixedWidth(24)
        self.btn_model_dropdown.setToolTip("展开模型列表")
        self.btn_model_dropdown.clicked.connect(lambda: self.model_combo.showPopup())
        model_row.addWidget(self.btn_model_dropdown)
        self.btn_refresh_models = QPushButton()
        icon = self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload)
        self.btn_refresh_models.setIcon(icon)
        self.btn_refresh_models.setObjectName("secondary")
        self.btn_refresh_models.setToolTip("重新检测本地 Whisper 模型")
        self.btn_refresh_models.setFixedWidth(32)
        self.btn_refresh_models.clicked.connect(self._populate_model_combo)
        model_row.addWidget(self.btn_refresh_models)
        self.btn_download_model = QPushButton("下载模型…")
        self.btn_download_model.setObjectName("secondary")
        self.btn_download_model.setToolTip("从 HuggingFace Hub 下载 Whisper 模型")
        self.btn_download_model.clicked.connect(self._open_download_dialog)
        model_row.addWidget(self.btn_download_model)
        col2.addLayout(model_row)
        pl.addLayout(col2)

        pl.addStretch(1)
        root.addWidget(param_box)

        # GPU 加速
        gpu_box = QGroupBox("3 · GPU 加速 (NVIDIA)")
        gvl = QVBoxLayout(gpu_box)
        gvl.setSpacing(6)
        # 第一行：开关 + 状态
        gr1 = QHBoxLayout()
        self.gpu_toggle = QCheckBox("使用显卡加速")
        self.gpu_toggle.toggled.connect(self.on_gpu_toggled)
        gr1.addWidget(self.gpu_toggle)
        self.gpu_status_label = QLabel("")
        self.gpu_status_label.setObjectName("gpuStatus")
        gr1.addWidget(self.gpu_status_label, 1)
        self.gpu_env_label = QLabel("")
        self.gpu_env_label.setObjectName("hint")
        self.gpu_env_label.setWordWrap(True)
        gr1.addWidget(self.gpu_env_label, 2)
        gvl.addLayout(gr1)

        root.addWidget(gpu_box)

        # 处理 + 进度
        proc_box = QGroupBox("4 · 处理")
        pl2 = QVBoxLayout(proc_box)
        btn_row = QHBoxLayout()
        self.btn_start = QPushButton("▶ 开始处理")
        self.btn_start.clicked.connect(self.start_processing)
        btn_row.addWidget(self.btn_start)
        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.setObjectName("danger")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self.cancel_processing)
        btn_row.addWidget(self.btn_cancel)
        btn_row.addStretch(1)
        pl2.addLayout(btn_row)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        pl2.addWidget(self.progress)
        self.status_label = QLabel("就绪")
        self.status_label.setObjectName("hint")
        pl2.addWidget(self.status_label)
        root.addWidget(proc_box)

        # 歌词预览入口（点击打开独立窗口，避免主界面拥挤）
        prev_box = QGroupBox("5 · 歌词预览与时间轴微调")
        pb = QHBoxLayout(prev_box)
        self.prev_hint = QLabel("点击右侧按钮打开预览窗口，查看歌词、微调时间轴并导出（需先处理歌曲生成歌词）")
        self.prev_hint.setObjectName("hint")
        self.prev_hint.setWordWrap(True)
        pb.addWidget(self.prev_hint, 1)
        self.btn_open_preview = QPushButton("打开预览窗口…")
        self.btn_open_preview.clicked.connect(self.open_preview)
        pb.addWidget(self.btn_open_preview)
        root.addWidget(prev_box)

        # 导出（处理完成后启用）
        exp_box = QGroupBox("6 · 导出")
        el = QHBoxLayout(exp_box)
        self.btn_lrc = QPushButton("导出 LRC (逐行)")
        self.btn_lrc.setEnabled(False)
        self.btn_lrc.clicked.connect(lambda: self.export("lrc"))
        el.addWidget(self.btn_lrc)
        self.btn_txt = QPushButton("导出纯文本")
        self.btn_txt.setEnabled(False)
        self.btn_txt.clicked.connect(lambda: self.export("txt"))
        el.addWidget(self.btn_txt)
        el.addStretch(1)
        root.addWidget(exp_box)

        # 状态栏
        sb = QStatusBar()
        self.setStatusBar(sb)
        self.statusBar().showMessage("CPU 模式")

    # ---------------- Whisper 模型检测 ----------------

    def _open_download_dialog(self) -> None:
        """打开 Whisper 模型下载对话框。"""
        dlg = DownloadModelDialog(self.config.whisper_model_dir, parent=self)
        dlg.models_updated.connect(self._populate_model_combo)
        dlg.exec()
        # 对话框关闭后再刷新一次（兜底）
        self._populate_model_combo()

    def _populate_model_combo(self) -> None:
        """根据本地检测到的模型填充下拉框，未检测到时给出提示。"""
        self.model_combo.blockSignals(True)
        self.model_combo.clear()

        available = detect_available_whisper_models(self.config.whisper_model_dir)
        self._models_available = bool(available)

        if not available:
            self.model_combo.addItem("未检测到模型，请放入 models/whisper/", None)
            self.model_combo.setEnabled(False)
            self.btn_start.setEnabled(False)
            self.model_combo.blockSignals(False)
            self.statusBar().showMessage(
                "未检测到 Whisper 模型：请将模型放入 models/whisper/<模型名>/ 目录", 8000
            )
            return

        # 显示名 -> 代码 的映射
        name_map = dict(WHISPER_MODELS)
        for code in available:
            display = name_map.get(code, code)
            self.model_combo.addItem(display, code)

        # 优先选上次保存的选择
        saved_model = self.config.get("whisper_model", None)
        selected_idx = -1
        if saved_model:
            for i in range(self.model_combo.count()):
                if self.model_combo.itemData(i) == saved_model:
                    selected_idx = i
                    break
        if selected_idx < 0:
            # 默认选 medium（如果可用），否则选最后一个（最准）
            for i in range(self.model_combo.count()):
                if self.model_combo.itemData(i) == "medium":
                    selected_idx = i
                    break
            if selected_idx < 0:
                selected_idx = self.model_combo.count() - 1
        self.model_combo.setCurrentIndex(selected_idx)

        self.model_combo.setEnabled(True)
        self.model_combo.blockSignals(False)

        # 启用开始按钮（若当前未在运行中）
        if self._worker is None or not self._worker.isRunning():
            self.btn_start.setEnabled(True)

        self.statusBar().showMessage(
            f"已检测到 {len(available)} 个 Whisper 模型：{', '.join(available)}", 5000
        )
        # 触发一次保存当前选择
        self.on_model_changed(self.model_combo.currentIndex())

    # ---------------- GPU 检测与开关 ----------------

    def _detect_gpu_env(self) -> None:
        try:
            self._gpu_status = detect_gpu()
        except Exception as e:
            self._gpu_status = GpuStatus(
                available=False, has_nvidia=False, cuda_libs_ok=False,
                gpu_name="", driver_version="", cuda_version="",
                missing_dlls=[], reason=f"GPU 检测出错：{e}",
            )
        self._gpu_available = self._gpu_status.available

    # ---------------- ffmpeg 检测 ----------------

    def _detect_ffmpeg(self) -> None:
        """启动时检测 ffmpeg，仅在状态栏提示。"""
        path, source = get_ffmpeg_info()
        if source == "local":
            self.statusBar().showMessage("ffmpeg：程序目录自带", 5000)
        elif source == "system":
            self.statusBar().showMessage(f"ffmpeg：系统环境变量（{path}）", 5000)
        else:
            self.statusBar().showMessage("ffmpeg：未找到，请安装 ffmpeg 或放入程序目录", 8000)

    def _refresh_gpu_ui(self, initial: bool = False) -> None:
        st = self._gpu_status
        self.gpu_env_label.setText(st.reason)
        if self._gpu_available:
            self.gpu_toggle.setEnabled(True)
            if initial:
                # 默认开启
                self.gpu_toggle.blockSignals(True)
                self.gpu_toggle.setChecked(True)
                self.gpu_toggle.blockSignals(False)
                self._use_gpu = True
        else:
            # GPU 环境不可用：开关置灰禁用，强制 CPU
            self.gpu_toggle.setEnabled(False)
            self.gpu_toggle.blockSignals(True)
            self.gpu_toggle.setChecked(False)
            self.gpu_toggle.blockSignals(False)
            self._use_gpu = False
        # 状态指示文字：CPU / GPU 模式
        colors = get_theme_colors("light")
        if self._use_gpu:
            self.gpu_status_label.setText("GPU 模式")
            self.gpu_status_label.setStyleSheet(f"color:{colors['gpu_on']}; font-weight:600;")
        else:
            self.gpu_status_label.setText("CPU 模式")
            self.gpu_status_label.setStyleSheet(f"color:{colors['gpu_off']}; font-weight:600;")
        self.statusBar().showMessage("GPU 模式" if self._use_gpu else "CPU 模式")

    def on_gpu_toggled(self, checked: bool) -> None:
        # 任务进行中禁止切换
        if self._worker is not None and self._worker.isRunning():
            # 还原开关状态
            self.gpu_toggle.blockSignals(True)
            self.gpu_toggle.setChecked(self._use_gpu)
            self.gpu_toggle.blockSignals(False)
            QMessageBox.warning(
                self, "无法切换",
                "当前有任务正在处理，请等待完成后再切换。",
            )
            return
        self._use_gpu = bool(checked) and self._gpu_available
        self._refresh_gpu_ui()
        # 两个模型 device 保持一致：统一由 _use_gpu 决定，在 worker 中传入
        msg = "已切换为 GPU 加速（Demucs + Whisper 均使用 CUDA）" if self._use_gpu \
            else "已切换为 CPU 模式（Demucs + Whisper 均使用 CPU）"
        self.status_label.setText(msg)

    # ---------------- Demucs 人声分离模型 ---------------

    def _demucs_checkpoint(self) -> str:
        """返回 demucs 模型检查点文件路径。"""
        torch_home = os.environ.get(
            "TORCH_HOME",
            os.path.join(self.config.models_dir, "hub"),
        )
        return os.path.join(
            torch_home, "hub", "checkpoints", "955717e8-8726e21a.th"
        )

    def _refresh_demucs_status(self) -> None:
        """刷新 demucs 模型状态显示。"""
        ckpt = self._demucs_checkpoint()
        exists = os.path.isfile(ckpt)
        if exists:
            size_mb = os.path.getsize(ckpt) / (1024 * 1024)
            self._demucs_status_label.setText(
                f"✅ 已缓存（{size_mb:.0f} MB）" if size_mb > 50
                else "✅ 已缓存"
            )
            self._demucs_status_label.setStyleSheet("color: #0f9d76;")
            self.btn_download_demucs.setVisible(False)
        else:
            self._demucs_status_label.setText("❌ 未下载（需联网自动下载，约 80 MB）")
            self.btn_download_demucs.setVisible(True)

    def _download_demucs_model(self) -> None:
        """从 Meta CDN 下载 demucs 人声分离模型（线程安全）。"""
        url = (
            "https://dl.fbaipublicfiles.com/demucs/"
            "hybrid_transformer/955717e8-8726e21a.th"
        )
        ckpt = self._demucs_checkpoint()
        target_dir = os.path.dirname(ckpt)
        os.makedirs(target_dir, exist_ok=True)

        progress = QProgressDialog(
            "正在下载 Demucs 模型…\n请耐心等待，文件约 80 MB", "取消",
            0, 100, self,
        )
        progress.setWindowTitle("下载人声分离模型")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        worker = _DemucsDownloadWorker(url, ckpt, parent=self)
        self._demucs_worker = worker  # 防止被 GC

        def _on_progress(pct):
            if not progress.wasCanceled():
                progress.setValue(pct)

        def _on_finished():
            progress.close()
            self._refresh_demucs_status()
            QMessageBox.information(self, "下载完成", "Demucs 人声分离模型下载完毕。")

        def _on_error(msg):
            progress.close()
            QMessageBox.critical(
                self, "下载失败",
                f"Demucs 模型下载失败：{msg}\n\n"
                f"可尝试手动下载：\n{url}\n"
                f"放入目录：{target_dir}",
            )

        def _on_cancel():
            worker.cancel()

        worker.progress.connect(_on_progress)
        worker.finished.connect(_on_finished)
        worker.error.connect(_on_error)
        progress.canceled.connect(_on_cancel)
        worker.start()

    # ---------------- 文件 ----------------

    def on_choose_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择音视频文件", "",
            "音视频文件 (*.mp3 *.wav *.flac *.aac *.m4a *.ogg *.wma *.mp4 *.mkv *.mov *.avi *.webm);;所有文件 (*.*)",
        )
        if path:
            self.file_edit.setText(path)
            self.on_file_changed(path)

    def on_file_changed(self, path: str) -> None:
        self._source_name = os.path.splitext(os.path.basename(path))[0]
        self.status_label.setText(f"已选择：{os.path.basename(path)}")

    # ---------------- 参数变更 ----------------

    def on_language_changed(self, idx: int) -> None:
        code = self.lang_combo.itemData(idx)
        # 手动指定语言时跳过自动检测逻辑：在 transcribe 中 language 传具体代码
        # （None 才触发自动检测），此处仅记录，无需额外动作
        if code is None:
            self.statusBar().showMessage("语言：自动检测")
        else:
            self.statusBar().showMessage(f"语言：手动指定（{code}）→ 跳过自动检测")

    def on_model_changed(self, idx: int) -> None:
        size = self.model_combo.itemData(idx)
        if size is None:
            return  # 占位项，不保存
        self.config.set("whisper_model", size)

    # ---------------- 处理流水线 ----------------

    def start_processing(self) -> None:
        path = self.file_edit.text().strip()
        if not path or not os.path.isfile(path):
            QMessageBox.warning(self, "未选择文件", "请先选择或拖入音视频文件。")
            return
        if self._worker is not None and self._worker.isRunning():
            QMessageBox.information(self, "处理中", "已有任务正在处理。")
            return

        lang = self.lang_combo.currentData()
        model_size = self.model_combo.currentData()
        if not model_size:
            QMessageBox.warning(
                self, "无可用模型",
                "未检测到 Whisper 模型。\n请将 faster-whisper 模型（CTranslate2 格式）"
                "放入程序目录下的 models/whisper/<模型名>/ 文件夹，\n"
                "然后点击模型旁的「⟳」按钮重新检测。"
            )
            return
        # 开关与检测联动：仅在可用时允许 GPU
        use_gpu = self._use_gpu and self._gpu_available

        self._set_running(True)
        self.progress.setValue(0)
        self.status_label.setText("启动任务…")

        self._worker = PipelineWorker(path, lang, model_size, use_gpu, parent=self)
        self._worker.progress.connect(self.on_progress)
        self._worker.finished_ok.connect(self.on_finished)
        self._worker.failed.connect(self.on_failed)
        self._worker.cancelled.connect(self.on_cancelled)
        self._worker.start()

    def on_progress(self, ratio: float, text: str) -> None:
        self.progress.setValue(int(ratio * 100))
        self.status_label.setText(text)

    def on_finished(self, result) -> None:
        self._lines = result.lines
        self._source_name = os.path.splitext(os.path.basename(result.source_path))[0]
        lang_info = result.language or "?"
        detect_info = "自动检测" if result.detected else "手动指定"
        self.status_label.setText(
            f"完成 · {len(self._lines)} 行 · 语言={lang_info}（{detect_info}）"
        )
        self.statusBar().showMessage(f"完成 · {len(self._lines)} 行歌词，可打开预览窗口微调")
        for b in (self.btn_lrc, self.btn_txt):
            b.setEnabled(True)  # 启用导出
        self._set_running(False)

    def on_failed(self, msg: str) -> None:
        self._set_running(False)
        self.status_label.setText(f"失败：{msg}")
        QMessageBox.critical(self, "处理失败", msg)

    def on_cancelled(self) -> None:
        self._set_running(False)
        self.status_label.setText("已取消")
        self.statusBar().showMessage("任务已取消")

    def cancel_processing(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
            self.status_label.setText("正在取消…")

    def _set_running(self, running: bool) -> None:
        self.btn_start.setEnabled(not running and self._models_available)
        self.btn_cancel.setEnabled(running)
        self.lang_combo.setEnabled(not running)
        self.model_combo.setEnabled(not running and self._models_available)
        self.btn_refresh_models.setEnabled(not running)
        self.btn_download_model.setEnabled(not running)
        self.file_edit.setEnabled(not running)
        # 任务进行中 GPU 开关也禁用（双重保险）
        self.gpu_toggle.setEnabled(not running and self._gpu_available)

    # ---------------- 预览窗口 ----------------

    def open_preview(self) -> None:
        """打开独立的歌词预览与时间轴微调窗口。"""
        dlg = PreviewDialog(self._lines, self._source_name, parent=self)
        dlg.show()

    # ---------------- 导出 ----------------

    def export(self, fmt: str) -> None:
        if not self._lines:
            QMessageBox.information(self, "提示", "暂无歌词，请先处理。")
            return
        default_name = (self._source_name or "lyrics") + f".{fmt}"
        default_path = os.path.join(self.config.output_dir, default_name)
        ext_filter = {
            "lrc": "LRC 歌词 (*.lrc)",
            "txt": "纯文本 (*.txt)",
        }[fmt]
        path, _ = QFileDialog.getSaveFileName(self, "导出", default_path, ext_filter)
        if not path:
            return
        header = {"ar": "LyricWhisper", "re": "LyricWhisper"}
        try:
            if fmt == "lrc":
                text = to_standard_lrc(self._lines, header)
            else:
                text = to_plain_text(self._lines)
            write_text(path, text)
        except Exception as e:
            QMessageBox.critical(self, "导出失败", str(e))
            return
        self.statusBar().showMessage(f"已导出：{path}")

    # ---------------- 关闭 ----------------

    def closeEvent(self, event) -> None:
        if self._worker is not None and self._worker.isRunning():
            reply = QMessageBox.question(
                self, "退出", "有任务正在处理，确定退出并取消任务吗？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self._worker.cancel()
            self._worker.wait(3000)
        # 保存模型选择
        self.on_model_changed(self.model_combo.currentIndex())
        # 清理 temp/ + HF 缓存
        self._cleanup_temp()
        self._cleanup_hf_cache()
        event.accept()

    def _cleanup_temp(self) -> None:
        """关闭时清理 temp/ 目录下的临时文件（分离中间产物等）。"""
        import shutil
        temp_dir = self.config.temp_dir
        if not os.path.isdir(temp_dir):
            return
        try:
            for entry in os.listdir(temp_dir):
                full = os.path.join(temp_dir, entry)
                try:
                    if os.path.isfile(full) or os.path.islink(full):
                        os.remove(full)
                    elif os.path.isdir(full):
                        shutil.rmtree(full)
                except Exception:
                    pass
        except Exception:
            pass

    def _cleanup_hf_cache(self) -> None:
        """关闭时清理 HuggingFace Hub 临时缓存目录。"""
        import shutil
        cache_dir = os.environ.get("HF_HOME") or os.environ.get("HF_HUB_CACHE")
        if not cache_dir or not os.path.isdir(cache_dir):
            return
        try:
            for entry in os.listdir(cache_dir):
                full = os.path.join(cache_dir, entry)
                try:
                    if os.path.isfile(full) or os.path.islink(full):
                        os.remove(full)
                    elif os.path.isdir(full):
                        shutil.rmtree(full)
                except Exception:
                    pass
        except Exception:
            pass
