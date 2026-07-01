"""Whisper 模型下载对话框。

从 HuggingFace Hub 下载 Systran 预转换的 CTranslate2 格式模型，
仅下载必要文件（跳过 .gitattributes、README.md 等非模型文件）。
支持镜像站切换，下载完成后自动通知主界面刷新模型列表。
"""
import os

import httpx
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QCheckBox, QPushButton,
    QProgressBar, QLabel, QGroupBox, QComboBox, QMessageBox,
)

from ..utils.hf_session import make_hf_client


# 已知模型及 HuggingFace 仓库
MODEL_REPOS: dict[str, str] = {
    "tiny":    "Systran/faster-whisper-tiny",
    "base":    "Systran/faster-whisper-base",
    "small":   "Systran/faster-whisper-small",
    "medium":  "Systran/faster-whisper-medium",
    "large-v3":"Systran/faster-whisper-large-v3",
}

MODEL_SIZES: dict[str, str] = {
    "tiny":    "~150 MB",
    "base":    "~300 MB",
    "small":   "~1 GB",
    "medium":  "~1.5 GB",
    "large-v3":"~3.1 GB",
}

# 镜像站列表（必须是合法的 HF endpoint：纯 scheme+域名，无尾斜杠、无路径）
MIRRORS: list[tuple[str, str]] = [
    ("rimuru 代理（推荐）", "https://hf.rimuru.work"),
    ("hf-mirror（国内镜像站）", "https://hf-mirror.com"),
    ("HuggingFace（国外）", "https://huggingface.co"),
]

# 下载时跳过的无用文件
_SKIP_FILES = {".gitattributes", "README.md"}


class _DownloadWorker(QThread):
    """后台下载线程，使用独立 httpx.Client 流式下载，支持即时取消与真实进度。

    关键设计：
    - 每个 worker 拥有独立的 httpx.Client（不共享 get_session()），
      这样取消时可以直接 close()，立即中断阻塞中的网络读取。
    - 文件列表通过直接 HTTP API 获取（GET /api/models/{repo_id}），
      同样使用此 client，确保列表阶段也可取消。
    """

    progress = Signal(int, str)   # (百分比, 状态文字)
    file_done = Signal(str, str)  # (模型名, 文件名)
    succeeded = Signal()          # 仅在全部下载成功时发出（不与 QThread.finished 冲突）
    error = Signal(str)
    cancelled = Signal()

    def __init__(
        self,
        selected: list[tuple[str, str]],  # [(model_name, repo_id), ...]
        model_dir: str,
        mirror_endpoint: str = "",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._selected = selected
        self._model_dir = model_dir
        self._mirror_endpoint = mirror_endpoint or "https://huggingface.co"
        self._cancelled = False
        self._client: httpx.Client | None = None
        self._file_sizes: dict[tuple[str, str], int] = {}

    def cancel(self) -> None:
        """设置取消标志并立即关闭 client，中断任何阻塞中的网络读取。"""
        self._cancelled = True
        client = self._client
        if client is not None:
            try:
                client.close()
            except Exception:
                pass

    def run(self) -> None:
        try:
            self._client = make_hf_client()
            self._run_impl()
        except Exception as e:
            # 捕获所有未被内部方法处理的异常（如 os.makedirs 失败），
            # 确保 UI 不会卡在 "忙碌" 状态
            if not self._cancelled:
                self.error.emit(f"下载过程中发生错误：{e}")
            else:
                self.cancelled.emit()
        finally:
            if self._client is not None:
                try:
                    self._client.close()
                except Exception:
                    pass
                self._client = None

    def _run_impl(self) -> None:
        """实际下载逻辑，由 run() 调用。"""
        # 1) 获取每个模型的文件列表 + 真实大小
        all_tasks: list[tuple[str, str, str]] = []
        for name, repo_id in self._selected:
            if self._cancelled:
                self.cancelled.emit()
                return
            info = self._fetch_repo_info_with_retry(repo_id, name)
            if info is None:
                return  # 错误或取消信号已发出
            for sibling in info.get("siblings", []):
                fname = sibling.get("rfilename", "")
                if not fname or fname in _SKIP_FILES:
                    continue
                size = sibling.get("size", 0)
                if size and size > 0:
                    self._file_sizes[(repo_id, fname)] = size
                all_tasks.append((name, repo_id, fname))

        if not all_tasks:
            self.error.emit("没有需要下载的文件")
            return

        # 2) 逐个下载
        total_files = len(all_tasks)
        for i, (name, repo_id, filename) in enumerate(all_tasks):
            if self._cancelled:
                self.cancelled.emit()
                return

            target_dir = os.path.join(self._model_dir, name)
            os.makedirs(target_dir, exist_ok=True)
            dest = os.path.join(target_dir, filename)

            ok = self._download_file(
                repo_id, filename, dest,
                file_index=i, total_files=total_files, model_name=name,
            )
            if not ok:
                return
            self.file_done.emit(name, filename)

        self.progress.emit(100, "下载完成")
        self.succeeded.emit()

    # ---- 文件列表获取 ----

    def _fetch_repo_info_with_retry(self, repo_id: str, name: str):
        """通过直接 HTTP API 获取仓库文件列表和大小，最多重试 3 次。

        返回 dict（含 siblings）或 None（错误/取消时已发信号）。
        """
        api_url = f"{self._mirror_endpoint}/api/models/{repo_id}"
        last_err = None
        for attempt in range(3):
            if self._cancelled:
                self.cancelled.emit()
                return None
            try:
                self.progress.emit(
                    0, f"正在获取 {name} 文件列表…（第 {attempt + 1} 次尝试）"
                )
                resp = self._client.get(api_url)
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                last_err = e
                if self._cancelled:
                    self.cancelled.emit()
                    return None
                if attempt < 2:
                    if self._interruptible_sleep(2):
                        self.cancelled.emit()
                        return None
        self.error.emit(f"获取 {name} 文件列表失败（已重试 3 次）：{last_err}")
        return None

    def _interruptible_sleep(self, seconds: float) -> bool:
        """可被取消中断的休眠。返回 True 表示被中断。"""
        elapsed = 0.0
        while elapsed < seconds:
            if self._cancelled:
                return True
            self.msleep(200)
            elapsed += 0.2
        return False

    # ---- 文件下载 ----

    def _download_file(
        self,
        repo_id: str,
        filename: str,
        dest: str,
        file_index: int,
        total_files: int,
        model_name: str,
    ) -> bool:
        """流式下载单个文件（可即时中断、真实进度）。"""
        url = f"{self._mirror_endpoint}/{repo_id}/resolve/main/{filename}"
        known_size = self._file_sizes.get((repo_id, filename), 0)
        tmp = dest + ".part"
        # 流式下载用较短 read 超时，配合 client.close() 实现即时取消
        stream_timeout = httpx.Timeout(connect=10, read=10, write=10, pool=10)

        try:
            with self._client.stream("GET", url, timeout=stream_timeout, follow_redirects=True) as resp:
                resp.raise_for_status()
                total = int(resp.headers.get("Content-Length", 0) or 0)
                if total <= 0:
                    total = known_size  # 用预取大小兜底
                downloaded = 0
                chunk_size = 1024 * 256
                incomplete = False
                with open(tmp, "wb") as f:
                    for chunk in resp.iter_bytes(chunk_size):
                        if self._cancelled:
                            incomplete = True
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        self._emit_progress(
                            file_index, total_files, model_name, filename,
                            downloaded, total,
                        )
        except Exception:
            # client 被 close()（取消）或网络异常都会走到这里
            self._safe_remove(tmp)
            if self._cancelled:
                self.cancelled.emit()
            else:
                self.error.emit(f"下载 {model_name}/{filename} 失败，请检查网络或更换镜像源")
            return False

        # 仅在下载未完成（被取消中断）时删除临时文件；
        # 如果所有 chunk 已收完，即使 _cancelled 在循环结束后才被置位，
        # 也应保留已下完的文件。
        if incomplete:
            self._safe_remove(tmp)
            self.cancelled.emit()
            return False

        try:
            os.replace(tmp, dest)
        except Exception as e:
            self._safe_remove(tmp)
            self.error.emit(f"保存 {model_name}/{filename} 失败：{e}")
            return False
        return True

    def _emit_progress(
        self, file_index, total_files, model_name, filename, downloaded, total,
    ) -> None:
        """计算跨文件的整体真实进度。支持 total=0 时用等分文件权重。"""
        if total > 0:
            file_ratio = min(downloaded / total, 1.0)
            mb = downloaded / (1024 * 1024)
            total_mb = total / (1024 * 1024)
            detail = f"{mb:.1f}/{total_mb:.1f} MB"
        else:
            file_ratio = 0.0
            mb = downloaded / (1024 * 1024)
            detail = f"{mb:.1f} MB（未知总大小）"
        overall = (file_index + file_ratio) / total_files
        pct = int(overall * 100)
        self.progress.emit(pct, f"下载 {model_name}/{filename}（{detail}）")

    @staticmethod
    def _safe_remove(path: str) -> None:
        try:
            if os.path.isfile(path):
                os.remove(path)
        except Exception:
            pass


class DownloadModelDialog(QDialog):
    """Whisper 模型下载对话框。"""

    # 下载完成时发出此信号，通知主界面刷新模型列表
    models_updated = Signal()

    def __init__(self, model_dir: str, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("下载 Whisper 模型")
        self.resize(500, 420)
        self.setMinimumSize(440, 380)

        self._model_dir = model_dir
        self._worker: _DownloadWorker | None = None
        self._checkboxes: dict[str, QCheckBox] = {}

        self._build_ui()
        self._refresh_status()

    # ---------------- UI 构建 ----------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(10)

        # 说明文字
        info = QLabel(
            "选择需要下载的 Whisper 模型，程序自动从 HuggingFace Hub 下载。\n"
            "仅下载必要的模型文件（model.bin、config.json 等），跳过无关文件。"
        )
        info.setWordWrap(True)
        info.setObjectName("hint")
        root.addWidget(info)

        # 镜像源选择
        mirror_row = QHBoxLayout()
        mirror_row.addWidget(QLabel("镜像源："))
        self.mirror_combo = QComboBox()
        for label, _ in MIRRORS:
            self.mirror_combo.addItem(label)
        self.mirror_combo.setToolTip("选择下载镜像源，rimuru 代理为第三方公益服务")
        self.mirror_combo.currentIndexChanged.connect(self._update_mirror_credit)
        mirror_row.addWidget(self.mirror_combo, 1)
        root.addLayout(mirror_row)

        # 镜像源致谢标注（根据所选镜像源动态显示）
        self.mirror_credit = QLabel("")
        self.mirror_credit.setObjectName("hint")
        self.mirror_credit.setOpenExternalLinks(True)
        self.mirror_credit.setWordWrap(True)
        root.addWidget(self.mirror_credit)
        self._update_mirror_credit(0)

        # 模型列表
        group = QGroupBox("可选模型")
        gl = QVBoxLayout(group)
        gl.setSpacing(6)

        for name in MODEL_REPOS:
            cb = QCheckBox(f"{name}（{MODEL_SIZES.get(name, '?')}）")
            self._checkboxes[name] = cb
            gl.addWidget(cb)

        root.addWidget(group, 1)

        # 状态与进度
        self._status_label = QLabel("就绪")
        self._status_label.setObjectName("hint")
        root.addWidget(self._status_label)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        root.addWidget(self._progress)

        # 按钮行
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        self.btn_download = QPushButton("开始下载")
        self.btn_download.clicked.connect(self._start_download)
        btn_row.addWidget(self.btn_download)

        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.setObjectName("danger")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._cancel_download)
        btn_row.addWidget(self.btn_cancel)

        self.btn_refresh = QPushButton("刷新列表")
        self.btn_refresh.setObjectName("secondary")
        self.btn_refresh.setToolTip("重新检测本地已下载的模型")
        self.btn_refresh.clicked.connect(self._refresh_status)
        btn_row.addWidget(self.btn_refresh)

        btn_row.addStretch(1)

        self.btn_close = QPushButton("关闭")
        self.btn_close.clicked.connect(self.close)
        btn_row.addWidget(self.btn_close)

        root.addLayout(btn_row)

    def _get_current_mirror(self) -> str:
        idx = self.mirror_combo.currentIndex()
        _, endpoint = MIRRORS[idx]
        return endpoint

    def _update_mirror_credit(self, idx: int) -> None:
        """根据所选镜像源显示对应的致谢/来源标注。"""
        _, endpoint = MIRRORS[idx]
        if endpoint == "https://hf.rimuru.work":
            self.mirror_credit.setText(
                'rimuru 代理由 <a href="https://github.com/AinzRimuru/HuggingfaceProxy">'
                "HuggingfaceProxy</a> 提供（MIT 许可证），感谢作者公益维护"
            )
            self.mirror_credit.setVisible(True)
        elif endpoint == "https://hf-mirror.com":
            self.mirror_credit.setText(
                '由 <a href="https://hf-mirror.com">https://hf-mirror.com</a> 提供镜像'
            )
            self.mirror_credit.setVisible(True)
        else:
            self.mirror_credit.setVisible(False)

    def _refresh_status(self) -> None:
        """刷新每个模型的已下载状态标签。

        使用与 core.transcriber 相同的完整校验（bin + config + vocab），
        避免下载一半的模型被误标为"已下载"。
        """
        from ..core.transcriber import _is_valid_model_dir

        for name, cb in self._checkboxes.items():
            model_dir = os.path.join(self._model_dir, name)
            if _is_valid_model_dir(model_dir):
                txt = cb.text()
                if "（已下载）" not in txt:
                    cb.setText(txt + " （已下载）")
            else:
                txt = cb.text()
                txt = txt.replace(" （已下载）", "")
                cb.setText(txt)

    # ---------------- 下载控制 ----------------

    def _start_download(self) -> None:
        selected: list[tuple[str, str]] = []
        for name, repo_id in MODEL_REPOS.items():
            if self._checkboxes[name].isChecked():
                selected.append((name, repo_id))

        if not selected:
            QMessageBox.information(self, "提示", "请至少选择一个模型。")
            return

        self._set_ui_busy(True)
        self._progress.setValue(0)
        mirror_endpoint = self._get_current_mirror()
        self._status_label.setText("准备中…")

        self._worker = _DownloadWorker(
            selected, self._model_dir, mirror_endpoint=mirror_endpoint, parent=self,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.file_done.connect(self._on_file_done)
        self._worker.succeeded.connect(self._on_succeeded)
        self._worker.error.connect(self._on_error)
        self._worker.cancelled.connect(self._on_cancelled)
        self._worker.start()

    def _cancel_download(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
            self._status_label.setText("正在取消…")
            self.btn_cancel.setEnabled(False)

    def _set_ui_busy(self, busy: bool) -> None:
        self.btn_download.setEnabled(not busy)
        self.btn_cancel.setEnabled(busy)
        self.btn_close.setEnabled(not busy)
        self.btn_refresh.setEnabled(not busy)
        self.mirror_combo.setEnabled(not busy)
        for cb in self._checkboxes.values():
            cb.setEnabled(not busy)

    # ---------------- 信号处理 ----------------

    def _on_progress(self, pct: int, text: str) -> None:
        self._progress.setValue(pct)
        self._status_label.setText(text)

    def _on_file_done(self, model_name: str, filename: str) -> None:
        pass  # 可通过 progress 追踪状态

    def _on_succeeded(self) -> None:
        self._set_ui_busy(False)
        self._progress.setValue(100)
        self._status_label.setText("全部下载完成！")
        self._refresh_status()
        self.models_updated.emit()
        QMessageBox.information(
            self, "下载完成",
            "模型下载完毕，列表已自动刷新。",
        )

    def _on_cancelled(self) -> None:
        self._set_ui_busy(False)
        self._progress.setValue(0)
        self._status_label.setText("已取消下载")
        self._refresh_status()

    def _on_error(self, msg: str) -> None:
        self._set_ui_busy(False)
        self._progress.setValue(0)
        self._status_label.setText("下载失败")
        QMessageBox.critical(self, "下载出错", msg)

    def closeEvent(self, event) -> None:
        if self._worker is not None and self._worker.isRunning():
            reply = QMessageBox.question(
                self, "确认", "正在下载中，确定取消并关闭？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self._worker.cancel()
            self._worker.wait(2000)
        event.accept()
