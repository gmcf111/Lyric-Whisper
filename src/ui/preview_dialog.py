"""歌词预览与时间轴微调独立窗口。

从主界面入口打开，避免主界面拥挤。包含：
- 歌词列表表格（只读，点选一行）
- 时间轴微调面板（提前/延后两行，各5个可编辑档位带单位切换）
- 全局偏移（带单位切换）
- 导出入口（LRC / 纯文本）
"""
import os

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTableWidget,
    QTableWidgetItem, QSpinBox, QComboBox, QHeaderView, QGroupBox,
    QMessageBox, QFileDialog, QAbstractSpinBox, QMenu,
)

from ..core.lyric_writer import LyricLine, to_standard_lrc, to_plain_text, write_text
from ..utils.config import get_config
from .constants import get_theme_colors

# 默认档位预设（均为 10 的倍数，LRC 仅支持百分秒精度）
_DEFAULT_STEPS = [10, 20, 50, 100, 200]
_UNITS = ["ms", "s"]


class PreviewDialog(QDialog):
    """歌词预览与时间轴微调独立窗口。"""

    def __init__(self, lines: list[LyricLine], source_name: str, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("歌词预览与时间轴微调")
        self.resize(960, 680)
        self.setMinimumSize(760, 520)
        self.setModal(False)  # 非模态：可与主窗口并存

        self._lines = lines
        self._source_name = source_name
        self._minus_slots: list[tuple[QSpinBox, QComboBox]] = []  # 提前行 5 档
        self._plus_slots: list[tuple[QSpinBox, QComboBox]] = []   # 延后行 5 档

        self._build_ui()
        self._populate()
        self._apply_theme_color()

    # ---------------- UI 构建 ----------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(10)

        # 顶部标题
        title = QLabel("歌词预览与时间轴微调")
        title.setObjectName("title")
        root.addWidget(title)

        # 歌词表格（只读，单选；LRC 只有开始时间，不显示结束）
        self.table = QTableWidget(0, 3)
        self.table.setObjectName("lyricTable")
        self.table.setHorizontalHeaderLabels(["#", "开始时间", "歌词"])
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.itemSelectionChanged.connect(self._on_row_selection_changed)
        # 右键编辑歌词
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._on_context_menu)
        self.table.itemChanged.connect(self._on_item_changed)
        root.addWidget(self.table, 1)

        # 微调面板
        tune_box = QGroupBox("时间轴微调面板")
        tv = QVBoxLayout(tune_box)
        tv.setSpacing(6)

        self.sel_label = QLabel("未选中歌词")
        self.sel_label.setObjectName("hint")
        self.sel_label.setWordWrap(True)
        tv.addWidget(self.sel_label)

        # 提前行（5 个独立档位）
        minus_row = QHBoxLayout()
        minus_row.setSpacing(4)
        minus_row.addWidget(QLabel("← 提前"))
        for i, default in enumerate(_DEFAULT_STEPS):
            sb, cb = self._make_slot(default)
            self._minus_slots.append((sb, cb))
            minus_row.addWidget(sb)
            minus_row.addWidget(cb)
            btn = QPushButton("提前")
            btn.setObjectName("secondary")
            btn.setFixedWidth(56)
            btn.clicked.connect(lambda _=False, idx=i: self._nudge_selected(-self._slot_ms(self._minus_slots, idx)))
            minus_row.addWidget(btn)
        minus_row.addStretch(1)
        tv.addLayout(minus_row)

        # 延后行（5 个独立档位）
        plus_row = QHBoxLayout()
        plus_row.setSpacing(4)
        plus_row.addWidget(QLabel("延后 →"))
        for i, default in enumerate(_DEFAULT_STEPS):
            sb, cb = self._make_slot(default)
            self._plus_slots.append((sb, cb))
            plus_row.addWidget(sb)
            plus_row.addWidget(cb)
            btn = QPushButton("延后")
            btn.setObjectName("secondary")
            btn.setFixedWidth(56)
            btn.clicked.connect(lambda _=False, idx=i: self._nudge_selected(self._slot_ms(self._plus_slots, idx)))
            plus_row.addWidget(btn)
        plus_row.addStretch(1)
        tv.addLayout(plus_row)

        root.addWidget(tune_box)

        # 全局偏移（带单位切换）
        grow = QHBoxLayout()
        grow.addWidget(QLabel("全局偏移（应用到全部歌词）"))
        self.global_offset = QSpinBox()
        self.global_offset.setRange(-999, 999)   # 最多 3 位，避免超长
        self.global_offset.setValue(0)
        self.global_offset.setFixedWidth(110)
        self.global_offset.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.global_offset.setSuffix("0")        # 末尾固定零（ms 模式）
        grow.addWidget(self.global_offset)
        self.global_unit = QComboBox()
        self.global_unit.addItems(_UNITS)
        self.global_unit.setFixedWidth(56)
        self.global_unit.currentTextChanged.connect(
            lambda text: self.global_offset.setSuffix("0" if text == "ms" else "")
        )
        grow.addWidget(self.global_unit)
        self.btn_apply_global = QPushButton("应用到全部")
        self.btn_apply_global.setObjectName("secondary")
        self.btn_apply_global.clicked.connect(self.apply_global_offset)
        grow.addWidget(self.btn_apply_global)
        grow.addStretch(1)
        root.addLayout(grow)

        # 底部：导出 + 关闭
        bottom = QHBoxLayout()
        bottom.addStretch(1)
        self.btn_lrc = QPushButton("导出 LRC (逐行)")
        self.btn_lrc.clicked.connect(lambda: self.export("lrc"))
        bottom.addWidget(self.btn_lrc)
        self.btn_txt = QPushButton("导出纯文本")
        self.btn_txt.clicked.connect(lambda: self.export("txt"))
        bottom.addWidget(self.btn_txt)
        self.btn_close = QPushButton("关闭")
        self.btn_close.setObjectName("secondary")
        self.btn_close.clicked.connect(self.accept)
        bottom.addWidget(self.btn_close)
        root.addLayout(bottom)

    def _make_slot(self, default_ms: int) -> tuple[QSpinBox, QComboBox]:
        """创建一个档位控件。ms 模式下末尾固定 0，用户输入值 ×10 = 实际毫秒。"""
        sb = QSpinBox()
        sb.setRange(1, 999)              # 用户最多输入 3 位，避免超长数字
        sb.setValue(default_ms // 10)    # 实际值 = value × 10
        sb.setFixedWidth(70)
        sb.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        sb.setSuffix("0")               # 末尾固定零（ms 模式）
        cb = QComboBox()
        cb.addItems(_UNITS)
        cb.setFixedWidth(52)
        cb.currentTextChanged.connect(lambda text, s=sb: self._on_slot_unit_changed(s, text))
        return sb, cb

    @staticmethod
    def _on_slot_unit_changed(sb: QSpinBox, unit: str) -> None:
        """单位切换：ms 时末尾固定 0，s 时无固定零。"""
        if unit == "ms":
            sb.setSuffix("0")
        else:
            sb.setSuffix("")

    def _slot_ms(self, slots: list, index: int) -> int:
        """读取指定档位列表第 index 个的毫秒值。ms 模式下 value×10，s 模式下 value×1000。"""
        sb, cb = slots[index]
        val = sb.value()
        if cb.currentText() == "ms":
            val = val * 10           # 末尾零机制：实际值 = 输入值 × 10
        else:
            val = val * 1000
        # 强制圆整到 10 的倍数（LRC 不支持个位数毫秒）
        val = max(10, int(round(val / 10) * 10))
        return val

    # ---------------- 数据填充 ----------------

    def _populate(self) -> None:
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        for i, ln in enumerate(self._lines):
            self._insert_row(i, ln)
        self.table.blockSignals(False)
        self._update_sel_label()

    def _insert_row(self, row: int, ln: LyricLine) -> None:
        self.table.insertRow(row)
        idx_item = QTableWidgetItem(str(row + 1))
        idx_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        start_item = QTableWidgetItem(self._fmt_lrc(ln.start))
        start_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        text_item = QTableWidgetItem(ln.text)
        for it in (idx_item, start_item, text_item):
            it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.table.setItem(row, 0, idx_item)
        self.table.setItem(row, 1, start_item)
        self.table.setItem(row, 2, text_item)

    @staticmethod
    def _fmt_lrc(seconds: float) -> str:
        if seconds < 0:
            seconds = 0.0
        m = int(seconds // 60)
        s = seconds - m * 60
        return f"{m:02d}:{s:05.2f}"

    def _refresh_row(self, row: int) -> None:
        ln = self._lines[row]
        self.table.item(row, 1).setText(self._fmt_lrc(ln.start))

    # ---------------- 右键编辑歌词 ----------------

    def _on_context_menu(self, pos) -> None:
        """右键单元格时弹出菜单，提供编辑歌词选项。"""
        item = self.table.itemAt(pos)
        if item is None:
            return
        row = item.row()
        menu = QMenu(self.table)
        act_edit = menu.addAction("编辑歌词文本")
        action = menu.exec(self.table.viewport().mapToGlobal(pos))
        if action == act_edit:
            # 延迟启动编辑，避免在菜单事件回调中直接调用 editItem 导致崩溃
            QTimer.singleShot(0, lambda: self._start_edit_text(row))

    def _start_edit_text(self, row: int) -> None:
        """将指定行的歌词列设为可编辑并进入编辑模式。"""
        if not (0 <= row < self.table.rowCount()):
            return
        text_item = self.table.item(row, 2)
        if text_item is None:
            return
        text_item.setFlags(text_item.flags() | Qt.ItemFlag.ItemIsEditable)
        self.table.editItem(text_item)

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        """编辑完成后同步回 self._lines，并将该单元格设回只读。"""
        if item.column() != 2:
            return
        row = item.row()
        if 0 <= row < len(self._lines):
            self._lines[row].text = item.text()
        # 延迟设回只读，避免在回调中直接修改 item flags 导致崩溃
        QTimer.singleShot(0, lambda it=item: self._reset_item_readonly(it))
        self._update_sel_label()

    def _reset_item_readonly(self, item: QTableWidgetItem) -> None:
        """将单元格设回只读。"""
        self.table.blockSignals(True)
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.table.blockSignals(False)

    # ---------------- 选中与微调 ----------------

    def _on_row_selection_changed(self) -> None:
        self._update_sel_label()

    def _update_sel_label(self) -> None:
        row = self.table.currentRow()
        enabled = 0 <= row < len(self._lines)
        if not enabled:
            self.sel_label.setText("未选中歌词")
            return
        ln = self._lines[row]
        self.sel_label.setText(
            f"已选第 {row + 1} 行：[{self._fmt_lrc(ln.start)}]  {ln.text}"
        )

    def _nudge_selected(self, delta_ms: int) -> None:
        if not self._lines:
            QMessageBox.information(self, "提示", "暂无歌词，请先处理。")
            return
        row = self.table.currentRow()
        if not (0 <= row < len(self._lines)):
            QMessageBox.information(self, "提示", "请先在表格中点选一行歌词。")
            return
        ln = self._lines[row]
        ln.shift_ms(delta_ms)
        # 按 start 时间重新排序，保证歌词始终按时间先后排列
        self._lines.sort(key=lambda x: x.start)
        self._populate()
        # 重新选中被调整的那行
        new_row = next((i for i, l in enumerate(self._lines) if l is ln), 0)
        self.table.selectRow(new_row)
        self._update_sel_label()
        sign = "+" if delta_ms >= 0 else ""
        self.setWindowTitle(f"歌词预览 · 第 {new_row + 1} 行已微调 {sign}{delta_ms} ms")

    def apply_global_offset(self) -> None:
        if not self._lines:
            QMessageBox.information(self, "提示", "暂无歌词可调整。")
            return
        delta = self.global_offset.value()
        if self.global_unit.currentText() == "ms":
            delta = delta * 10        # 末尾零机制
        else:
            delta = delta * 1000
        # 圆整到 10 的倍数（LRC 不支持个位数毫秒）
        delta = int(round(delta / 10) * 10)
        if delta == 0:
            return
        row = self.table.currentRow()
        for ln in self._lines:
            ln.shift_ms(delta)
        self._populate()
        if 0 <= row < len(self._lines):
            self.table.selectRow(row)

    # ---------------- 导出 ----------------

    def export(self, fmt: str) -> None:
        if not self._lines:
            QMessageBox.information(self, "提示", "暂无歌词，请先处理。")
            return
        default_name = (self._source_name or "lyrics") + f".{fmt}"
        default_path = os.path.join(get_config().output_dir, default_name)
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
        QMessageBox.information(self, "导出成功", f"已导出：{path}")

    # ---------------- 主题色 ----------------

    def _apply_theme_color(self) -> None:
        """预览窗口强制使用浅色白底，保证歌词可读性。"""
        colors = get_theme_colors("light")
        accent = colors["accent"]
        panel = colors["panel"]        # #ffffff
        border = colors["border"]
        text = colors["text"]
        text_strong = colors["text_strong"]
        text_muted = colors["text_muted"]
        self.setStyleSheet(
            f"""
            PreviewDialog {{ background-color: {colors['bg']}; }}
            QLabel {{ color: {text}; }}
            QLabel#title {{ color: {text_strong}; font-size: 15px; font-weight: 600; }}
            QLabel#hint {{ color: {text_muted}; }}
            QGroupBox {{
                color: {text_strong};
                border: 1px solid {border};
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 8px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }}
            QPushButton {{
                background-color: {colors['secondary']};
                color: {text};
                border: 1px solid {border};
                border-radius: 4px;
                padding: 4px 12px;
            }}
            QPushButton:hover {{ border-color: {accent}; }}
            QPushButton#secondary {{ background-color: {colors['panel_alt']}; }}
            QSpinBox, QComboBox {{
                background-color: {panel};
                color: {text};
                border: 1px solid {border};
                border-radius: 4px;
                padding: 2px 4px;
            }}
            QComboBox QAbstractItemView {{
                background-color: {panel};
                color: {text};
                selection-background-color: {accent};
                selection-color: #ffffff;
            }}
            /* 表格：白底，选中行蓝色高亮 */
            QTableWidget#lyricTable {{
                background-color: {panel};
                color: {text};
                border: 1px solid {border};
                gridline-color: {border};
            }}
            QTableWidget#lyricTable::item {{
                background-color: {panel};
                color: {text};
            }}
            QTableWidget#lyricTable::item:selected {{
                background-color: {accent};
                color: #ffffff;
            }}
            QHeaderView::section {{
                background-color: {colors['panel_alt']};
                color: {text};
                border: 1px solid {border};
                padding: 4px;
            }}
            /* 编辑器：不透明白底，避免底层文字透出造成重叠 */
            QTableWidget QLineEdit {{
                background-color: {panel};
                color: {text};
                border: 1px solid {accent};
                padding: 1px 2px;
            }}
            """
        )
