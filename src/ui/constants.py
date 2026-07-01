"""界面常量：语言列表、样式。"""

# 语言下拉框选项：(显示名, ISO 639-1 代码 or None=自动检测)
LANGUAGES = [
    ("自动检测", None),
    ("中文", "zh"),
    ("英语", "en"),
    ("日语", "ja"),
    ("韩语", "ko"),
    ("粤语", "yue"),
    ("法语", "fr"),
    ("德语", "de"),
    ("西班牙语", "es"),
    ("俄语", "ru"),
    ("意大利语", "it"),
    ("葡萄牙语", "pt"),
    ("泰语", "th"),
    ("越南语", "vi"),
    ("印尼语", "id"),
    ("阿拉伯语", "ar"),
]

# Whisper 模型档位
WHISPER_MODELS = [
    ("tiny (最快，精度低)", "tiny"),
    ("base (快)", "base"),
    ("small (均衡)", "small"),
    ("medium (推荐)", "medium"),
    ("large-v3 (最准，最慢)", "large-v3"),
]


# ---------- 主题样式 ----------
# 两套主题（深色 / 浅色），通过 get_qss(theme) 获取。
# 颜色变量化，保证两种模式下 UI 元素可读性与一致性。

_DARK_VARS = {
    "bg": "#1e1f26",
    "panel": "#2a2b32",
    "panel_alt": "#3a3b42",
    "border": "#3a3b42",
    "text": "#e6e6e6",
    "text_strong": "#ffffff",
    "text_muted": "#9aa0a6",
    "accent": "#2d7ff9",
    "accent_hover": "#1a6be0",
    "danger": "#d93025",
    "danger_hover": "#b3261e",
    "secondary": "#3a3b42",
    "secondary_hover": "#4a4b53",
    "disabled_bg": "#44454d",
    "disabled_fg": "#888888",
    "checkbox_border": "#5a5b63",
    "gpu_on": "#4dd0a8",
    "gpu_off": "#ffb74d",
    "row_alt": "#23242b",
}

_LIGHT_VARS = {
    "bg": "#f4f5f7",
    "panel": "#ffffff",
    "panel_alt": "#eceef1",
    "border": "#d4d7dd",
    "text": "#23242b",
    "text_strong": "#111317",
    "text_muted": "#6b7280",
    "accent": "#2d7ff9",
    "accent_hover": "#1a6be0",
    "danger": "#d93025",
    "danger_hover": "#b3261e",
    "secondary": "#e2e5ea",
    "secondary_hover": "#d3d7de",
    "disabled_bg": "#d9dbe0",
    "disabled_fg": "#a0a3aa",
    "checkbox_border": "#b3b7bf",
    "gpu_on": "#0f9d76",
    "gpu_off": "#c77700",
    "row_alt": "#f7f8fa",
}


def _render_qss(v: dict) -> str:
    return f"""
QMainWindow, QWidget#central {{ background: {v['bg']}; }}
QLabel {{ color: {v['text']}; }}
QLabel#title {{ font-size: 20px; font-weight: 600; color: {v['text_strong']}; }}
QLabel#hint {{ color: {v['text_muted']}; font-size: 12px; }}
QLabel#gpuStatus {{ font-weight: 600; }}
QGroupBox {{
    color: {v['text']}; border: 1px solid {v['border']}; border-radius: 6px;
    margin-top: 12px; padding-top: 10px; font-weight: 600;
}}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; }}
QPushButton {{
    background: {v['accent']}; color: white; border: none; padding: 8px 16px;
    border-radius: 5px; font-weight: 600;
}}
QPushButton:hover {{ background: {v['accent_hover']}; }}
QPushButton:disabled {{ background: {v['disabled_bg']}; color: {v['disabled_fg']}; }}
QPushButton#secondary {{ background: {v['secondary']}; color: {v['text']}; }}
QPushButton#secondary:hover {{ background: {v['secondary_hover']}; }}
QPushButton#danger {{ background: {v['danger']}; color: white; }}
QPushButton#danger:hover {{ background: {v['danger_hover']}; }}
QLineEdit, QComboBox, QSpinBox {{
    background: {v['panel']}; color: {v['text']}; border: 1px solid {v['border']};
    border-radius: 4px; padding: 5px 8px;
}}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus {{ border: 1px solid {v['accent']}; }}
QComboBox::drop-down {{ border: none; }}
QComboBox QAbstractItemView {{
    background: {v['panel']}; color: {v['text']}; selection-background-color: {v['accent']};
    selection-color: white;
}}
QProgressBar {{
    background: {v['panel']}; border: 1px solid {v['border']}; border-radius: 4px;
    text-align: center; color: {v['text']}; height: 18px;
}}
QProgressBar::chunk {{ background: {v['accent']}; border-radius: 3px; }}
QTableWidget {{
    background: {v['panel']}; color: {v['text']}; gridline-color: {v['border']};
    selection-background-color: {v['accent']}; selection-color: white;
    alternate-background-color: {v['row_alt']};
}}
QHeaderView::section {{ background: {v['panel_alt']}; color: {v['text']}; padding: 6px; border: none; }}
QCheckBox {{ color: {v['text']}; }}
QCheckBox::indicator {{ width: 16px; height: 16px; }}
QCheckBox::indicator:unchecked {{ background: {v['panel']}; border: 1px solid {v['checkbox_border']}; border-radius: 3px; }}
QCheckBox::indicator:checked {{ background: {v['accent']}; border: 1px solid {v['accent']}; border-radius: 3px; }}
QScrollArea {{ border: none; }}
QStatusBar {{ color: {v['text_muted']}; }}
QToolButton#themeBtn {{
    background: {v['secondary']}; color: {v['text']}; border: 1px solid {v['border']};
    border-radius: 5px; padding: 6px 12px; font-weight: 600;
}}
QToolButton#themeBtn:hover {{ background: {v['secondary_hover']}; }}
"""


# 主题名 -> 颜色变量
THEME_VARS = {"dark": _DARK_VARS, "light": _LIGHT_VARS}


def get_qss(theme: str = "dark") -> str:
    """返回指定主题的 QSS。theme: 'dark' | 'light'。"""
    return _render_qss(THEME_VARS.get(theme, _DARK_VARS))


def get_theme_colors(theme: str = "dark") -> dict:
    """返回指定主题的颜色变量（供代码内联设置样式用，如 GPU 状态文字）。"""
    return THEME_VARS.get(theme, _DARK_VARS)


# 向后兼容：默认深色 QSS
QSS = get_qss("dark")
