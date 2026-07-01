"""歌词生成：将 faster-whisper 转写结果整理为 LRC。

数据结构：
- LyricLine: 一行歌词，含 start/end (秒) 和 word-level 时间戳列表
- word_timestamps: [{word, start, end}, ...]
"""
from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class WordTS:
    word: str
    start: float  # 秒
    end: float

    def copy(self) -> "WordTS":
        return WordTS(self.word, self.start, self.end)


@dataclass
class LyricLine:
    text: str
    start: float  # 行起始 秒
    end: float    # 行结束 秒
    words: list[WordTS] = field(default_factory=list)

    def shift_ms(self, delta_ms: int) -> None:
        """对该行整体时间戳偏移（可正可负）。"""
        s = delta_ms / 1000.0
        self.start = max(0.0, self.start + s)
        self.end = max(0.0, self.end + s)
        for w in self.words:
            w.start = max(0.0, w.start + s)
            w.end = max(0.0, w.end + s)


# ---------- 格式化 ----------

def _fmt_lrc_time(seconds: float) -> str:
    """[mm:ss.xx]"""
    if seconds < 0:
        seconds = 0.0
    m = int(seconds // 60)
    s = seconds - m * 60
    return f"[{m:02d}:{s:05.2f}]"


# ---------- 构建 LyricLine ----------

def build_lines(segments) -> list[LyricLine]:
    """从 faster-whisper segments 构建 LyricLine 列表。

    segments: 迭代器，每个含 .text, .start, .end, .words(list)
    """
    lines: list[LyricLine] = []
    for seg in segments:
        text = (seg.text or "").strip()
        words = []
        for w in getattr(seg, "words", None) or []:
            wtext = (w.word or "").strip()
            if wtext == "":
                continue
            ws = w.start if w.start is not None else seg.start
            we = w.end if w.end is not None else seg.end
            words.append(WordTS(wtext, float(ws), float(we)))
        lines.append(LyricLine(
            text=text,
            start=float(seg.start),
            end=float(seg.end),
            words=words,
        ))
    return lines


# ---------- 导出 ----------

def to_standard_lrc(lines: Iterable[LyricLine], header: dict | None = None) -> str:
    """标准逐行 LRC。"""
    out: list[str] = []
    if header:
        for k, v in header.items():
            out.append(f"[{k}:{v}]")
        out.append("")
    for ln in lines:
        out.append(f"{_fmt_lrc_time(ln.start)}{ln.text}")
    return "\n".join(out) + "\n"


def to_plain_text(lines: Iterable[LyricLine]) -> str:
    """纯文本歌词（仅歌词文本，每行一句，不含时间轴）。"""
    out: list[str] = []
    for ln in lines:
        text = (ln.text or "").strip()
        if text:
            out.append(text)
    return "\n".join(out) + "\n"


def write_text(path: str, text: str, encoding: str = "utf-8") -> None:
    with open(path, "w", encoding=encoding, newline="\n") as f:
        f.write(text)
