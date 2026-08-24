"""文字前處理：注音括號安全的斷句與分段。

有些中文 TTS 用 `字[:ㄘ3]` 這種括號語法標注音，切段時絕對不能切在括號中間，
所以這裡所有掃描都會追蹤中括號深度。沒用到注音標註的引擎不受影響。
"""

from __future__ import annotations

import re
from typing import Optional

# 句尾標點（會保留在該段結尾）
SENTENCE_END = "。！？!?；;\n"
# 次級斷點（句子太長時才用）
CLAUSE_END = "，,、：: "


def strip_bopomofo(text: str) -> str:
    """把 `字[:ㄘ3]` 還原成 `字`，用來計算「真正的字數」與顯示。"""
    return re.sub(r"\[:[^\]]*\]", "", text)


def visible_length(text: str) -> int:
    return len(strip_bopomofo(text).strip())


def _split_keep(text: str, delims: str) -> list[str]:
    """依 delims 斷句，分隔符保留在前一段結尾；中括號內不斷。"""
    out: list[str] = []
    buf: list[str] = []
    depth = 0
    for ch in text:
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth = max(0, depth - 1)
        buf.append(ch)
        if depth == 0 and ch in delims:
            piece = "".join(buf).strip()
            if piece:
                out.append(piece)
            buf = []
    tail = "".join(buf).strip()
    if tail:
        out.append(tail)
    return out


def _hard_chunk(text: str, max_chars: int) -> list[str]:
    """最後手段：照字數硬切，但不切在括號中間。"""
    out: list[str] = []
    buf: list[str] = []
    depth = 0
    count = 0
    for ch in text:
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth = max(0, depth - 1)
        buf.append(ch)
        if depth == 0:
            count += 1
            if count >= max_chars:
                out.append("".join(buf).strip())
                buf, count = [], 0
    tail = "".join(buf).strip()
    if tail:
        out.append(tail)
    return [p for p in out if p]


def split_segments(text: str, max_chars: int = 60) -> list[str]:
    """把長文切成適合逐段合成的片段。"""
    text = (text or "").strip()
    if not text:
        return []

    sentences = _split_keep(text, SENTENCE_END)
    if not sentences:
        sentences = [text]

    # 句子本身太長 -> 再依逗號切 -> 還是太長就硬切
    pieces: list[str] = []
    for s in sentences:
        if visible_length(s) <= max_chars:
            pieces.append(s)
            continue
        for c in _split_keep(s, CLAUSE_END):
            if visible_length(c) <= max_chars:
                pieces.append(c)
            else:
                pieces.extend(_hard_chunk(c, max_chars))

    # 把太短的片段往前合併，減少段數
    merged: list[str] = []
    for p in pieces:
        if merged and visible_length(merged[-1]) + visible_length(p) <= max_chars:
            merged[-1] = merged[-1] + p
        else:
            merged.append(p)

    return [m.strip() for m in merged if m.strip()]


def ensure_terminated(text: str) -> str:
    """多數 TTS 對句尾有標點的輸入表現比較穩，缺的話補一個句號。"""
    t = text.strip()
    if not t:
        return t
    if strip_bopomofo(t)[-1:] in "。！？!?、，,;；:：…":
        return t
    return t + "。"


# --------------------------------------------------------------- 客觀評分
# 比對「輸入文字」與「把合成音檔丟回 ASR 得到的文字」時，標點、空白、
# 全半形差異都不算錯，只比實際唸出來的字。
_CER_KEEP = re.compile(r"[^0-9a-zA-Z一-鿿㐀-䶿]")


def normalize_for_cer(text: str) -> str:
    """轉成只剩中日韓漢字與英數字的比對字串。"""
    t = strip_bopomofo(text or "")
    # 全形英數 -> 半形，方便比對
    t = "".join(
        chr(ord(ch) - 0xFEE0) if "！" <= ch <= "～" else ch for ch in t
    )
    return _CER_KEEP.sub("", t).lower()


def edit_distance(a: str, b: str) -> int:
    """標準 Levenshtein 距離（只留兩列，長文也不吃記憶體）。"""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]


def char_error_rate(reference: str, hypothesis: str) -> Optional[dict]:
    """字錯誤率：(增 + 刪 + 改) / 參考字數。0 = 完全一致，越低越好。

    回傳 None 代表參考字串正規化後是空的，算不出來。
    注意 CER 可能大於 1（ASR 亂加字時）。
    """
    ref = normalize_for_cer(reference)
    hyp = normalize_for_cer(hypothesis)
    if not ref:
        return None
    dist = edit_distance(ref, hyp)
    return {
        "cer": round(dist / len(ref), 4),
        "edits": dist,
        "ref_chars": len(ref),
        "hyp_chars": len(hyp),
        "ref_norm": ref,
        "hyp_norm": hyp,
    }
