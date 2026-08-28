"""寬鬆 CER —— 把「書寫形式不同」從「引擎唸錯」裡扣掉。

為什麼需要這一版
----------------
`app.text_utils.normalize_for_cer()` 只處理標點、空白與全半形，**不做簡繁轉換、
也不統一數字書寫形式**。所以平台預設算出來的 CER 混了兩種與引擎品質無關的誤差：

    輸入「兩萬三千四百五十六」  ASR 寫「23456」     -> 整串都算錯
    引擎唸得完全正確            ASR 吐簡體字         -> 整段都算錯

第二種特別麻煩：Whisper 會依照它聽到的口音決定要寫簡體還是正體，所以「ASR 轉出
簡體」同時是**弱證據**（可能真的帶大陸腔）與**雜訊**（可能只是 ASR 的書寫習慣）。
兩者混在同一個數字裡，就分不出「這顆引擎唸錯了」還是「這顆引擎口音不同」。

所以拿 CER 下結論之前，兩版都要算：

    嚴格 CER    輸入什麼字就該唸出什麼字，書寫形式不同也算錯
    寬鬆 CER    先把簡繁與數字寫法正規化，只留下真正唸錯的部分

正規化做了什麼
--------------
1. 兩邊都用 zhconv 轉成正體字
2. 中文數字換成阿拉伯數字（一二三 -> 123）
3. 刪掉位值字十百千萬億 —— 「兩萬三千四百五十六」攤平成「23456」
   例外：「十」開頭的十位數要補回 1，否則「十七」會變成「7」而不是「17」
4. 拿掉數字之間的「點」（小數點的中文寫法）：「五點四」-> 「54」
5. 再送進 `normalize_for_cer()` 做原本的標點與全半形處理

第 3 步是**攤平**不是**展開**：「三千八百億」變成「38」而不是 380000000000。
展開會讓參考字串暴長，一個字唸錯就被放大成十幾個編輯距離，反而失真。攤平之後
兩邊用同一套規則處理，比出來的是「數字的字面序列一不一樣」。

已知落差
--------
這份定義是 2026-08-28 重新整理的。更早的一次性腳本算出來的寬鬆 CER 有少數格子
對不上（100 格裡 10 格，多數差在 0.01 以內、最大 0.09），舊定義已經找不回來了。
**同一份報告裡的所有引擎一定要用同一份定義重算**，不要拿新舊兩版混著比。
"""

from __future__ import annotations

import re
from typing import Optional

from app.text_utils import edit_distance, normalize_for_cer

# 中文數字 -> 阿拉伯數字。「兩」跟「二」都要收。
_DIGITS = {"零": "0", "〇": "0", "一": "1", "二": "2", "兩": "2", "三": "3",
           "四": "4", "五": "5", "六": "6", "七": "7", "八": "8", "九": "9"}
_UNITS = "十百千萬億"
_DOT_BETWEEN_DIGITS = re.compile(r"(?<=\d)[點点](?=\d)")


def _to_arabic(text: str) -> str:
    """中文數字攤平成阿拉伯數字，位值字刪掉。"""
    out: list[str] = []
    for ch in text:
        if ch in _DIGITS:
            out.append(_DIGITS[ch])
        elif ch in _UNITS:
            # 「十七」的十在最前面，代表 1；「二十一」的十前面已經有數字，直接刪
            if ch == "十" and not (out and out[-1].isdigit()):
                out.append("1")
        else:
            out.append(ch)
    return _DOT_BETWEEN_DIGITS.sub("", "".join(out))


def normalize_lenient(text: str) -> str:
    """寬鬆比對用的正規化字串。zhconv 沒裝的話會直接講清楚要裝什麼。"""
    try:
        import zhconv
    except ModuleNotFoundError as exc:      # pragma: no cover - 環境問題，不是邏輯
        raise ModuleNotFoundError(
            "寬鬆 CER 需要 zhconv 做簡繁轉換：pip install -r requirements-bench.txt"
        ) from exc
    return normalize_for_cer(_to_arabic(zhconv.convert(text or "", "zh-tw")))


def lenient_cer(reference: str, hypothesis: str) -> Optional[float]:
    """寬鬆字錯誤率。參考字串正規化後是空的就回 None（跟平台的嚴格版一致）。"""
    ref = normalize_lenient(reference)
    hyp = normalize_lenient(hypothesis)
    if not ref:
        return None
    return round(edit_distance(ref, hyp) / len(ref), 4)
