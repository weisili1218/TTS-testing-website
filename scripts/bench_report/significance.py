"""成對顯著性檢定 —— 判斷兩顆引擎的差距是真的，還是抽樣抖出來的。

為什麼不用 t 檢定
-----------------
耗時與 CER 都是**重尾**分布：長段落的耗時是短句的十倍，CER 有一堆 0 再加上幾句
整句崩潰的 0.7。平均數與 t 檢定會被那幾句拉著走。所以一律用無母數的方法：

    Wilcoxon 符號等級檢定   差距的方向與大小排名，主檢定
    符號檢定                只看方向（誰贏），對極端值最不敏感
    中位數的信賴區間        差距到底有多大，比 p 值好用

為什麼是「成對」
----------------
每一句都同時送進所有引擎，所以第 i 句的難度對每顆引擎都一樣。成對比較把「這句
本來就難」這個共同變異消掉，剩下的才是引擎之間的差別。獨立樣本檢定會把句子難度
的變異也算進雜訊裡，白白損失檢定力。

怎麼讀結果
----------
`p` 小代表「差距不太可能是抽樣抖出來的」，**不代表差距很大**。差距多大要看
`diff_ci`（差值中位數與 95% 區間）與效果量 `effect_r`。n = 50 的時候，
r ≈ 0.1 是幾乎沒差、0.3 中等、0.5 以上很明顯。

三顆以上引擎要注意：兩兩各跑一次就是多重比較，p 值沒有校正。這裡刻意不做校正
（Bonferroni 之類），因為報告是拿來看趨勢的，不是拿來宣告發現。要當結論用的話
自己把門檻拉嚴。
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

Number = Optional[float]


def _paired(a: Sequence[Number], b: Sequence[Number]) -> list[tuple[float, float]]:
    return [(x, y) for x, y in zip(a, b) if x is not None and y is not None]


def _norm_cdf(z: float) -> float:
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


# ---------------------------------------------------------------- Wilcoxon
def wilcoxon(a: Sequence[Number], b: Sequence[Number]) -> dict:
    """成對 Wilcoxon 符號等級檢定（差值 = a - b）。

    用常態近似 + 連續性校正，平手的等級取平均。差值為 0 的配對直接排除 ——
    這是 Wilcoxon 的標準做法，所以回傳的 n 會小於配對數。
    """
    diffs = [x - y for x, y in _paired(a, b) if x != y]
    n = len(diffs)
    if n < 1:
        return {"n": 0}

    order = sorted(range(n), key=lambda i: abs(diffs[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:                                  # 同樣大小的差值共用平均等級
        j = i
        while j + 1 < n and abs(diffs[order[j + 1]]) == abs(diffs[order[i]]):
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1

    w_pos = sum(r for r, d in zip(ranks, diffs) if d > 0)
    w_neg = sum(r for r, d in zip(ranks, diffs) if d < 0)
    w = min(w_pos, w_neg)
    mu = n * (n + 1) / 4
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    z = (w - mu + 0.5) / sigma if sigma else 0.0
    p = 2 * _norm_cdf(z) if z < 0 else 2 * (1 - _norm_cdf(z))
    return {"n": n, "W": round(w, 1), "W_pos": round(w_pos, 1), "W_neg": round(w_neg, 1),
            "z": round(z, 3), "p": round(max(0.0, min(1.0, p)), 6),
            "effect_r": round(abs(z) / math.sqrt(n), 3)}


# ---------------------------------------------------------------- 符號檢定
def _binom_two_sided(k: int, n: int) -> float:
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def sign_test(a: Sequence[Number], b: Sequence[Number]) -> dict:
    """a - b 的符號檢定。pos = a 比較大的句數（CER 與耗時都是越小越好，要留意方向）。"""
    pairs = _paired(a, b)
    pos = sum(1 for x, y in pairs if x > y)
    neg = sum(1 for x, y in pairs if x < y)
    return {"n": pos + neg, "pos": pos, "neg": neg,
            "ties": len(pairs) - pos - neg,
            "p": round(_binom_two_sided(min(pos, neg), pos + neg), 6)}


# ---------------------------------------------------------------- 區間
def diff_ci(a: Sequence[Number], b: Sequence[Number]) -> dict:
    """成對差的中位數，與中位數的 95% 無母數信賴區間。

    用 order statistic 取界（常態近似決定要取第幾名），不是差值分布的 2.5/97.5
    百分位 —— 後者描述的是「差距的散佈範圍」，不是「中位數有多確定」，兩者差很多：
    同一批資料前者可能是 0.58–12.2，後者是 2.98–4.47。
    """
    diffs = sorted(x - y for x, y in _paired(a, b))
    n = len(diffs)
    if not n:
        return {}
    mid = diffs[n // 2] if n % 2 else (diffs[n // 2 - 1] + diffs[n // 2]) / 2
    out = {"median": round(mid, 4)}
    k = math.ceil(n / 2 - 1.96 * math.sqrt(n) / 2)      # 1-based 下界秩
    if 1 <= k <= n // 2:
        out["lo95"] = round(diffs[k - 1], 4)
        out["hi95"] = round(diffs[n - k], 4)
    return out
