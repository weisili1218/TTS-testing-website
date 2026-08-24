"""評測結果的統計：Elo 排名、對戰矩陣、分佈量化。

為什麼不能只看勝率
------------------
兩個引擎比的時候，勝率就是答案。四個引擎兩兩對戰時勝率會騙人：一個模型如果
剛好比較常抽到弱的對手，勝率就會虛高。所以主排名用 **Elo** —— 它會把「贏了誰」
算進去，贏強的加比較多分，輸給弱的扣比較多分。

投票怎麼變成成對結果
--------------------
兩種盲測模式最後都化約成同一種東西 —— 一組 (贏家, 輸家) 或平手：

    對決模式：A/B 二選一        -> 1 組成對結果
    排名模式：C > A > D > B     -> 6 組成對結果（每兩個之間都有大小關係）

排名拆出來的 6 組**不是互相獨立的**，它們來自同一次聆聽、同一個人的同一個判斷。
當成 6 場獨立對戰灌進 Elo 會嚴重高估這一筆的份量，所以排名來源的每一組
K 值除以 (n-1)，讓「一次排名」跟「一次對決」對排名的影響力量級接近。

信賴區間
--------
Elo 分數在樣本少的時候非常抖，只看點估計會over-read。所以做 bootstrap：
把成對結果重抽樣幾百次、每次重算 Elo，取 5%–95% 百分位當區間。
區間重疊 = 這兩個模型目前分不出高下，**不要下結論**。
"""

from __future__ import annotations

import math
import random
from typing import Iterable, Optional

from . import config

# 排名模式拆出來的成對結果會被標成這個來源，Elo 據此調整 K
SOURCE_DUEL = "duel"
SOURCE_RANKING = "ranking"


# ---------------------------------------------------------------- 分佈
def mean(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def median(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = sorted(float(v) for v in values if v is not None)
    if not vals:
        return None
    mid = len(vals) // 2
    return round(vals[mid] if len(vals) % 2 else (vals[mid - 1] + vals[mid]) / 2, 4)


def percentile(values: Iterable[Optional[float]], pct: float) -> Optional[float]:
    """線性內插的百分位數。p95 用來看「最糟的情況有多糟」。"""
    vals = sorted(float(v) for v in values if v is not None)
    if not vals:
        return None
    if len(vals) == 1:
        return round(vals[0], 4)
    pos = (len(vals) - 1) * max(0.0, min(1.0, pct))
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return round(vals[lo], 4)
    return round(vals[lo] + (vals[hi] - vals[lo]) * (pos - lo), 4)


def stdev(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None]
    if len(vals) < 2:
        return None
    m = sum(vals) / len(vals)
    return round(math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1)), 4)


def cv(values: Iterable[Optional[float]]) -> Optional[float]:
    """變異係數（標準差 ÷ 平均）—— 穩定性指標，跟單位無關所以跨引擎可比。"""
    vals = [float(v) for v in values if v is not None]
    if len(vals) < 2:
        return None
    m = sum(vals) / len(vals)
    if abs(m) < 1e-9:
        return None
    sd = stdev(vals)
    return round(sd / m, 4) if sd is not None else None


def spread(values: Iterable[Optional[float]]) -> dict:
    """一組數字的完整摘要，速度／CER 都用這個。"""
    vals = [float(v) for v in values if v is not None]
    return {
        "n": len(vals),
        "mean": mean(vals),
        "median": median(vals),
        "min": round(min(vals), 4) if vals else None,
        "max": round(max(vals), 4) if vals else None,
        "p95": percentile(vals, 0.95),
        "stdev": stdev(vals),
        "cv": cv(vals),
    }


# ---------------------------------------------------------------- 成對結果
def pairs_from_vote(trial: dict) -> list[dict]:
    """把一筆評測的投票拆成成對結果。沒投票或投了「都不行」就回空的。

    「都不行」刻意不產生任何成對結果：它表達的是「兩個都不及格」，
    並沒有說誰比較好，硬要記成平手會污染排名。
    """
    vote = trial.get("vote")
    if not vote:
        return []
    slots: dict = trial.get("slots", {})
    if not slots:
        return []

    created = trial.get("created_at", "")
    trial_id = trial.get("id", "")
    category = trial.get("category", "")

    def _pair(win: str, lose: str, source: str, group: int, tie: bool = False) -> dict:
        return {"winner": win, "loser": lose, "tie": tie, "source": source,
                "group_size": group, "trial_id": trial_id,
                "created_at": created, "category": category}

    # --- 排名模式：C > A > D > B 拆成每兩個之間的大小關係 ---
    ranking = vote.get("ranking")
    if ranking:
        keys = [slots.get(s) for s in ranking if slots.get(s)]
        n = len(keys)
        if n < 2:
            return []
        out = []
        for i in range(n):
            for j in range(i + 1, n):
                out.append(_pair(keys[i], keys[j], SOURCE_RANKING, n))
        return out

    # --- 對決模式 ---
    choice = vote.get("choice")
    if choice == "tie":
        keys = [k for k in slots.values() if k]
        if len(keys) != 2:
            return []
        return [_pair(keys[0], keys[1], SOURCE_DUEL, 2, tie=True)]
    if choice == "bad":
        return []
    winner = vote.get("winner")
    loser = vote.get("loser")
    if winner and loser:
        return [_pair(winner, loser, SOURCE_DUEL, 2)]
    return []


def all_pairs(trials: list[dict]) -> list[dict]:
    """全部評測的成對結果，依時間由舊到新排好（Elo 是有順序的）。"""
    out: list[dict] = []
    for trial in trials:
        out.extend(pairs_from_vote(trial))
    out.sort(key=lambda p: (p.get("created_at") or "", p.get("trial_id") or ""))
    return out


# ---------------------------------------------------------------- Elo
def _expected(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def elo(pairs: list[dict], keys: Iterable[str],
        k: Optional[float] = None, start: Optional[float] = None) -> dict[str, float]:
    """依時間順序重播所有成對結果，算出 Elo 分數。

    排名模式拆出來的成對結果 K 值除以 (group_size - 1)：一次四方排名會產生
    6 組結果，但它們來自同一個判斷，不該有 6 場獨立對戰的份量。
    """
    k = config.ELO_K if k is None else k
    start = config.ELO_START if start is None else start
    ratings: dict[str, float] = {key: float(start) for key in keys}

    for pair in pairs:
        a, b = pair["winner"], pair["loser"]
        if a not in ratings or b not in ratings or a == b:
            continue
        weight = 1.0
        if pair.get("source") == SOURCE_RANKING:
            weight = 1.0 / max(1, int(pair.get("group_size", 2)) - 1)
        eff_k = k * weight

        exp_a = _expected(ratings[a], ratings[b])
        score_a = 0.5 if pair.get("tie") else 1.0
        ratings[a] += eff_k * (score_a - exp_a)
        ratings[b] += eff_k * ((1.0 - score_a) - (1.0 - exp_a))
    return {key: round(v, 1) for key, v in ratings.items()}


def elo_intervals(pairs: list[dict], keys: list[str],
                  rounds: Optional[int] = None) -> dict[str, dict]:
    """bootstrap 出 Elo 的信賴區間。

    區間重疊代表兩個模型目前分不出高下 —— 這比點估計誠實得多，
    樣本少的時候 Elo 的點估計幾乎沒有意義。
    """
    rounds = config.ELO_BOOTSTRAP if rounds is None else rounds
    point = elo(pairs, keys)
    if rounds <= 0 or len(pairs) < 4:
        return {key: {"elo": point[key], "low": None, "high": None,
                      "samples": len(pairs)} for key in keys}

    rng = random.Random(20260821)   # 固定種子：同一批資料每次看到的區間一樣
    samples: dict[str, list[float]] = {key: [] for key in keys}
    n = len(pairs)
    for _ in range(rounds):
        resampled = [pairs[rng.randrange(n)] for _ in range(n)]
        resampled.sort(key=lambda p: (p.get("created_at") or "", p.get("trial_id") or ""))
        for key, value in elo(resampled, keys).items():
            samples[key].append(value)

    out = {}
    for key in keys:
        vals = samples[key]
        out[key] = {
            "elo": point[key],
            "low": percentile(vals, 0.05),
            "high": percentile(vals, 0.95),
            "samples": n,
        }
    return out


# ---------------------------------------------------------------- 對戰矩陣
def win_matrix(pairs: list[dict], keys: list[str]) -> dict:
    """誰對誰打了幾場、贏幾場。四個模型時這張表比總勝率有用得多。"""
    cells: dict[str, dict[str, dict]] = {
        a: {b: {"wins": 0, "losses": 0, "ties": 0} for b in keys if b != a}
        for a in keys
    }
    for pair in pairs:
        a, b = pair["winner"], pair["loser"]
        if a not in cells or b not in cells.get(a, {}):
            continue
        if pair.get("tie"):
            cells[a][b]["ties"] += 1
            cells[b][a]["ties"] += 1
        else:
            cells[a][b]["wins"] += 1
            cells[b][a]["losses"] += 1

    rows = []
    for a in keys:
        row = {"engine": a, "cells": {}}
        for b in keys:
            if b == a:
                row["cells"][b] = None
                continue
            c = cells[a][b]
            decided = c["wins"] + c["losses"]
            row["cells"][b] = {
                **c,
                "games": decided + c["ties"],
                "win_rate": round(c["wins"] / decided, 3) if decided else None,
            }
        rows.append(row)
    return {"engines": keys, "rows": rows}


def head_to_head_totals(pairs: list[dict], keys: list[str]) -> dict[str, dict]:
    """每個引擎的總戰績（勝／敗／平／勝率）。"""
    out = {key: {"wins": 0, "losses": 0, "ties": 0} for key in keys}
    for pair in pairs:
        a, b = pair["winner"], pair["loser"]
        if a not in out or b not in out:
            continue
        if pair.get("tie"):
            out[a]["ties"] += 1
            out[b]["ties"] += 1
        else:
            out[a]["wins"] += 1
            out[b]["losses"] += 1
    for key, row in out.items():
        decided = row["wins"] + row["losses"]
        row["games"] = decided + row["ties"]
        row["win_rate"] = round(row["wins"] / decided, 3) if decided else None
    return out
