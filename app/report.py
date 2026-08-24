"""排行榜與報表：把一堆評測紀錄聚合成看得懂的結論。

跟舊版最大的差別
----------------
舊版只有「平均耗時 + 勝率」。兩個引擎時那樣就夠了，四個引擎時會誤導：

* **勝率會被對手組成左右** —— 常抽到弱對手的模型勝率虛高。主排名改用 Elo。
* **平均值藏住尾巴** —— 一個平均 2 秒但偶爾 15 秒的模型，跟穩定 3 秒的模型，
  平均看起來前者贏。所以每個速度指標都給 p95 與變異係數。
* **總分蓋掉分類差異** —— 「誰比較好」通常沒有單一答案，而是「數字誰強、
  破音字誰強」。所以拆分類戰報。

判讀警語（caveats）
-------------------
這個模組會主動產生「你現在還不能下這個結論」的提醒，例如樣本太少、
Elo 信賴區間重疊、CER 差距小於 ASR 自己的誤差。報表最容易出事的地方
不是算錯，是**算對了但被過度解讀**。
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from datetime import datetime, timezone

from . import asr as asr_mod
from . import bench, config, engines, stats, voices

log = logging.getLogger("tts.report")

# 樣本數低於這個就標「還不夠下結論」
MIN_TRIALS_FOR_SPEED = 3
MIN_VOTES_FOR_ELO = 10


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


# ---------------------------------------------------------------- ASR 底線
_asr_baseline_cache: Optional[dict] = None


def measure_asr_baseline(voice_id: str = "") -> dict:
    """量這個 ASR 自己有多準：真人錄音的 CER 就是所有引擎的實務下限。

    沒有這個數字，就沒辦法判讀「CER 0.12 到底算好還壞」——
    如果真人錄音也有 0.10，那 0.12 其實已經很接近上限了。
    """
    global _asr_baseline_cache

    candidates = [v for v in voices.list_voices()
                  if v.get("has_audio") and v.get("transcription")]
    if voice_id:
        candidates = [v for v in candidates if v["id"] == voice_id] or candidates
    if not candidates:
        raise ValueError("音色庫裡沒有「有參考音也有逐字稿」的音色，量不了基準線")

    v = candidates[0]
    result = asr_mod.baseline(voices.prompt_path(v["id"]), v["transcription"])
    _asr_baseline_cache = {
        "voice_id": v["id"],
        "voice_name": v["name"],
        "cer": result.get("cer"),
        "asr_text": result.get("asr_text"),
        "reference_text": v["transcription"],
        "error": result.get("error"),
        "measured_at": _now(),
    }
    return _asr_baseline_cache


def asr_baseline() -> Optional[dict]:
    return _asr_baseline_cache


# ---------------------------------------------------------------- 聚合
def _collect(trials: list[dict]) -> dict[str, dict]:
    """把每個引擎在所有評測裡的原始數字收集起來。"""
    known = {a.key: a.label for a in engines.all_engines()}
    for trial in trials:
        for key, sample in trial.get("samples", {}).items():
            known.setdefault(key, sample.get("engine_label") or key)

    buckets: dict[str, dict] = {
        key: {
            "engine": key, "engine_label": label,
            "trials": 0, "ok": 0, "failed": 0,
            "elapsed": [], "rtf": [], "realtime": [], "first": [], "duration": [],
            "cer": [], "run_elapsed": [],
            "silence": [], "bandwidth_ratio": [], "lead_silence": [],
            "clip": [], "rms": [], "cps": [],
            "sample_rates": set(), "voice_labels": set(),
            "custom_ref_trials": 0,
            "errors": {},
            "warnings": {},
        }
        for key, label in known.items()
    }

    for trial in trials:
        for key, sample in trial.get("samples", {}).items():
            b = buckets.get(key)
            if b is None:
                continue
            b["trials"] += 1
            if sample.get("error") or not sample.get("filename"):
                b["failed"] += 1
                err = str(sample.get("error") or "未知錯誤")[:120]
                b["errors"][err] = b["errors"].get(err, 0) + 1
                continue

            b["ok"] += 1
            b["elapsed"].append(sample.get("elapsed_seconds"))
            b["rtf"].append(sample.get("rtf"))
            b["realtime"].append(sample.get("realtime_x"))
            b["first"].append(sample.get("first_chunk_seconds"))
            b["duration"].append(sample.get("duration_seconds"))
            # 重複跑的每一次都收進來，穩定性要看單次分佈而不是每筆的中位數
            for run in sample.get("runs") or []:
                b["run_elapsed"].append(run.get("elapsed_seconds"))
            if sample.get("cer") is not None:
                b["cer"].append(sample["cer"])
            if sample.get("voice_label"):
                b["voice_labels"].add(sample["voice_label"])
            if sample.get("uses_custom_reference"):
                b["custom_ref_trials"] += 1
            if sample.get("sample_rate"):
                b["sample_rates"].add(int(sample["sample_rate"]))

            m = sample.get("audio_metrics") or {}
            if not m.get("error"):
                b["silence"].append(m.get("silence_ratio"))
                b["bandwidth_ratio"].append(m.get("bandwidth_ratio"))
                b["lead_silence"].append(m.get("lead_silence_seconds"))
                b["clip"].append(m.get("clip_ratio"))
                b["rms"].append(m.get("rms_dbfs"))
                b["cps"].append(m.get("chars_per_voiced_second"))
            for w in sample.get("audio_warnings") or []:
                head = w.split("，")[0][:60]
                b["warnings"][head] = b["warnings"].get(head, 0) + 1

    return buckets


def _category_table(trials: list[dict], keys: list[str]) -> dict:
    """分類戰報：每個引擎在每一類文本上的 CER 與速度。

    「哪個模型最好」通常沒有單一答案。數字唸得準的可能破音字很爛，
    這張表才是換模型時真正在看的東西。
    """
    cells: dict[str, dict[str, dict]] = {}
    for trial in trials:
        cat = trial.get("category") or "custom"
        row = cells.setdefault(cat, {})
        for key, sample in trial.get("samples", {}).items():
            if key not in keys or not sample.get("filename"):
                continue
            cell = row.setdefault(key, {"cer": [], "elapsed": [], "rtf": [], "n": 0})
            cell["n"] += 1
            if sample.get("cer") is not None:
                cell["cer"].append(sample["cer"])
            cell["elapsed"].append(sample.get("elapsed_seconds"))
            cell["rtf"].append(sample.get("rtf"))

    out: dict[str, Any] = {}
    for cat, row in cells.items():
        summarized = {}
        for key, cell in row.items():
            summarized[key] = {
                "n": cell["n"],
                "cer": stats.median(cell["cer"]),
                "cer_n": len(cell["cer"]),
                "elapsed": stats.median(cell["elapsed"]),
                "rtf": stats.median(cell["rtf"]),
            }
        scored = {k: v["cer"] for k, v in summarized.items() if v["cer"] is not None}
        fastest = {k: v["elapsed"] for k, v in summarized.items() if v["elapsed"] is not None}
        out[cat] = {
            "label": bench.CATEGORY_LABELS.get(cat, cat),
            "engines": summarized,
            "best_cer": min(scored, key=scored.get) if scored else None,
            "fastest": min(fastest, key=fastest.get) if fastest else None,
            "trials": sum(1 for t in trials if (t.get("category") or "custom") == cat),
        }
    return out


# ---------------------------------------------------------------- 判讀警語
def _caveats(rows: list[dict], pairs: list[dict], baseline: Optional[dict]) -> list[str]:
    """自動產生「現在還不能下這個結論」的提醒。"""
    out: list[str] = []
    active = [r for r in rows if r["trials"] > 0]
    if not active:
        return ["還沒有任何評測紀錄。"]

    thin = [r["engine_label"] for r in active if r["ok"] < MIN_TRIALS_FOR_SPEED]
    if thin:
        out.append(f"{'、'.join(thin)} 成功筆數少於 {MIN_TRIALS_FOR_SPEED} 筆，"
                   "速度數字還只是參考值。")

    if len(pairs) < MIN_VOTES_FOR_ELO:
        out.append(f"目前只有 {len(pairs)} 組成對比較，Elo 幾乎沒有鑑別力，"
                   f"至少累積 {MIN_VOTES_FOR_ELO} 組再看排名。")
    else:
        ranked = [r for r in active if r.get("elo_low") is not None]
        ranked.sort(key=lambda r: -(r["elo"] or 0))
        overlapping = []
        for i in range(len(ranked) - 1):
            a, b = ranked[i], ranked[i + 1]
            if a["elo_low"] is not None and b["elo_high"] is not None \
                    and a["elo_low"] <= b["elo_high"]:
                overlapping.append(f"{a['engine_label']} 與 {b['engine_label']}")
        if overlapping:
            out.append(f"{'、'.join(overlapping)} 的 Elo 信賴區間重疊，"
                       "以目前的樣本分不出高下。")

    # CER 差距要拿 ASR 自己的誤差當尺 —— 差距比雜訊還小就不是差距
    scored = [r for r in active if r.get("cer") is not None]
    if len(scored) >= 2:
        scored.sort(key=lambda r: r["cer"])
        gap = scored[1]["cer"] - scored[0]["cer"]
        floor = (baseline or {}).get("cer")
        if floor is not None and gap < floor * 0.25:
            out.append(
                f"{scored[0]['engine_label']} 與 {scored[1]['engine_label']} 的 CER "
                f"只差 {gap:.3f}，相對於 ASR 自己的誤差底線（{floor:.3f}）太小，"
                "不足以說誰的吐字比較準。")
    if not baseline:
        out.append("還沒量 ASR 誤差底線，CER 只能拿來互相比大小，"
                   "沒辦法判斷絕對水準。按排行榜上的「量 ASR 誤差底線」。")

    # 取樣率不一致時提醒：音質觀感會被取樣率影響，跟模型本身的好壞是兩件事
    rates = {r["engine_label"]: r["sample_rate"] for r in active if r.get("sample_rate")}
    if len(set(rates.values())) > 1:
        detail = "、".join(f"{k} {v}Hz" for k, v in rates.items())
        out.append(f"各引擎輸出取樣率不同（{detail}），盲測時高取樣率天生佔便宜，"
                   "判斷音質差異時要把這點算進去。")

    mixed_voice = [r["engine_label"] for r in active
                   if r.get("custom_ref_trials") and r["custom_ref_trials"] < r["ok"]]
    if mixed_voice:
        out.append(f"{'、'.join(mixed_voice)} 有些筆數用了指定參考音、有些沒有，"
                   "音色相關的比較請改看單筆紀錄。")
    return out


# ---------------------------------------------------------------- 排行榜
def leaderboard(limit: Optional[int] = None) -> dict:
    """把所有評測聚合成每個引擎的完整戰報。"""
    trials = bench.list_trials(limit=limit or config.TRIALS_LIMIT)
    buckets = _collect(trials)
    keys = list(buckets)

    pairs = stats.all_pairs(trials)
    intervals = stats.elo_intervals(pairs, keys)
    totals = stats.head_to_head_totals(pairs, keys)

    rows: list[dict] = []
    for key, b in buckets.items():
        h2h = totals.get(key, {})
        rates = sorted(b["sample_rates"])
        rows.append({
            "engine": key,
            "engine_label": b["engine_label"],
            "trials": b["trials"],
            "ok": b["ok"],
            "failed": b["failed"],
            "success_rate": round(b["ok"] / b["trials"], 3) if b["trials"] else None,
            "voice": "／".join(sorted(b["voice_labels"])) or "—",
            "custom_ref_trials": b["custom_ref_trials"],
            "sample_rate": rates[-1] if rates else None,
            "sample_rates": rates,

            # --- 主觀排名 ---
            "elo": intervals[key]["elo"] if b["trials"] else None,
            "elo_low": intervals[key]["low"],
            "elo_high": intervals[key]["high"],
            "wins": h2h.get("wins", 0),
            "losses": h2h.get("losses", 0),
            "ties": h2h.get("ties", 0),
            "games": h2h.get("games", 0),
            "win_rate": h2h.get("win_rate"),

            # --- 速度 ---
            "elapsed": stats.spread(b["elapsed"]),
            "rtf": stats.spread(b["rtf"]),
            "realtime_x": stats.median(b["realtime"]),
            "first_chunk": stats.spread(b["first"]),
            "duration": stats.median(b["duration"]),
            # 穩定性用「每一次跑」的分佈，不是每筆的中位數
            "stability_cv": stats.cv(b["run_elapsed"]),
            "runs_measured": len(b["run_elapsed"]),

            # --- 吐字 ---
            "cer": stats.median(b["cer"]),
            "cer_spread": stats.spread(b["cer"]),
            "scored_count": len(b["cer"]),

            # --- 音訊客觀指標 ---
            "audio": {
                "silence_ratio": stats.median(b["silence"]),
                "bandwidth_ratio": stats.median(b["bandwidth_ratio"]),
                "lead_silence": stats.median(b["lead_silence"]),
                "clip_ratio": stats.median(b["clip"]),
                "rms_dbfs": stats.median(b["rms"]),
                "chars_per_voiced_second": stats.median(b["cps"]),
            },
            "audio_warnings": sorted(b["warnings"].items(), key=lambda kv: -kv[1])[:4],
            "top_errors": sorted(b["errors"].items(), key=lambda kv: -kv[1])[:3],
        })

    # 主排序：有投票紀錄的排前面，其中 Elo 高的在前。
    # 沒投過票的引擎 Elo 都停在起始分，混在中間排會看起來像「跟第二名同級」。
    rows.sort(key=lambda r: (0 if r["games"] else 1, -(r["elo"] or 0), r["engine"]))

    baseline = asr_baseline()
    active_keys = [r["engine"] for r in rows if r["trials"] > 0]

    return {
        "engines": rows,
        "matrix": stats.win_matrix(pairs, active_keys),
        "categories": _category_table(trials, active_keys),
        "category_labels": bench.CATEGORY_LABELS,
        "totals": {
            "trials": len(trials),
            "votes": sum(1 for t in trials if t.get("vote")),
            "pairs": len(pairs),
            "ties": sum(1 for t in trials
                        if (t.get("vote") or {}).get("choice") == "tie"),
            "both_bad": sum(1 for t in trials
                            if (t.get("vote") or {}).get("choice") == "bad"),
            "rankings": sum(1 for t in trials if (t.get("vote") or {}).get("ranking")),
            "scored": sum(1 for t in trials for s in t.get("samples", {}).values()
                          if s.get("cer") is not None),
        },
        "asr_baseline": baseline,
        "caveats": _caveats(rows, pairs, baseline),
        "elo_config": {"k": config.ELO_K, "start": config.ELO_START,
                       "bootstrap": config.ELO_BOOTSTRAP},
    }


# ---------------------------------------------------------------- 匯出
CSV_COLUMNS = [
    "trial_id", "created_at", "category", "preset_id", "chars", "mode", "repeats",
    "engine", "engine_label", "voice_label", "uses_custom_reference",
    "elapsed_seconds", "rtf", "realtime_x", "first_chunk_seconds", "duration_seconds",
    "elapsed_stdev", "cer", "asr_text", "sample_rate", "effective_bandwidth_hz",
    "silence_ratio", "lead_silence_seconds", "clip_ratio", "rms_dbfs",
    "vote_choice", "vote_winner", "error", "text",
]


def export_rows(limit: Optional[int] = None) -> list[dict]:
    """一列一個 (評測 × 引擎) 的攤平資料，給 CSV 匯出與外部分析用。"""
    out: list[dict] = []
    for trial in bench.list_trials(limit=limit or config.TRIALS_LIMIT):
        vote = trial.get("vote") or {}
        for key in trial.get("engines", []):
            sample = trial.get("samples", {}).get(key)
            if not sample:
                continue
            m = sample.get("audio_metrics") or {}
            spread = sample.get("elapsed_spread") or {}
            out.append({
                "trial_id": trial["id"],
                "created_at": trial["created_at"],
                "category": trial.get("category", ""),
                "preset_id": trial.get("preset_id", ""),
                "chars": trial.get("chars"),
                "mode": trial.get("mode", ""),
                "repeats": trial.get("repeats", 1),
                "engine": key,
                "engine_label": sample.get("engine_label", key),
                "voice_label": sample.get("voice_label", ""),
                "uses_custom_reference": sample.get("uses_custom_reference", False),
                "elapsed_seconds": sample.get("elapsed_seconds"),
                "rtf": sample.get("rtf"),
                "realtime_x": sample.get("realtime_x"),
                "first_chunk_seconds": sample.get("first_chunk_seconds"),
                "duration_seconds": sample.get("duration_seconds"),
                "elapsed_stdev": spread.get("stdev"),
                "cer": sample.get("cer"),
                "asr_text": sample.get("asr_text") or "",
                "sample_rate": sample.get("sample_rate"),
                "effective_bandwidth_hz": m.get("effective_bandwidth_hz"),
                "silence_ratio": m.get("silence_ratio"),
                "lead_silence_seconds": m.get("lead_silence_seconds"),
                "clip_ratio": m.get("clip_ratio"),
                "rms_dbfs": m.get("rms_dbfs"),
                "vote_choice": vote.get("choice", ""),
                "vote_winner": vote.get("winner") or "",
                "error": sample.get("error") or "",
                "text": trial.get("text", ""),
            })
    return out


def export_csv(limit: Optional[int] = None) -> str:
    import csv
    import io

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=CSV_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    for row in export_rows(limit):
        writer.writerow(row)
    return buf.getvalue()
