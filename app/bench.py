"""TTS 評測核心：同一段文字送進多個引擎，比速度、吐字、音色。

引擎本身是可插拔的（見 app/engines/），這裡完全不知道有哪些引擎存在，
只透過登錄表拿轉接器。加新引擎不用改這個檔案。

三種評測模式
------------
`open`      不盲測，直接列出全部結果。想快速看數據時用。
`duel`      跑完全部選定的引擎，但只**盲測其中兩個**讓人投票。
`ranking`   跑完全部，全部盲測，讓人從最好排到最差。

`duel` 為什麼要「跑全部、只比兩個」
----------------------------------
速度與 CER 是客觀的，多跑一顆引擎就多一筆客觀資料，反正都已經在等了。
但主觀投票是二選一才準：一次聽四段再排序很累，而且容易被順序影響。
所以客觀資料吃滿、主觀負擔壓到最低。要省時間就把 `pair_only` 打開。

對決要配哪兩個
--------------
不是純隨機 —— 純隨機在四個模型時會讓某些組合遲遲配不到，對戰矩陣一直有空格。
改成**優先配「目前交手次數最少」的那一對**，讓矩陣均勻長滿。

重複跑的順序
------------
`repeats > 1` 時採**輪替**（第一輪 A,B,C,D、第二輪重新洗牌再跑一遍），
而不是「A 跑三次再換 B」。GPU 會隨著時間熱起來、記憶體會碎片化，
連續跑同一顆會把這些漂移全部算到它頭上。

盲測協定
--------
投票前，API 只回格子與音檔網址，其他一律不給：引擎名稱藏起來是基本，
連秒數也要藏 —— 引擎之間速度常常差好幾倍，看到「1.6 秒 vs 11.4 秒」
就等於直接知道哪個是哪個了。投完票（或按「直接揭曉」）才把完整數據放出來。
"""

from __future__ import annotations

import itertools
import json
import logging
import random
import threading
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from . import asr as asr_mod
from . import config, engines, stats, voice_sync
from .audio_utils import save_wav
from .text_utils import visible_length

log = logging.getLogger("tts.bench")

Progress = Callable[[str, float], None]

TRIALS_FILE = config.STORAGE_DIR / "trials.json"
_lock = threading.Lock()

SLOTS = ("A", "B", "C", "D", "E", "F")
MODES = ("open", "duel", "ranking")
# 對決模式的投票選項。bad = 兩個都不行（不產生排名資訊，見 stats.pairs_from_vote）
DUEL_CHOICES = {"tie", "bad"} | set(SLOTS)


# ---------------------------------------------------------------- 測試句庫
# 挑的原則：每一類都是中文 TTS 的已知痛點，能把模型的差距拉開。
# `category` 會被記進每一筆評測，排行榜據此拆出「誰在哪類文本強」。
PRESETS = [
    {"id": "short", "category": "short", "label": "短句",
     "text": "今天天氣真好，適合出門走走。"},
    {"id": "news", "category": "news", "label": "氣象播報",
     "text": "苗栗以北地區上空幾乎沒有什麼雲，今天清晨輻射冷卻效應會比較明顯，"
             "低溫幾乎都出現在北台灣。"},
    {"id": "numbers", "category": "numbers", "label": "數字與單位",
     "text": "今日加權指數收在兩萬三千四百五十六點，上漲百分之一點二七，"
             "成交量新台幣三千八百億元。"},
    {"id": "digits", "category": "numbers", "label": "阿拉伯數字",
     "text": "訂單編號 A2024-8871，總金額 NT$45,300，預計 3 到 5 個工作天送達，"
             "客服電話 0800-092-000。"},
    {"id": "mixed", "category": "mixed", "label": "中英夾雜",
     "text": "這次 update 把 API 的 latency 從 300 毫秒降到 80 毫秒，"
             "團隊在 Slack 上開了一個 channel 追蹤。"},
    {"id": "polyphone", "category": "polyphone", "label": "破音字",
     "text": "他長期在銀行行員的位置上，重複處理重要的帳務，還得空出時間學樂理。"},
    {"id": "polyphone2", "category": "polyphone", "label": "破音字（進階）",
     "text": "這批貨還沒還清，得先重新分配；那位曾參與研究的曾先生說，"
             "血液裡的血塊要盡快處理。"},
    {"id": "punctuation", "category": "prosody", "label": "語氣與停頓",
     "text": "等一下——你確定嗎？我是說，真的、真的確定？「好吧，」他嘆了口氣，"
             "「那就這樣吧。」"},
    {"id": "rare", "category": "rare", "label": "罕用字與人名",
     "text": "龔琳娜與馬爾濟斯在饕餮盛宴上相遇，鬱鬱蔥蔥的蒹葭旁，"
             "有人低聲吟誦著詰屈聱牙的辭賦。"},
    {"id": "long", "category": "long", "label": "長段落",
     "text": "語音合成技術這幾年進步得非常快，從早期需要大量錄音的拼接式合成，"
             "到後來的參數式合成，再到現在只要幾秒鐘的參考音就能複製音色的"
             "零樣本模型。評估這些系統的時候，除了聽起來像不像本人，"
             "更要看它有沒有把字唸對、速度能不能跟上實際應用的需求。"},
]

CATEGORY_LABELS = {
    "short": "短句", "news": "播報", "numbers": "數字", "mixed": "中英夾雜",
    "polyphone": "破音字", "prosody": "語氣停頓", "rare": "罕用字", "long": "長段落",
    "custom": "自訂",
}


def preset_categories() -> list[dict]:
    seen: dict[str, int] = {}
    for p in PRESETS:
        seen[p["category"]] = seen.get(p["category"], 0) + 1
    return [{"id": c, "label": CATEGORY_LABELS.get(c, c), "count": n}
            for c, n in seen.items()]


# ---------------------------------------------------------------- 小工具
def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _new_id() -> str:
    return datetime.now().strftime("%y%m%d%H%M%S") + "-" + format(random.getrandbits(16), "04x")


# ---------------------------------------------------------------- 儲存
def _load() -> list[dict]:
    if not TRIALS_FILE.exists():
        return []
    try:
        data = json.loads(TRIALS_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:  # noqa: BLE001
        log.warning("trials.json 讀取失敗，當成空的處理")
        return []


def _drop_audio(trial: dict) -> None:
    for sample in trial.get("samples", {}).values():
        name = sample.get("filename")
        if name:
            (config.OUTPUTS_DIR / name).unlink(missing_ok=True)


def _save(items: list[dict]) -> None:
    keep = items[: config.TRIALS_LIMIT]
    for trial in items[config.TRIALS_LIMIT :]:
        _drop_audio(trial)
    TRIALS_FILE.write_text(json.dumps(keep, ensure_ascii=False, indent=2), encoding="utf-8")


def list_trials(limit: int = 100) -> list[dict]:
    with _lock:
        return _load()[:limit]


def get_trial(trial_id: str) -> Optional[dict]:
    with _lock:
        for t in _load():
            if t["id"] == trial_id:
                return t
    return None


def _upsert(trial: dict) -> None:
    with _lock:
        items = [t for t in _load() if t["id"] != trial["id"]]
        items.insert(0, trial)
        _save(items)


def delete_trial(trial_id: str) -> bool:
    with _lock:
        items = _load()
        target = next((t for t in items if t["id"] == trial_id), None)
        if not target:
            return False
        _drop_audio(target)
        _save([t for t in items if t["id"] != trial_id])
    return True


def clear_trials() -> int:
    with _lock:
        items = _load()
        for trial in items:
            _drop_audio(trial)
        _save([])
    return len(items)


# ---------------------------------------------------------------- 暖機
def warmup(engine_keys: Optional[list[str]] = None,
           progress: Optional[Progress] = None) -> dict:
    """把每顆引擎的模型載進 GPU。

    四包 gateway 有專屬的 POST /v1/warmup，冷啟動載模型要 30–90 秒。
    不先做這件事，第一筆評測量到的是「載模型 + 合成」，跟其他引擎完全不能比。
    """
    adapters = [a for a in engines.enabled_engines()
                if engine_keys is None or a.key in engine_keys]
    out: dict[str, Any] = {}
    for i, adapter in enumerate(adapters):
        if progress:
            progress(f"暖機 {adapter.label}…（載模型可能要 30–90 秒）",
                     i / max(len(adapters), 1))
        try:
            out[adapter.key] = adapter.warmup()
        except Exception as exc:  # noqa: BLE001
            out[adapter.key] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    if progress:
        progress("暖機完成", 1.0)
    return out


def voice_opts(voice: dict) -> dict:
    """相容用的轉呼叫；實作在 voice_sync（那裡才知道各引擎的遠端 voice id）。"""
    return voice_sync.voice_opts(voice)


# ---------------------------------------------------------------- 可比性
def comparability(engine_keys: list[str], opts: dict) -> dict:
    """這一組引擎目前哪些維度可以公平比。

    音色是分**組**的，不是單一布林值：qwen3-tts 只有內建 speaker，天生沒辦法
    跟克隆出來的音色比像不像；但另外三顆只要都推了同一段參考音就可以互比。
    把「不可比」一路傳染到全體，會讓三顆本來可比的引擎也被白白放棄。
    """
    custom_ref: list[str] = []
    own_voice: list[str] = []
    for key in engine_keys:
        adapter = engines.get(key)
        if adapter is None:
            continue
        (custom_ref if adapter.uses_custom_reference(opts) else own_voice).append(key)

    if len(custom_ref) >= 2:
        note = (f"{'、'.join(engines.label_for(k) for k in custom_ref)} 都用同一段參考音，"
                "這幾顆之間的音色相似度可以比。")
        if own_voice:
            note += (f"{'、'.join(engines.label_for(k) for k in own_voice)} 用的是自己的音色，"
                     "不列入音色比較。")
    elif custom_ref:
        note = (f"只有 {engines.label_for(custom_ref[0])} 用了指定參考音，"
                "沒有對手可以比音色。")
    else:
        note = "沒有引擎使用指定的參考音，音色相似度不在比較範圍內。"

    return {
        "speed": True,
        "intelligibility": bool(opts.get("score")) and config.ASR_ENABLED,
        "timbre": len(custom_ref) >= 2,
        "timbre_group": custom_ref,
        "own_voice_engines": own_voice,
        "note": note,
    }


# ---------------------------------------------------------------- 對決配對
def pick_duel_pair(candidates: list[str],
                   history: Optional[list[dict]] = None) -> list[str]:
    """從成功的引擎裡挑兩個對決：優先挑交手次數最少的那一對。

    純隨機在四個模型（6 種組合）時會讓某些組合遲遲配不到，對戰矩陣一直有空格，
    Elo 也會因為對手覆蓋不均而失真。這裡改成先補最缺的那一格。
    """
    if len(candidates) <= 2:
        return list(candidates)

    history = history if history is not None else list_trials(limit=config.TRIALS_LIMIT)
    played: dict[frozenset, int] = {}
    for pair in stats.all_pairs(history):
        key = frozenset((pair["winner"], pair["loser"]))
        played[key] = played.get(key, 0) + 1

    combos = list(itertools.combinations(sorted(candidates), 2))
    random.shuffle(combos)          # 同分時隨機打破平手，避免固定偏好某一對
    combos.sort(key=lambda c: played.get(frozenset(c), 0))
    return list(combos[0])


# ---------------------------------------------------------------- 跑一次評測
def _blank_sample(engine_key: str, label: str, voice_label: str, error: str) -> dict:
    return {"engine": engine_key, "engine_label": label, "filename": None,
            "voice_label": voice_label, "error": error, "runs": []}


def _aggregate_runs(runs: list[dict]) -> dict:
    """把重複跑的多次結果收斂成一組統計。單次跑時就等於那一次的值。"""
    elapsed = [r["elapsed_seconds"] for r in runs]
    return {
        "elapsed_seconds": stats.median(elapsed),
        "elapsed_spread": stats.spread(elapsed),
        "rtf": stats.median([r["rtf"] for r in runs if r.get("rtf")]),
        "realtime_x": stats.median([r["realtime_x"] for r in runs if r.get("realtime_x")]),
        "first_chunk_seconds": stats.median(
            [r["first_chunk_seconds"] for r in runs if r.get("first_chunk_seconds")]),
        "duration_seconds": stats.median([r["duration_seconds"] for r in runs]),
    }


def run_trial(text: str, engine_keys: list[str], opts: dict, progress: Progress) -> dict:
    """跑一次完整評測，回傳存好的 trial（完整版，呼叫端自己決定要不要遮蔽）。"""
    trial_id = _new_id()
    mode = opts.get("mode", "duel")
    repeats = max(1, min(int(opts.get("repeats", 1)), config.MAX_REPEATS))
    chars = visible_length(text)

    samples: dict[str, dict] = {}
    errors: dict[str, str] = {}
    run_log: list[list[str]] = []

    total_steps = len(engine_keys) * repeats + (len(engine_keys) if opts.get("score") else 0)
    step = 0

    def bump(msg: str, frac: float = 1.0) -> None:
        progress(msg, min((step + frac) / max(total_steps, 1), 0.99))

    # --- 合成：輪替跑，一次一顆，避免互搶 GPU 讓速度數字失真 ---
    for rep in range(repeats):
        order = list(engine_keys)
        if config.RANDOMIZE_RUN_ORDER:
            random.shuffle(order)
        run_log.append(order)

        for engine_key in order:
            label = engines.label_for(engine_key)
            prefix = f"{label}" + (f"（第 {rep + 1}/{repeats} 次）" if repeats > 1 else "")
            try:
                def scoped(msg: str, frac: float = 0.0, _p=prefix) -> None:
                    bump(f"{_p}：{msg}", min(max(frac, 0.0), 1.0))

                scoped("送出合成請求…", 0.0)
                adapter = engines.require(engine_key)
                audio, sr, info = adapter.synthesize(text, opts, scoped)

                duration = info.get("duration_seconds") or 0.0
                elapsed = info["elapsed_seconds"]
                run = {
                    "run": rep + 1,
                    "elapsed_seconds": elapsed,
                    "duration_seconds": duration,
                    # RTF < 1 = 比即時快；realtime_x 是「1 秒算力產出幾秒語音」
                    "rtf": round(elapsed / duration, 3) if duration else None,
                    "realtime_x": round(duration / elapsed, 2) if elapsed else None,
                    "first_chunk_seconds": info.get("first_chunk_seconds"),
                    "chars_per_second": round(chars / elapsed, 2) if elapsed else None,
                }

                sample = samples.get(engine_key)
                if sample is None or sample.get("error"):
                    sample = {
                        "engine": engine_key,
                        "engine_label": label,
                        "filename": None,
                        "voice_label": adapter.voice_label(opts),
                        "uses_custom_reference": adapter.uses_custom_reference(opts),
                        "sample_rate": sr,
                        "segments": info.get("segments"),
                        "requests": info.get("requests"),
                        "endpoint": info.get("endpoint"),
                        "runs": [],
                        "asr_text": None, "cer": None,
                        "score_error": None, "error": None,
                    }
                    samples[engine_key] = sample

                sample["runs"].append(run)

                # 只留第一次的音檔給人聽；重複跑是為了看速度穩定性，不是要存 N 個檔案
                if not sample.get("filename"):
                    filename = f"{trial_id}_{engine_key}.wav"
                    save_wav(config.OUTPUTS_DIR / filename, audio, sr)
                    sample["filename"] = filename
                    sample["audio_metrics"] = info.get("audio_metrics")
                    sample["audio_warnings"] = info.get("audio_warnings") or []

            except Exception as exc:  # noqa: BLE001
                log.exception("%s 合成失敗", label)
                message = f"{type(exc).__name__}: {exc}"
                errors[engine_key] = message
                if engine_key not in samples or not samples[engine_key].get("runs"):
                    samples[engine_key] = _blank_sample(
                        engine_key, label, "—", message)
                else:
                    samples[engine_key]["partial_error"] = message
            step += 1

    for sample in samples.values():
        if sample.get("runs"):
            sample.update(_aggregate_runs(sample["runs"]))

    # --- 客觀評分：全部合成完才做，免得 ASR 佔用資源影響速度量測 ---
    if opts.get("score"):
        for engine_key in engine_keys:
            sample = samples.get(engine_key, {})
            if not sample.get("filename"):
                step += 1
                continue
            bump(f"{sample['engine_label']}：辨識吐字…", 0.0)
            result = asr_mod.score(text, config.OUTPUTS_DIR / sample["filename"])
            sample["asr_text"] = result.get("asr_text")
            sample["cer"] = result.get("cer")
            sample["asr_seconds"] = result.get("asr_seconds")
            sample["score_error"] = result.get("error")
            step += 1

    # --- 盲測配對 ---
    ok_engines = [k for k in engine_keys if samples.get(k, {}).get("filename")]
    blind_pool: list[str] = []
    if mode == "duel" and len(ok_engines) >= 2:
        blind_pool = pick_duel_pair(ok_engines)
    elif mode == "ranking" and len(ok_engines) >= 2:
        blind_pool = list(ok_engines)[: len(SLOTS)]

    shuffled = list(blind_pool)
    random.shuffle(shuffled)
    slot_map = {SLOTS[i]: key for i, key in enumerate(shuffled)}
    blind = bool(slot_map) and mode in ("duel", "ranking")

    trial = {
        "id": trial_id,
        "created_at": _now(),
        "text": text,
        "chars": chars,
        "category": opts.get("category") or "custom",
        "preset_id": opts.get("preset_id") or "",
        "engines": engine_keys,
        "run_order": run_log,
        "mode": mode,
        "repeats": repeats,
        "speed": opts["speed"],
        "voice_id": opts.get("voice_id", ""),
        "voice_name": opts.get("voice_name", ""),
        "scored": bool(opts.get("score")),
        "comparability": comparability(engine_keys, opts),
        "slots": slot_map,
        "samples": samples,
        "errors": errors,
        "blind": blind,
        "revealed": not blind,
        "vote": None,
        "batch_id": opts.get("batch_id") or "",
    }
    _upsert(trial)
    progress("完成", 1.0)
    return trial


# ---------------------------------------------------------------- 遮蔽
def to_public(trial: dict) -> dict:
    """要送給前端的版本。盲測未揭曉時，把所有能認出引擎的線索都拿掉。"""
    base = {
        "id": trial["id"],
        "created_at": trial["created_at"],
        "text": trial["text"],
        "chars": trial["chars"],
        "category": trial.get("category", "custom"),
        "preset_id": trial.get("preset_id", ""),
        "mode": trial.get("mode", "duel"),
        "repeats": trial.get("repeats", 1),
        "speed": trial["speed"],
        "voice_name": trial.get("voice_name", ""),
        "scored": trial.get("scored", False),
        "blind": trial.get("blind", False),
        "revealed": trial.get("revealed", False),
        "vote": trial.get("vote"),
        "batch_id": trial.get("batch_id", ""),
    }

    if not trial.get("revealed"):
        # 只給格子與音檔網址。除了引擎名稱，這幾樣也都要藏：
        #   * 秒數 —— 引擎之間速度常差好幾倍，看到就等於知道答案
        #   * 參賽名單 —— 知道候選是誰就會帶著預期去聽，盲測就不純了
        #   * comparability 的 note 與 timbre_group —— 裡面直接寫著引擎名稱
        base["slots"] = [
            {"slot": slot, "url": f"/api/trials/{trial['id']}/audio/{slot}"}
            for slot in sorted(trial.get("slots", {}))
        ]
        base["samples"] = None
        base["errors"] = {}
        base["engines"] = None
        cmp_ = trial.get("comparability", {})
        base["comparability"] = {k: cmp_.get(k) for k in ("speed", "intelligibility", "timbre")}
        # 有幾顆引擎跑了但沒進盲測（duel 模式會這樣），只講數量不講是誰
        base["hidden_count"] = max(0, len(trial["engines"]) - len(trial.get("slots", {})))
        return base

    base["engines"] = trial["engines"]
    base["comparability"] = trial.get("comparability", {})
    base["slots"] = [
        {
            "slot": slot,
            "engine": key,
            "engine_label": engines.label_for(key),
            "url": f"/api/trials/{trial['id']}/audio/{slot}",
        }
        for slot, key in sorted(trial.get("slots", {}).items())
    ]
    base["samples"] = [
        {**s, "url": f"/api/outputs/{s['filename']}" if s.get("filename") else None}
        for s in (trial["samples"].get(k) for k in trial["engines"])
        if s
    ]
    base["errors"] = trial.get("errors", {})
    base["run_order"] = trial.get("run_order")
    return base


# ---------------------------------------------------------------- 投票
def vote(trial_id: str, choice: str = "", ranking: Optional[list[str]] = None,
         note: str = "") -> dict:
    """記錄一次主觀投票並揭曉。

    投的是「格子」不是引擎名稱：投票的人到最後一刻都不知道自己選了誰。
    `ranking` 是格子的排序（最好排前面），對決模式用 `choice`。
    """
    trial = get_trial(trial_id)
    if not trial:
        raise KeyError(trial_id)
    if trial.get("vote"):
        raise ValueError("這筆評測已經投過票了")

    slot_map = trial.get("slots", {})
    if not slot_map:
        raise ValueError("這筆評測沒有盲測格子，不能投票")

    record: dict[str, Any] = {"note": (note or "").strip()[:500], "voted_at": _now()}

    if ranking:
        order = [str(s).strip().upper() for s in ranking]
        if sorted(order) != sorted(slot_map.keys()):
            raise ValueError(
                f"排名要把每一格都排進去且不能重複，這筆有 {sorted(slot_map)} 這幾格")
        record["ranking"] = order
        record["choice"] = "ranking"
        record["winner"] = slot_map[order[0]]
        record["loser"] = slot_map[order[-1]]
        record["ranked_engines"] = [slot_map[s] for s in order]
    else:
        choice = (choice or "").strip()
        if choice not in DUEL_CHOICES:
            raise ValueError(f"choice 只接受 {sorted(DUEL_CHOICES)}，或改用 ranking")
        record["choice"] = choice
        if choice in SLOTS:
            winner = slot_map.get(choice)
            if not winner:
                raise ValueError(f"這筆評測沒有 {choice} 這一格")
            record["winner"] = winner
            record["loser"] = next((k for s, k in slot_map.items() if s != choice), None)
        else:
            record["winner"] = None
            record["loser"] = None

    trial["vote"] = record
    trial["revealed"] = True
    _upsert(trial)
    return trial


def reveal(trial_id: str) -> dict:
    """不投票、直接看答案。這種筆數不會算進排名。"""
    trial = get_trial(trial_id)
    if not trial:
        raise KeyError(trial_id)
    trial["revealed"] = True
    _upsert(trial)
    return trial
