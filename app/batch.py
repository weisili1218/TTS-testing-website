"""批次跑測：一鍵把整個測試句庫 × 全部引擎跑完，產出報表。

為什麼要有這個
--------------
一句一句手動送，四個模型跑十句就要點四十次，而且中途很容易改到設定（換音色、
改語速），最後拿到的是一堆條件不一致的資料。批次跑測把「同一組設定、同一批
文字、同一批引擎」綁成一次任務，跑完存成一份不可變的報表。

每一句仍然是一筆正常的 trial（存進 trials.json、算進排行榜），只是多帶一個
`batch_id`。所以批次跑完的資料跟手動評測是同一種東西，不是另一套平行世界。

模式
----
`open`（預設）  只收客觀數據，不盲測。跑最快，適合換模型後的第一輪普查。
`duel`/`ranking` 每一句都留成待投票的盲測。跑完到競技場慢慢聽、慢慢投，
                主觀資料就會一起累積起來。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from . import bench, config, engines, stats

log = logging.getLogger("tts.batch")

Progress = Callable[[str, float], None]


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _batch_id() -> str:
    return "batch-" + datetime.now().strftime("%y%m%d-%H%M%S")


# ---------------------------------------------------------------- 題目挑選
def resolve_items(preset_ids: Optional[list[str]] = None,
                  categories: Optional[list[str]] = None,
                  custom_texts: Optional[list[str]] = None) -> list[dict]:
    """決定這次批次要跑哪些句子。三種來源可以混用，重複的會去掉。"""
    items: list[dict] = []
    seen: set[str] = set()

    def _add(item: dict) -> None:
        text = (item.get("text") or "").strip()
        if not text or text in seen:
            return
        seen.add(text)
        items.append(item)

    if preset_ids:
        wanted = set(preset_ids)
        for p in bench.PRESETS:
            if p["id"] in wanted:
                _add(dict(p))
    if categories:
        wanted = set(categories)
        for p in bench.PRESETS:
            if p["category"] in wanted:
                _add(dict(p))
    for text in custom_texts or []:
        _add({"id": "", "category": "custom", "label": "自訂", "text": str(text)})

    # 什麼都沒指定 = 跑完整題庫
    if not items:
        items = [dict(p) for p in bench.PRESETS]
    return items


# ---------------------------------------------------------------- 儲存
def _path(batch_id: str) -> "Any":
    from pathlib import Path

    return config.REPORTS_DIR / f"{Path(batch_id).name}.json"


def save_report(report: dict) -> None:
    _path(report["id"]).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def get_report(batch_id: str) -> Optional[dict]:
    path = _path(batch_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        log.warning("報表 %s 解析失敗", batch_id)
        return None


def list_reports(limit: int = 50) -> list[dict]:
    out = []
    for path in sorted(config.REPORTS_DIR.glob("batch-*.json"), reverse=True)[:limit]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        out.append({k: data.get(k) for k in
                    ("id", "created_at", "finished_at", "mode", "repeats",
                     "engines", "voice_name", "item_count", "done", "failed",
                     "summary")})
    return out


def delete_report(batch_id: str) -> bool:
    path = _path(batch_id)
    if not path.exists():
        return False
    path.unlink()
    return True


# ---------------------------------------------------------------- 執行
def _summarize(trial_ids: list[str], engine_keys: list[str]) -> dict:
    """跑完之後就地聚合這一批（不含其他歷史紀錄）的結果。"""
    wanted = set(trial_ids)
    trials = [t for t in bench.list_trials(limit=config.TRIALS_LIMIT) if t["id"] in wanted]

    per_engine: dict[str, dict] = {}
    for key in engine_keys:
        elapsed, rtf, cer, first, runs = [], [], [], [], []
        ok = failed = 0
        errors: dict[str, int] = {}
        for trial in trials:
            sample = trial.get("samples", {}).get(key)
            if not sample:
                continue
            if sample.get("error") or not sample.get("filename"):
                failed += 1
                errors[str(sample.get("error") or "未知")[:100]] = \
                    errors.get(str(sample.get("error") or "未知")[:100], 0) + 1
                continue
            ok += 1
            elapsed.append(sample.get("elapsed_seconds"))
            rtf.append(sample.get("rtf"))
            first.append(sample.get("first_chunk_seconds"))
            if sample.get("cer") is not None:
                cer.append(sample["cer"])
            for run in sample.get("runs") or []:
                runs.append(run.get("elapsed_seconds"))

        per_engine[key] = {
            "engine": key,
            "engine_label": engines.label_for(key),
            "ok": ok, "failed": failed,
            "success_rate": round(ok / (ok + failed), 3) if (ok + failed) else None,
            "elapsed": stats.spread(elapsed),
            "rtf": stats.spread(rtf),
            "first_chunk": stats.median(first),
            "cer": stats.median(cer),
            "cer_spread": stats.spread(cer),
            "stability_cv": stats.cv(runs),
            "top_errors": sorted(errors.items(), key=lambda kv: -kv[1])[:3],
        }

    scored = {k: v["cer"] for k, v in per_engine.items() if v["cer"] is not None}
    fast = {k: v["elapsed"]["median"] for k, v in per_engine.items()
            if v["elapsed"]["median"] is not None}
    steady = {k: v["stability_cv"] for k, v in per_engine.items()
              if v["stability_cv"] is not None}

    return {
        "per_engine": per_engine,
        "best_cer": min(scored, key=scored.get) if scored else None,
        "fastest": min(fast, key=fast.get) if fast else None,
        "most_stable": min(steady, key=steady.get) if steady else None,
        "categories": _category_breakdown(trials, engine_keys),
    }


def _category_breakdown(trials: list[dict], engine_keys: list[str]) -> dict:
    out: dict[str, dict] = {}
    for trial in trials:
        cat = trial.get("category") or "custom"
        row = out.setdefault(cat, {"label": bench.CATEGORY_LABELS.get(cat, cat),
                                   "engines": {}})
        for key in engine_keys:
            sample = trial.get("samples", {}).get(key)
            if not sample or not sample.get("filename"):
                continue
            cell = row["engines"].setdefault(key, {"cer": [], "elapsed": []})
            if sample.get("cer") is not None:
                cell["cer"].append(sample["cer"])
            cell["elapsed"].append(sample.get("elapsed_seconds"))
    for row in out.values():
        for key, cell in row["engines"].items():
            row["engines"][key] = {
                "cer": stats.median(cell["cer"]),
                "elapsed": stats.median(cell["elapsed"]),
                "n": len(cell["elapsed"]),
            }
    return out


def run_batch(items: list[dict], engine_keys: list[str], opts: dict,
              progress: Progress) -> dict:
    """把整批句子逐一跑完。每一句都是一筆正常的 trial。

    整批**依序**跑（不是同時），理由跟單筆評測一樣：引擎共用同一台 GPU，
    並行測出來的秒數不能比。這也代表批次會跑很久 —— 十句 × 四個引擎，
    每句兩秒的話大約一分半，長段落會更久。
    """
    batch_id = _batch_id()
    started = _now()
    trial_ids: list[str] = []
    failures: list[dict] = []
    total = max(len(items), 1)

    report = {
        "id": batch_id,
        "created_at": started,
        "finished_at": None,
        "mode": opts.get("mode", "open"),
        "repeats": opts.get("repeats", 1),
        "speed": opts.get("speed", 1.0),
        "scored": bool(opts.get("score")),
        "engines": engine_keys,
        "engine_labels": {k: engines.label_for(k) for k in engine_keys},
        "voice_id": opts.get("voice_id", ""),
        "voice_name": opts.get("voice_name", ""),
        "item_count": len(items),
        "items": [{"id": i.get("id"), "label": i.get("label"),
                   "category": i.get("category"), "chars": len(i.get("text", ""))}
                  for i in items],
        "trial_ids": [],
        "done": 0,
        "failed": 0,
        "failures": [],
        "summary": None,
    }
    save_report(report)

    for index, item in enumerate(items):
        label = item.get("label") or f"第 {index + 1} 句"
        base = index / total

        def scoped(msg: str, frac: float = 0.0, _b=base, _l=label) -> None:
            progress(f"[{index + 1}/{total}] {_l}：{msg}",
                     min(_b + (max(0.0, min(frac, 1.0)) / total), 0.99))

        trial_opts = dict(opts)
        trial_opts.update({
            "category": item.get("category") or "custom",
            "preset_id": item.get("id") or "",
            "batch_id": batch_id,
        })
        try:
            trial = bench.run_trial(item["text"], engine_keys, trial_opts, scoped)
            trial_ids.append(trial["id"])
            report["done"] += 1
        except Exception as exc:  # noqa: BLE001
            log.exception("批次第 %d 句失敗", index + 1)
            failures.append({"index": index, "label": label,
                             "error": f"{type(exc).__name__}: {exc}"})
            report["failed"] += 1

        report["trial_ids"] = trial_ids
        report["failures"] = failures
        save_report(report)      # 逐句存檔：中途中斷也留得住已經跑完的部分

    progress("彙整報表…", 0.99)
    report["finished_at"] = _now()
    report["summary"] = _summarize(trial_ids, engine_keys)
    save_report(report)
    progress("批次完成", 1.0)
    return report
