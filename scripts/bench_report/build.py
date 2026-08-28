"""把一次（或多次）批次跑測整理成多引擎比較報告的資料檔。

這支腳本做什麼
--------------
平台跑完批次之後，`storage/reports/<batch>.json` 只有那一批的摘要，
逐句的原始數據散在 `storage/trials.json` 裡。要做跨引擎比較還缺三件事：

    1. 把不同批次的同一句對起來（引擎不一定同一批跑的）
    2. 補上平台沒算的寬鬆 CER
    3. 兩兩跑顯著性檢定，而不是只列中位數

輸出是一份 JSON，給報告頁（或任何畫圖的東西）直接吃。

為什麼要能跨批次
----------------
理想上所有引擎在同一次批次裡逐句輪替跑，速度才嚴格可比。但實務上常常是後來才
加一顆進來 —— 這時候只能拿舊批次的結果對。腳本允許這樣做，但會把每顆引擎來自
哪一批**記進 meta.runs**，報告要據此標示「這兩顆不是同場跑的」。
跨場次的速度比較混著系統性誤差，不能當成同場結果讀。

題目怎麼對齊
------------
用**文字內容**對，不是用順序或編號。順序在不同批次可能不同，編號則是批次自己的
流水號。文字對不上就直接中止 —— 題庫不一樣的兩批本來就不該放在同一份報告裡。

用法
----
    python scripts/bench_report/build.py \
        --engine fun-cosyvoice3=batch-260828-172718 \
        --engine qwen3-tts=batch-260826-150024 \
        --engine f5=batch-260826-150024 \
        --out storage/reports/compare.json

`--engine` 的順序就是報告裡的呈現順序。同一顆引擎只能指定一次。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app import stats                                    # noqa: E402
from scripts.bench_report.lenient_cer import lenient_cer  # noqa: E402
from scripts.bench_report.significance import (           # noqa: E402
    diff_ci, sign_test, wilcoxon)

CATEGORY_ORDER = [("short", "短句"), ("news", "播報"), ("numbers", "數字"),
                  ("mixed", "中英夾雜"), ("polyphone", "破音字"),
                  ("prosody", "語氣停頓"), ("rare", "罕用字"), ("long", "長段落")]


# ---------------------------------------------------------------- 讀檔
def load_batch(storage: Path, batch_id: str) -> tuple[dict, list[dict]]:
    report = json.loads((storage / "reports" / f"{batch_id}.json").read_text(encoding="utf-8"))
    trials = {t["id"]: t for t in
              json.loads((storage / "trials.json").read_text(encoding="utf-8"))}
    missing = [i for i in report["trial_ids"] if i not in trials]
    if missing:
        raise SystemExit(
            f"批次 {batch_id} 有 {len(missing)} 筆 trial 已經被 TRIALS_LIMIT 擠掉了"
            f"（例如 {missing[0]}）—— 逐句數據補不回來，這批不能拿來做報告。")
    return report, [trials[i] for i in report["trial_ids"]]


# ---------------------------------------------------------------- 逐句
def cell(trial: dict, engine_key: str) -> dict | None:
    """一句 × 一顆引擎的所有指標。合成失敗回 None。"""
    sample = (trial.get("samples") or {}).get(engine_key)
    if not sample or not sample.get("filename"):
        return None
    m = sample.get("audio_metrics") or {}
    asr = sample.get("asr_text") or ""
    return {
        "e": sample.get("elapsed_seconds"), "d": sample.get("duration_seconds"),
        "rtf": sample.get("rtf"), "rx": sample.get("realtime_x"),
        "c": sample.get("cer"), "cl": lenient_cer(trial["text"], asr), "asr": asr,
        "cps": m.get("chars_per_voiced_second"),
        "bw": m.get("effective_bandwidth_hz"), "bwr": m.get("bandwidth_ratio"),
        "sil": m.get("silence_ratio"), "lead": m.get("lead_silence_seconds"),
        "tail": m.get("tail_silence_seconds"), "rms": m.get("rms_dbfs"),
        "sr": sample.get("sample_rate"),
    }


FIELDS = [("e", "elapsed"), ("rtf", "rtf"), ("rx", "rx"), ("c", "cer"), ("cl", "cerL"),
          ("d", "dur"), ("cps", "cps"), ("bw", "bw"), ("bwr", "bwr"),
          ("sil", "sil"), ("lead", "lead"), ("tail", "tail"), ("rms", "rms")]


def _spread(values: list) -> dict:
    """app.stats.spread 再補上四分位 —— 畫分布圖要靠 p25/p75 標出主體範圍。

    刻意沿用 app.stats 而不是自己算，這樣報告裡的數字跟平台 UI 上看到的一致。
    附帶的代價：`cv` 是用**四捨五入到小數第四位的標準差**除以平均算的，數值很小的
    欄位（例如結尾靜音 0.04 秒）相對誤差會被放大。要拿 cv 做結論就自己重算。
    """
    clean = [v for v in values if v is not None]
    return {**stats.spread(values),
            "p25": stats.percentile(clean, 0.25),
            "p75": stats.percentile(clean, 0.75)}


def engine_block(key: str, label: str, cells: list[dict | None]) -> dict:
    ok = [c for c in cells if c]
    col = lambda f: [c[f] for c in ok]                                # noqa: E731
    block = {"key": key, "label": label, "ok": len(ok), "n": len(cells)}
    for field, name in FIELDS:
        block[name] = _spread(col(field))
    block["wall"] = round(sum(x for x in col("e") if x is not None), 1)
    block["audio"] = round(sum(x for x in col("d") if x is not None), 1)
    rates = [c["sr"] for c in ok if c.get("sr")]
    block["sr"] = max(set(rates), key=rates.count) if rates else None
    return block


# ---------------------------------------------------------------- 分類與檢定
def category_block(rows: list[dict], key: str) -> dict:
    ok = [r[key] for r in rows if r.get(key)]
    if not ok:
        return {"n": 0}
    col = lambda f: [c[f] for c in ok if c.get(f) is not None]        # noqa: E731
    avg = lambda xs: round(sum(xs) / len(xs), 4) if xs else None      # noqa: E731
    med = lambda xs: round(stats.median(xs), 4) if xs else None       # noqa: E731
    return {"n": len(ok),
            "cer_median": med(col("c")), "cer_mean": avg(col("c")),
            "cer_lenient_median": med(col("cl")), "cer_lenient_mean": avg(col("cl")),
            "elapsed_median": med(col("e")), "rtf_median": med(col("rtf")),
            "cps_median": med(col("cps"))}


def pair_tests(rows: list[dict], a: str, b: str) -> dict:
    """a 對 b 的成對比較。差值一律定義為 a - b；CER 與耗時都是越小越好。"""
    paired = [(r[a], r[b]) for r in rows if r.get(a) and r.get(b)]
    out = {"a": a, "b": b, "n_paired": len(paired)}
    for field, name in (("c", "cer"), ("cl", "cerL"), ("e", "elapsed"), ("rtf", "rtf")):
        xa = [p[0][field] for p in paired]
        xb = [p[1][field] for p in paired]
        both = [(u, v) for u, v in zip(xa, xb) if u is not None and v is not None]
        block = {"wilcoxon": wilcoxon(xa, xb), "sign": sign_test(xa, xb),
                 "ci": diff_ci(xa, xb),
                 "a_better": sum(1 for u, v in both if u < v),
                 "b_better": sum(1 for u, v in both if u > v),
                 "equal": sum(1 for u, v in both if u == v)}
        if field == "e":
            ratios = [v / u for u, v in both if u]        # b 是 a 的幾倍
            block["ratio_median"] = round(stats.median(ratios), 3) if ratios else None
        out[name] = block
    return out


def vs_baseline(rows: list[dict], key: str, baseline: float) -> dict:
    v = [r[key]["c"] for r in rows if r.get(key) and r[key].get("c") is not None]
    return {"baseline": baseline, "n": len(v),
            "below_baseline": sum(1 for x in v if x < baseline),
            "at_or_above": sum(1 for x in v if x >= baseline),
            "under_005": sum(1 for x in v if x < 0.05),
            "under_010": sum(1 for x in v if x < 0.10),
            "over_030": sum(1 for x in v if x > 0.30),
            "over_050": sum(1 for x in v if x > 0.50)}


def strict_minus_lenient(rows: list[dict], key: str) -> dict:
    """嚴格與寬鬆的落差 —— 這一顆有多少 CER 其實只是書寫形式不同。"""
    d = [r[key]["c"] - r[key]["cl"] for r in rows
         if r.get(key) and r[key].get("c") is not None and r[key].get("cl") is not None]
    if not d:
        return {}
    return {"mean": round(sum(d) / len(d), 4), "median": round(stats.median(d), 4),
            "max": round(max(d), 4), "items_improved": sum(1 for x in d if x > 0)}


# ---------------------------------------------------------------- 組裝
def build(storage: Path, engines: list[tuple[str, str]], baseline: float) -> dict:
    """engines 是 [(engine_key, batch_id), ...]，順序就是報告的呈現順序。"""
    loaded = {}
    for engine_key, batch_id in engines:
        if batch_id not in loaded:
            loaded[batch_id] = load_batch(storage, batch_id)

    # 以第一顆引擎那一批當基準題庫，其他批用「文字」對齊
    base_batch = engines[0][1]
    base_report, base_trials = loaded[base_batch]
    by_text = {bid: {t["text"]: t for t in tr} for bid, (_, tr) in loaded.items()}
    for engine_key, batch_id in engines:
        missing = [t["text"] for t in base_trials if t["text"] not in by_text[batch_id]]
        if missing:
            raise SystemExit(
                f"批次 {batch_id} 對不上基準題庫（{base_batch}）："
                f"缺 {len(missing)} 句，例如「{missing[0][:24]}…」。"
                "題庫不同的兩批不能放進同一份報告。")

    rows = []
    for item, trial in zip(base_report["items"], base_trials):
        row = {"id": item.get("id"), "label": item.get("label"),
               "cat": item.get("category") or "custom",
               "chars": trial.get("chars"), "text": trial["text"]}
        for engine_key, batch_id in engines:
            row[engine_key] = cell(by_text[batch_id][trial["text"]], engine_key)
        rows.append(row)

    keys = [k for k, _ in engines]
    labels = {}
    for engine_key, batch_id in engines:
        report, _ = loaded[batch_id]
        labels[engine_key] = (report.get("engine_labels") or {}).get(engine_key, engine_key)

    payload = {
        "meta": {
            "items": len(rows),
            "repeats": base_report.get("repeats"),
            "speed": base_report.get("speed"),
            "asrBaseline": baseline,
            "runs": [{"engine": k, "batch": b,
                      "start": loaded[b][0].get("created_at"),
                      "end": loaded[b][0].get("finished_at"),
                      "voice": loaded[b][0].get("voice_name")} for k, b in engines],
            "sameBatch": len({b for _, b in engines}) == 1,
        },
        "order": keys,
        "engines": {k: engine_block(k, labels[k], [r[k] for r in rows]) for k in keys},
        "cats": [{"key": ck, "label": cl,
                  "n": sum(1 for r in rows if r["cat"] == ck),
                  **{k: category_block([r for r in rows if r["cat"] == ck], k) for k in keys}}
                 for ck, cl in CATEGORY_ORDER
                 if any(r["cat"] == ck for r in rows)],
        "tests": {
            "pairs": {f"{a}|{b}": pair_tests(rows, a, b)
                      for i, a in enumerate(keys) for b in keys[i + 1:]},
            "vsBase": {k: vs_baseline(rows, k, baseline) for k in keys},
            "sml": {k: strict_minus_lenient(rows, k) for k in keys},
        },
        "rows": rows,
    }
    return payload


# ---------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="把批次跑測整理成多引擎比較報告的資料檔",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("用法\n----\n")[-1])
    ap.add_argument("--engine", action="append", required=True, metavar="KEY=BATCH",
                    help="引擎與它的批次，可以重複；順序就是報告的呈現順序")
    ap.add_argument("--storage", default=str(ROOT / "storage"), type=Path,
                    help="storage 目錄（預設：專案裡的 storage/）")
    ap.add_argument("--baseline", type=float, default=0.0,
                    help="ASR 誤差底線（真人參考音的 CER），用來分級逐句結果")
    ap.add_argument("--out", type=Path, help="輸出檔；不給就印到 stdout")
    args = ap.parse_args(argv)

    engines: list[tuple[str, str]] = []
    for spec in args.engine:
        if "=" not in spec:
            ap.error(f"--engine 要寫成 KEY=BATCH，收到 {spec!r}")
        key, batch = spec.split("=", 1)
        if key in dict(engines):
            ap.error(f"引擎 {key} 指定了兩次")
        engines.append((key.strip(), batch.strip()))

    payload = build(Path(args.storage), engines, args.baseline)
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    if args.out:
        args.out.write_text(text, encoding="utf-8")
        e = payload["engines"]
        print(f"寫出 {args.out}（{len(text) / 1000:.0f} KB，{payload['meta']['items']} 句）")
        for k in payload["order"]:
            print(f"  {e[k]['label']:16s} 耗時中位數 {e[k]['elapsed']['median']:6.2f}s"
                  f"  CER {e[k]['cer']['median']:.4f}"
                  f"  寬鬆 {e[k]['cerL']['median']:.4f}"
                  f"  有效頻寬 {e[k]['bw']['median']:.0f} Hz")
        if not payload["meta"]["sameBatch"]:
            print("  ⚠ 這些引擎不是同一批跑的 —— 速度比較混著跨場次誤差，報告要標示出來。")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
