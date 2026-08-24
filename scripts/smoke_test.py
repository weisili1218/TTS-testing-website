"""不需要任何遠端服務的完整流程測試。

把所有遠端相依都換成假的（各引擎的 HTTP 呼叫、Whisper ASR），跑完整條路：

    建音色 → 推送到各引擎 → 四方評測 → 盲測遮蔽 → 排名投票
    → Elo 與對戰矩陣 → 分類戰報 → 批次跑測 → CSV 匯出 → 刪除

重點在驗證幾件「錯了就會安靜地得到錯誤結論」的事：
  * 未揭曉時 API 有沒有真的把引擎名稱與**秒數**都藏起來
  * 排名投票有沒有正確拆成成對結果、Elo 有沒有把排序反映出來
  * 對決配對會不會自動去補「交手次數最少」的組合
  * 音色推送的過期偵測（改了逐字稿之後遠端那份要被判定成 stale）
  * 音訊指標抓不抓得到升採樣（宣稱 48k、實際只有 4k 內容）

最後再用一個真的 HTTP stub 伺服器驗證 gateway 轉接器送出去的 payload 長相。

執行： python scripts/smoke_test.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_TMP = tempfile.mkdtemp(prefix="tts-smoke-")
os.environ["STORAGE_DIR"] = _TMP
os.environ["CHECK_REMOTE_ON_START"] = "false"      # 冒煙測試不連任何遠端
os.environ["RANDOMIZE_RUN_ORDER"] = "false"        # 測試要可重現
os.environ["ELO_BOOTSTRAP"] = "40"                 # 跑快一點
os.environ["AUTO_PUSH_VOICE"] = "false"            # 推送單獨測，不要在評測裡偷跑
os.environ["ASR_BASE_URL"] = "http://127.0.0.1:1/fake"
# 四顆假引擎，位址隨便填 —— 反正等一下整個 request_segment 都會被換掉
os.environ["ENGINES_FILE"] = str(Path(_TMP) / "engines.json")
Path(_TMP, "engines.json").write_text(json.dumps([
    {"key": "cosyvoice2",     "label": "CosyVoice2",    "base_url": "http://127.0.0.1:1"},
    {"key": "fun-cosyvoice3", "label": "FunCosyVoice3", "base_url": "http://127.0.0.1:1"},
    {"key": "qwen3-tts",      "label": "Qwen3-TTS",     "base_url": "http://127.0.0.1:1"},
    {"key": "voxcpm2",        "label": "VoxCPM2",       "base_url": "http://127.0.0.1:1"},
]), encoding="utf-8")

import logging  # noqa: E402

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

# 每個 API 呼叫都印一行的話，檢查結果會被沖走
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("tts").setLevel(logging.ERROR)
for _n in ("tts.bench", "tts.engine", "tts.api", "tts.voicesync", "tts.batch",
           "tts.engines", "tts.config", "tts.asr", "tts.voices", "tts.metrics"):
    logging.getLogger(_n).setLevel(logging.ERROR)

from app import config  # noqa: E402

FAILS: list[str] = []


def check(name: str, cond: bool, extra: str = "") -> None:
    print(("  ✅ " if cond else "  ❌ ") + name + (f"  → {extra}" if extra and not cond else ""))
    if not cond:
        FAILS.append(name)


def section(title: str) -> None:
    print(f"\n\033[1m{title}\033[0m")


def _tone(seconds: float, sr: int, hz: float) -> np.ndarray:
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    return (0.3 * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def _speechlike(seconds: float, sr: int, cutoff_hz: float, seed: int = 0) -> np.ndarray:
    """帶限雜訊 —— 用來模擬語音的寬頻內容。

    不能用純正弦波：一個 220Hz 的純音頻寬比本來就只有 0.018，會讓「偵測升採樣」
    的測試變成永遠成立。真正要驗的是「內容頻寬接近奈奎斯特 vs 遠低於奈奎斯特」，
    所以這裡產生一段能量分佈到 cutoff_hz 為止的雜訊。
    """
    rng = np.random.default_rng(seed)
    n = int(sr * seconds)
    spec = np.fft.rfft(rng.standard_normal(n))
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    spec[freqs > cutoff_hz] = 0                      # 截掉 cutoff 以上
    wav = np.fft.irfft(spec, n=n).astype(np.float32)
    peak = float(np.max(np.abs(wav))) or 1.0
    return (wav / peak * 0.3).astype(np.float32)


# ------------------------------------------------- 假的引擎（不連遠端）
from app.engines.gateway import GatewayAdapter  # noqa: E402

# 每顆引擎的假特性：(內容頻寬上限 Hz, 每段秒數, 取樣率, 每段延遲)
# 前三顆的內容都接近自己的奈奎斯特頻率（正常）；
# voxcpm2 刻意宣稱 48k 但內容只到 4k —— 音訊指標應該要抓到這是升採樣上來的。
FAKE = {
    "cosyvoice2":     (11000.0, 1.2, 24000, 0.010),
    "fun-cosyvoice3": (10500.0, 1.1, 24000, 0.014),
    "qwen3-tts":      (7200.0, 1.0, 16000, 0.006),
    "voxcpm2":        (4000.0, 1.3, 48000, 0.020),
}

CAPS = {
    "cosyvoice2":     {"modes": ["clone"], "presets": [], "sample_rate": 24000},
    "fun-cosyvoice3": {"modes": ["clone"], "presets": [], "sample_rate": 24000},
    "qwen3-tts":      {"modes": ["preset"], "presets": ["Vivian", "Ethan"], "sample_rate": 16000},
    "voxcpm2":        {"modes": ["clone", "design", "default"], "presets": [], "sample_rate": 48000},
}

_pushed: dict[tuple, str] = {}     # (engine_key, voice_id) -> 遠端 id


def install_fakes() -> None:
    def fake_caps(self, refresh=False):
        c = dict(CAPS[self.key])
        c.update({"loaded": True, "status": "ready", "needs_ref_audio": "clone" in c["modes"],
                  "id": self.key, "engine": self.key, "error": None})
        return c

    def fake_segment(self, segment, opts):
        cutoff, seconds, sr, delay = FAKE[self.key]
        time.sleep(delay)
        return _speechlike(seconds, sr, cutoff, seed=len(segment)), sr

    GatewayAdapter.capabilities = fake_caps
    GatewayAdapter.request_segment = fake_segment
    GatewayAdapter.warmup = lambda self: {"ok": True, "error": None, "loaded": True}

    # 假的音色推送：直接記在字典裡，不真的發 HTTP
    from app import voice_sync

    def fake_push_one(local_id, adapter, force=False):
        from app import voices as voices_mod

        voice = voices_mod.get_voice(local_id)
        wav_sha = voice_sync._wav_sha(local_id)
        text_sha = voice_sync._text_sha(voice.get("transcription", ""))
        existing = voice_sync._load_map().get(local_id, {}).get(adapter.key, {})
        if (not force and existing.get("remote_id")
                and existing.get("wav_sha") == wav_sha
                and existing.get("transcript_sha") == text_sha):
            return {"engine": adapter.key, "ok": True, "state": voice_sync.SYNCED,
                    "remote_id": existing["remote_id"], "skipped": True}
        remote_id = f"voice_{adapter.key}_{local_id[:6]}"
        _pushed[(adapter.key, local_id)] = remote_id
        voice_sync._record(local_id, adapter.key, {
            "remote_id": remote_id, "error": None, "warnings": [],
            "pushed_at": voice_sync._now(), "wav_sha": wav_sha, "transcript_sha": text_sha})
        return {"engine": adapter.key, "ok": True, "state": voice_sync.SYNCED,
                "remote_id": remote_id}

    voice_sync.push_one = fake_push_one
    voice_sync._delete_remote = lambda adapter, remote_id: True


# ------------------------------------------------------------- 假的 ASR
from app import asr as asr_mod  # noqa: E402

# 每顆引擎「唸錯」的程度不同，CER 才有高低可比
_ASR_TEXT: dict[str, str] = {}


def _fake_transcribe(wav_path):
    """靠檔名裡的引擎代號決定要「聽錯」幾個字。"""
    name = Path(wav_path).name
    engine = next((k for k in FAKE if f"_{k}." in name), "")
    text = _ASR_TEXT.get(engine, _ASR_TEXT.get("__default__", ""))
    return {"text": text, "language": "zh", "elapsed_seconds": 0.01}


asr_mod.transcribe = _fake_transcribe
asr_mod.check_remote = lambda: {"ok": True, "base_url": "fake", "endpoint": "fake", "model": "fake"}


# ---------------------------------------------------------------- 小工具
def wait_job(client, job_id: str, timeout: float = 40.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "error"):
            return job
        time.sleep(0.05)
    return {"status": "timeout", "error": "任務逾時", "result": None}


def make_sample_wav(path: Path, seconds: float = 6.0) -> None:
    sr = 16000
    sf.write(str(path), _tone(seconds, sr, 180.0), sr, subtype="PCM_16")


# ==================================================================== 主流程
def main() -> int:
    install_fakes()
    from app import bench, stats, voice_sync
    from app.main import app

    client = TestClient(app)

    # ------------------------------------------------------------------
    section("1. 引擎登錄與能力探測")
    st = client.get("/api/status").json()
    keys = [e["key"] for e in st["engines"]]
    check("四顆 gateway 都登錄了", all(k in keys for k in FAKE), keys)
    check("能力從 /v1/models 探測（不是寫死）",
          st["clone_capable"] == ["cosyvoice2", "fun-cosyvoice3", "voxcpm2"],
          st["clone_capable"])
    check("qwen3-tts 被正確識別為 preset-only",
          "qwen3-tts" not in st["clone_capable"])
    qwen = next(e for e in st["engines"] if e["key"] == "qwen3-tts")
    check("preset 引擎不強制挑音色", qwen["needs_voice"] is False)
    check("preset 清單探測得到", qwen["presets"] == ["Vivian", "Ethan"], qwen["presets"])
    vox = next(e for e in st["engines"] if e["key"] == "voxcpm2")
    check("voxcpm2 宣稱 48kHz", vox["sample_rate"] == 48000)

    # ------------------------------------------------------------------
    section("2. 音色庫與推送")
    wav = Path(_TMP) / "ref.wav"
    make_sample_wav(wav)
    with wav.open("rb") as fh:
        res = client.post("/api/voices",
                          files={"file": ("ref.wav", fh, "audio/wav")},
                          data={"name": "測試音色", "transcription": "這是一段測試用的參考音"})
    check("建立音色", res.status_code == 200, res.text[:200])
    voice_id = res.json()["voice"]["id"]

    res = client.post(f"/api/voices/{voice_id}/push", json={})
    job = wait_job(client, res.json()["job_id"])
    check("推送任務完成", job["status"] == "done", str(job.get("error"))[:200])
    pushed = {r["engine"] for r in (job["result"] or {}).get("results", [])}
    check("只推給能克隆的三顆", pushed == {"cosyvoice2", "fun-cosyvoice3", "voxcpm2"}, pushed)

    sync = client.get(f"/api/voices/{voice_id}/sync").json()["sync"]
    check("三顆狀態是 synced",
          all(sync[k]["state"] == "synced" for k in ["cosyvoice2", "fun-cosyvoice3", "voxcpm2"]))
    check("qwen3-tts 標成 unsupported 而不是失敗",
          sync["qwen3-tts"]["state"] == "unsupported", sync["qwen3-tts"])

    # 改逐字稿之後，遠端那份就過期了 —— 這個偵測壞掉會安靜地拿舊參考音去比
    client.patch(f"/api/voices/{voice_id}", json={"transcription": "改過的逐字稿內容不一樣了"})
    sync2 = client.get(f"/api/voices/{voice_id}/sync").json()["sync"]
    check("改逐字稿後偵測到 stale", sync2["cosyvoice2"]["state"] == "stale", sync2["cosyvoice2"])
    check("stale 的音色不會被當成可用",
          voice_sync.remote_ids(voice_id) == {}, voice_sync.remote_ids(voice_id))
    client.patch(f"/api/voices/{voice_id}", json={"transcription": "這是一段測試用的參考音"})
    check("改回來之後恢復 synced",
          client.get(f"/api/voices/{voice_id}/sync").json()["sync"]["cosyvoice2"]["state"] == "synced")

    # ------------------------------------------------------------------
    section("3. 四方評測與盲測遮蔽")
    _ASR_TEXT.update({
        "__default__": "這是一段測試用的文字",
        "cosyvoice2":     "這是一段測試用的文字",       # CER 0
        "fun-cosyvoice3": "這是一段測試用的文本",       # 錯 1 字
        "qwen3-tts":      "這是一段測試用文字",         # 少 1 字
        "voxcpm2":        "這是一段測試的文字",         # 少 1 字
    })
    TEXT = "這是一段測試用的文字"

    res = client.post("/api/trials", json={
        "text": TEXT, "engines": list(FAKE), "mode": "ranking",
        "voice_id": voice_id, "score": True,
        "category": "news", "preset_id": "news",
    })
    check("送出四方評測", res.status_code == 200, res.text[:300])
    job = wait_job(client, res.json()["job_id"])
    check("評測跑完", job["status"] == "done", str(job.get("error"))[:300])
    trial = job["result"]
    trial_id = trial["id"]

    check("四個引擎都有格子", len(trial["slots"]) == 4, trial["slots"])
    check("未揭曉時不給 samples", trial["samples"] is None)
    blob = json.dumps(trial, ensure_ascii=False)
    check("未揭曉時不洩漏引擎名稱", not any(k in blob for k in FAKE), blob[:200])
    check("未揭曉時不洩漏秒數（速度差距本身就是識別特徵）",
          "elapsed_seconds" not in blob)
    check("未揭曉時不洩漏參賽名單", trial["engines"] is None, trial["engines"])
    check("未揭曉時的 comparability 不含引擎名稱",
          "note" not in (trial.get("comparability") or {}), trial.get("comparability"))

    # 音檔要走中性網址，檔名不能帶引擎代號
    slot_url = trial["slots"][0]["url"]
    audio = client.get(slot_url)
    check("盲測音檔拿得到", audio.status_code == 200)
    check("下載檔名不含引擎代號",
          not any(k in audio.headers.get("content-disposition", "") for k in FAKE),
          audio.headers.get("content-disposition"))

    # ------------------------------------------------------------------
    section("4. 排名投票與揭曉")
    slots = [s["slot"] for s in trial["slots"]]
    res = client.post(f"/api/trials/{trial_id}/vote", json={"ranking": slots})
    check("排名投票成功", res.status_code == 200, res.text[:300])
    revealed = res.json()
    check("揭曉後有 samples", revealed["samples"] is not None)
    check("揭曉後每個格子都有引擎名稱",
          all(s.get("engine_label") for s in revealed["slots"]))
    check("重複投票會被擋",
          client.post(f"/api/trials/{trial_id}/vote", json={"choice": "A"}).status_code == 400)

    by_engine = {s["engine"]: s for s in revealed["samples"]}
    check("CER 有算出來", all(s.get("cer") is not None for s in by_engine.values()))
    check("完全唸對的 CER 是 0", by_engine["cosyvoice2"]["cer"] == 0.0,
          by_engine["cosyvoice2"]["cer"])
    check("唸錯的 CER 大於 0", by_engine["fun-cosyvoice3"]["cer"] > 0)

    # ------------------------------------------------------------------
    section("5. 音訊客觀指標")
    vox_metrics = by_engine["voxcpm2"].get("audio_metrics") or {}
    check("有算音訊指標", bool(vox_metrics) and not vox_metrics.get("error"), vox_metrics)
    check("抓到 voxcpm2 是升採樣（宣稱 48k、內容只到 4k）",
          (vox_metrics.get("bandwidth_ratio") or 1) < 0.4,
          f"bandwidth_ratio={vox_metrics.get('bandwidth_ratio')}")
    check("有對應的警訊文字",
          any("頻寬" in w for w in by_engine["voxcpm2"].get("audio_warnings", [])),
          by_engine["voxcpm2"].get("audio_warnings"))
    cosy_metrics = by_engine["cosyvoice2"].get("audio_metrics") or {}
    check("正常引擎不會被誤報升採樣",
          (cosy_metrics.get("bandwidth_ratio") or 0) > 0.5,
          f"bandwidth_ratio={cosy_metrics.get('bandwidth_ratio')}")

    # ------------------------------------------------------------------
    section("6. 排名如何變成 Elo")
    pairs = stats.pairs_from_vote(bench.get_trial(trial_id))
    check("四方排名拆成 6 組成對結果", len(pairs) == 6, len(pairs))
    check("排名來源有標記", all(p["source"] == "ranking" for p in pairs))
    winner_key = by_engine[[s["engine"] for s in revealed["slots"] if s["slot"] == slots[0]][0]]["engine"]
    check("第一名贏了其他三個",
          sum(1 for p in pairs if p["winner"] == winner_key) == 3)

    board = client.get("/api/leaderboard").json()
    ranked = {r["engine"]: r for r in board["engines"]}
    check("Elo 有算出來", ranked[winner_key]["elo"] > config.ELO_START,
          ranked[winner_key]["elo"])
    check("排名第一的 Elo 最高",
          max(board["engines"], key=lambda r: r["elo"] or 0)["engine"] == winner_key)
    check("樣本太少時會提醒",
          any("Elo" in c for c in board["caveats"]), board["caveats"])
    check("沒量 ASR 底線時會提醒",
          any("誤差底線" in c for c in board["caveats"]))
    check("取樣率不一致時會提醒",
          any("取樣率" in c for c in board["caveats"]), board["caveats"])

    # ------------------------------------------------------------------
    section("7. 對決配對會補最少交手的組合")
    seen_pairs = set()
    for i in range(6):
        res = client.post("/api/trials", json={
            "text": f"第{i}次對決測試。", "engines": list(FAKE),
            "mode": "duel", "voice_id": voice_id, "score": False})
        job = wait_job(client, res.json()["job_id"])
        if job["status"] != "done":
            check(f"第 {i+1} 次對決", False, str(job.get("error"))[:200])
            break
        t = job["result"]
        full = bench.get_trial(t["id"])
        seen_pairs.add(frozenset(full["slots"].values()))
        client.post(f"/api/trials/{t['id']}/vote", json={"choice": t["slots"][0]["slot"]})
    check("六次對決把 6 種組合全部配到", len(seen_pairs) == 6, f"{len(seen_pairs)} 種")

    board = client.get("/api/leaderboard").json()
    matrix = board["matrix"]
    filled = sum(1 for row in matrix["rows"] for k, c in row["cells"].items() if c and c["games"])
    check("對戰矩陣填滿（4×3 = 12 格）", filled == 12, f"{filled} 格")

    # ------------------------------------------------------------------
    section("8. duel 模式：跑全部、只盲測兩個")
    duel = bench.get_trial([t for t in bench.list_trials(10)][0]["id"])
    check("duel 跑了全部四顆引擎", len(duel["samples"]) == 4, len(duel["samples"]))
    check("但只有兩個格子", len(duel["slots"]) == 2, duel["slots"])

    res = client.post("/api/trials", json={
        "text": "遮蔽測試。", "engines": list(FAKE), "mode": "duel",
        "voice_id": voice_id, "score": False})
    hidden = wait_job(client, res.json()["job_id"])["result"]
    check("未揭曉時會告知有幾顆被藏起來", hidden.get("hidden_count") == 2,
          hidden.get("hidden_count"))
    client.post(f"/api/trials/{hidden['id']}/reveal", json={})

    # ------------------------------------------------------------------
    section("9. 重複跑與穩定性")
    res = client.post("/api/trials", json={
        "text": "穩定性測試。", "engines": ["cosyvoice2", "qwen3-tts"],
        "mode": "open", "repeats": 3, "voice_id": voice_id, "score": False})
    job = wait_job(client, res.json()["job_id"])
    check("重複跑完成", job["status"] == "done", str(job.get("error"))[:200])
    rep_trial = job["result"]
    s = next(x for x in rep_trial["samples"] if x["engine"] == "cosyvoice2")
    check("每顆引擎跑了 3 次", len(s["runs"]) == 3, len(s["runs"]))
    check("有耗時分佈統計", s["elapsed_spread"]["n"] == 3, s.get("elapsed_spread"))
    full = bench.get_trial(rep_trial["id"])
    check("輪替執行（不是同一顆連跑三次）",
          len(full["run_order"]) == 3 and all(len(r) == 2 for r in full["run_order"]),
          full["run_order"])

    # ------------------------------------------------------------------
    section("10. 分類戰報")
    board = client.get("/api/leaderboard").json()
    check("分類表有資料", bool(board["categories"]), board["categories"])
    news = board["categories"].get("news")
    check("news 類別有記錄到", news is not None and news["trials"] >= 1, news)
    check("每類都算得出 CER 最低的引擎",
          news is not None and news.get("best_cer") in FAKE, news and news.get("best_cer"))

    # ------------------------------------------------------------------
    section("11. 批次跑測")
    res = client.post("/api/batches", json={
        "engines": ["cosyvoice2", "qwen3-tts"], "categories": ["polyphone"],
        "mode": "open", "voice_id": voice_id, "score": True})
    check("批次送出", res.status_code == 200, res.text[:300])
    job = wait_job(client, res.json()["job_id"], timeout=60)
    check("批次跑完", job["status"] == "done", str(job.get("error"))[:300])
    rep = job["result"]
    check("批次跑了 2 句（polyphone 類別有兩句）", rep["item_count"] == 2, rep["item_count"])
    check("批次全部成功", rep["done"] == 2 and rep["failed"] == 0, rep)
    check("報表有摘要", rep["summary"] is not None)
    check("摘要選得出最快的引擎", rep["summary"]["fastest"] in FAKE, rep["summary"]["fastest"])
    check("報表存得起來", client.get(f"/api/batches/{rep['id']}").status_code == 200)
    check("批次的評測帶得到 batch_id",
          all(t["batch_id"] == rep["id"]
              for t in client.get(f"/api/trials?batch_id={rep['id']}").json()["items"]))

    # ------------------------------------------------------------------
    section("12. 待投票佇列")
    res = client.post("/api/batches", json={
        "engines": list(FAKE), "categories": ["short"],
        "mode": "duel", "voice_id": voice_id, "score": False})
    wait_job(client, res.json()["job_id"], timeout=60)
    pending = client.get("/api/trials/pending").json()
    check("盲測批次會進待投票佇列", pending["total"] >= 1, pending["total"])
    check("待投票的項目仍然是遮蔽的",
          all(t["samples"] is None for t in pending["items"]))

    # ------------------------------------------------------------------
    section("13. ASR 誤差底線")
    # 沒有這個數字就沒辦法判讀 CER 的絕對水準：Whisper 聽真人錄音也不會是 0
    _ASR_TEXT["__default__"] = "這是一段測試用的參考音"     # 跟音色逐字稿一字不差
    res = client.post("/api/asr-baseline", json={})
    job = wait_job(client, res.json()["job_id"])
    check("底線量測完成", job["status"] == "done", str(job.get("error"))[:200])
    base = job["result"]
    check("完全聽對時底線是 0", base["cer"] == 0.0, base)

    # 參考逐字稿「這是一段測試用的參考音」是 11 個字，漏掉「的」-> 1/11
    _ASR_TEXT["__default__"] = "這是一段測試用參考音"
    base = wait_job(client, client.post("/api/asr-baseline", json={})
                    .json()["job_id"])["result"]
    check("聽漏一個字時底線 = 1/11", base["cer"] == round(1 / 11, 4), base["cer"])

    board = client.get("/api/leaderboard").json()
    check("排行榜帶上底線",
          (board["asr_baseline"] or {}).get("cer") == round(1 / 11, 4),
          board["asr_baseline"])
    check("量了底線之後就不再提醒「還沒量」",
          not any("還沒量" in c for c in board["caveats"]), board["caveats"])

    # ------------------------------------------------------------------
    section("14. 匯出與清理")
    csv = client.get("/api/export.csv")
    check("CSV 匯出成功", csv.status_code == 200)
    lines = csv.text.strip().splitlines()
    check("CSV 有表頭與資料列", len(lines) > 2, len(lines))
    check("CSV 含音訊指標欄位", "effective_bandwidth_hz" in lines[0])

    check("路徑穿越被擋",
          client.get("/api/outputs/..%2F..%2Fetc%2Fpasswd").status_code in (400, 404))
    check("找不到的評測回 404",
          client.get("/api/trials/no-such-trial").status_code == 404)

    before = len(client.get("/api/trials?limit=500").json()["items"])
    client.delete(f"/api/trials/{trial_id}")
    after = len(client.get("/api/trials?limit=500").json()["items"])
    check("刪除單筆評測", after == before - 1, f"{before} -> {after}")
    check("音檔一起刪掉",
          not list(config.OUTPUTS_DIR.glob(f"{trial_id}_*.wav")))

    removed = client.delete("/api/trials").json()["removed"]
    check("清空全部紀錄", removed > 0 and not client.get("/api/trials").json()["items"])
    check("音檔全部清乾淨", not list(config.OUTPUTS_DIR.glob("*.wav")))

    # ------------------------------------------------------------------
    section("15. 真 HTTP：gateway 送出去的 payload")
    check_remote_call()

    print()
    if FAILS:
        print(f"\033[31m{len(FAILS)} 項失敗：\033[0m")
        for f in FAILS:
            print("  ·", f)
        return 1
    print("\033[32m全部通過\033[0m")
    return 0


# --------------------------------------------------- 真 HTTP payload 驗證
def check_remote_call() -> None:
    """起一個真的 HTTP 伺服器，驗證 GatewayAdapter 送出去的 JSON 長相。

    前面的測試把 request_segment 整個換掉了，所以 payload 的組法完全沒被驗到。
    這裡補上：欄位名稱錯了、speed 該不該送搞反了，實機才會發現就太晚了。
    """
    import io
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    captured: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("content-length", 0))
            captured.append({
                "path": self.path,
                "body": json.loads(self.rfile.read(length) or b"{}"),
                "auth": self.headers.get("authorization"),
            })
            buf = io.BytesIO()
            sf.write(buf, _tone(0.4, 22050, 300.0), 22050, format="WAV", subtype="PCM_16")
            data = buf.getvalue()
            self.send_response(200)
            self.send_header("content-type", "audio/wav")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *_a):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]

    from app import config as cfg

    # 用乾淨的類別（不帶前面塞進去的假 request_segment）
    import importlib

    import app.engines.gateway as gw_mod
    importlib.reload(gw_mod)

    spec = dict(cfg.GATEWAY_DEFAULTS)
    spec.update({"key": "probe", "label": "Probe", "model": "probe-model",
                 "base_url": f"http://127.0.0.1:{port}", "api_key": "secret-key",
                 "speed_mode": "remote", "language": "zh",
                 "extra_fields": {"auto_bopomofo": True}})
    adapter = gw_mod.GatewayAdapter(spec)
    adapter.capabilities = lambda refresh=False: {
        "modes": ["clone"], "presets": [], "loaded": True, "sample_rate": 22050}

    opts = {"speed": 1.25, "normalize_volume": False,
            "voice_id": "local-1", "voice_name": "測試音色",
            "voice_transcription": "逐字稿",
            "_voice_remote_ids": {"probe": "voice_remote_abc"}}
    try:
        adapter.synthesize("測試一句話。", opts, None)
    except Exception as exc:  # noqa: BLE001
        check("gateway 合成請求送得出去", False, str(exc)[:200])
        server.shutdown()
        return

    check("gateway 合成請求送得出去", len(captured) == 1, f"{len(captured)} 次")
    if not captured:
        server.shutdown()
        return
    req = captured[0]
    body = req["body"]
    check("打的是 /v1/audio/speech", req["path"] == "/v1/audio/speech", req["path"])
    check("帶 Bearer token", req["auth"] == "Bearer secret-key", req["auth"])
    check("model 用設定的名稱", body.get("model") == "probe-model", body.get("model"))
    check("voice 用推送後的遠端 id（不是本機 id）",
          body.get("voice") == "voice_remote_abc", body.get("voice"))
    check("speed_mode=remote 時有送 speed", body.get("speed") == 1.25, body.get("speed"))
    check("language 有帶", body.get("language") == "zh", body.get("language"))
    check("extra_fields 有合併進去", body.get("auto_bopomofo") is True, body)
    check("預設不送 instructions（會讓 CosyVoice 離開 zero-shot）",
          "instructions" not in body, body)

    # speed_mode=local 時不該送 speed —— 送一個對方會忽略的參數比不送更糟
    captured.clear()
    spec2 = dict(spec)
    spec2["speed_mode"] = "local"
    adapter2 = gw_mod.GatewayAdapter(spec2)
    adapter2.capabilities = lambda refresh=False: {
        "modes": ["clone"], "presets": [], "loaded": True, "sample_rate": 22050}
    adapter2.synthesize("測試一句話。", dict(opts), None)
    check("speed_mode=local 時不送 speed", "speed" not in captured[0]["body"],
          captured[0]["body"])

    server.shutdown()


if __name__ == "__main__":
    code = main()
    print(f"\n暫存資料夾：{_TMP}")
    sys.exit(code)
