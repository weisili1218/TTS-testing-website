"""TTS 評測平台 — FastAPI 後端。

同一段文字送進多個 TTS 引擎，比速度、比吐字、比音色。
引擎是設定驅動的：加一顆模型改 engines.json 就好，這個檔案不用動。

啟動：  python -m app.main   或   uvicorn app.main:app --port 8080
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import (
    Body,
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    PlainTextResponse,
    RedirectResponse,
)
from fastapi.staticfiles import StaticFiles

from . import asr, batch, bench, config, engines, jobs, report, voice_sync, voices
from .text_utils import visible_length

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger("tts.api")

ALLOWED_AUDIO_SUFFIX = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".webm", ".mp4"}

app = FastAPI(title="TTS 評測平台", version="3.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def require_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    if config.API_KEY and x_api_key != config.API_KEY:
        raise HTTPException(status_code=401, detail="API key 不正確")


# ================================================================== 狀態
@app.get("/api/status")
def api_status(_=Depends(require_key)):
    """平台健康狀況：所有引擎的能力宣告 + ASR + 哪些維度可比。"""
    # 先並行把能力抓回快取，下面的 describe() 才不會逐顆同步等連線逾時
    engines.prefetch_capabilities()
    engine_rows = [a.describe() for a in engines.all_engines()]
    active = engines.enabled_engines()
    clone_keys = engines.clone_capable_keys()

    if len(active) < 2:
        note = ("目前只有一個引擎可用，沒辦法做盲測對決。"
                "在 engines.json 補上其他引擎的位址就會自動出現在選項裡。")
    elif len(clone_keys) >= 2:
        others = [a.label for a in active if a.key not in clone_keys]
        note = (f"{'、'.join(engines.label_for(k) for k in clone_keys)} 都能用指定的參考音，"
                "推送同一個音色之後這幾顆之間的音色相似度可以比。")
        if others:
            note += f"{'、'.join(others)} 只有內建音色，不列入音色比較。"
    else:
        note = ("目前沒有兩顆以上的引擎能用指定參考音，音色相似度不在比較範圍內；"
                "速度與吐字正確率仍然可以公平比較。")

    return {
        "engines": engine_rows,
        "enabled_engines": [a.key for a in active],
        "clone_capable": clone_keys,
        "asr": asr.status(),
        "queue": jobs.queue_depth(),
        "voice_count": len(voices.list_voices()),
        "needs_voice": any(a.needs_voice for a in active),
        "comparability": {
            "speed": True,
            "intelligibility": config.ASR_ENABLED,
            "timbre": len(clone_keys) >= 2,
            "timbre_group": clone_keys,
            "can_duel": len(active) >= 2,
            "can_rank": len(active) >= 3,
            "note": note,
        },
        "limits": {
            "max_text_chars": config.MAX_TEXT_CHARS,
            "min_prompt_seconds": config.MIN_PROMPT_SECONDS,
            "max_prompt_seconds": config.MAX_PROMPT_SECONDS,
            "max_chars_per_segment": config.MAX_CHARS_PER_SEGMENT,
            "max_repeats": config.MAX_REPEATS,
        },
        "defaults": {
            "score": config.SCORE_BY_DEFAULT,
            "mode": config.DEFAULT_TRIAL_MODE,
            "repeats": config.DEFAULT_REPEATS,
        },
    }


@app.get("/api/health-asr")
def api_health_asr(_=Depends(require_key)):
    return asr.status()


@app.get("/api/health/{engine_key}")
def api_health_engine(engine_key: str, deep: bool = False, _=Depends(require_key)):
    """單一引擎的健康檢查。

    deep=true 會實際合成一句短句。這是唯一可信的檢查方式：實測過有服務在 GPU
    掉進 CUDA 錯誤之後，健康端點仍然回報一切正常，但每一次合成都會失敗。
    """
    adapter = engines.get(engine_key)
    if adapter is None:
        raise HTTPException(status_code=404, detail=f"沒有這個引擎：{engine_key}")
    return {"key": adapter.key, "label": adapter.label,
            "enabled": adapter.enabled, **adapter.health(deep=deep)}


@app.post("/api/engines/reload")
def api_reload_engines(_=Depends(require_key)):
    """重讀 engines.json。改完設定不用重啟服務。"""
    keys = engines.reload_from_config()
    engines.prefetch_capabilities(refresh=True)
    return {"engines": keys, "enabled": engines.enabled_keys()}


@app.post("/api/warmup")
def api_warmup(payload: dict = Body(default={}), _=Depends(require_key)):
    """把模型載進 GPU。冷啟動要 30–90 秒，正式評測前一定要先做。"""
    raw = payload.get("engines")
    engine_keys = [k for k in engines.enabled_keys() if not raw or k in raw] or None
    job_id = jobs.create_job("warmup", {"engines": engine_keys})
    jobs.submit(job_id, lambda progress: bench.warmup(engine_keys, progress))
    return {"job_id": job_id}


# ================================================================== 音色庫
@app.get("/api/voices")
def api_list_voices(_=Depends(require_key)):
    """音色清單，含每個音色在各引擎上的推送狀態。"""
    return {"voices": voice_sync.library_status(),
            "clone_capable": engines.clone_capable_keys()}


@app.post("/api/voices")
async def api_create_voice(
    file: UploadFile = File(...),
    name: str = Form(""),
    transcription: str = Form(""),
    note: str = Form(""),
    auto_transcribe: bool = Form(False),
    _=Depends(require_key),
):
    suffix = Path(file.filename or "upload.wav").suffix.lower()
    if suffix and suffix not in ALLOWED_AUDIO_SUFFIX:
        raise HTTPException(
            status_code=400,
            detail=f"不支援的音檔格式 {suffix}，請用 {', '.join(sorted(ALLOWED_AUDIO_SUFFIX))}",
        )

    config.TMP_DIR.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkstemp(suffix=suffix or ".wav", dir=config.TMP_DIR)[1])
    try:
        with tmp.open("wb") as fh:
            shutil.copyfileobj(file.file, fh)
        meta = voices.create_voice(
            name=name or Path(file.filename or "voice").stem,
            src_audio=tmp,
            transcription=transcription,
            note=note,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        log.exception("建立音色失敗")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        tmp.unlink(missing_ok=True)
        await file.close()

    job_id = None
    if auto_transcribe and not meta.get("transcription"):
        job_id = _submit_transcribe(meta["id"])
    return {"voice": meta, "transcribe_job_id": job_id}


def _submit_transcribe(voice_id: str) -> str:
    """用遠端 Whisper 產生參考音逐字稿（跟評分用的是同一個服務）。"""
    job_id = jobs.create_job("transcribe", {"voice_id": voice_id})

    def _work(progress):
        progress("送去遠端 Whisper 辨識…", 0.3)
        result = asr.transcribe(voices.prompt_path(voice_id))
        progress("寫入音色庫…", 0.8)
        meta = voices.update_voice(voice_id, transcription=result["text"])
        # 逐字稿變了，遠端那份就過期了 —— 立刻重推，不然下次評測會用到舊的
        if config.AUTO_PUSH_VOICE:
            progress("把新的逐字稿推送到各引擎…", 0.9)
            voice_sync.push_voice(voice_id)
        return {"voice_id": voice_id, "transcription": result["text"], "voice": meta,
                "sync": voice_sync.sync_status(voice_id)}

    jobs.submit(job_id, _work)
    return job_id


@app.post("/api/voices/{voice_id}/transcribe")
def api_transcribe(voice_id: str, _=Depends(require_key)):
    if not voices.get_voice(voice_id):
        raise HTTPException(status_code=404, detail="找不到這個音色")
    if not config.ASR_ENABLED:
        raise HTTPException(status_code=400, detail="未設定 ASR 服務，請手動填逐字稿")
    return {"job_id": _submit_transcribe(voice_id)}


@app.patch("/api/voices/{voice_id}")
def api_update_voice(voice_id: str, payload: dict = Body(...), _=Depends(require_key)):
    try:
        meta = voices.update_voice(voice_id, **payload)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="找不到這個音色") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"voice": meta, "sync": voice_sync.sync_status(voice_id)}


@app.delete("/api/voices/{voice_id}")
def api_delete_voice(voice_id: str, _=Depends(require_key)):
    if not voices.get_voice(voice_id):
        raise HTTPException(status_code=404, detail="找不到這個音色")
    # 先把各引擎上的副本刪掉，再刪本機的 —— 順序反過來就找不到對照表了
    removed = voice_sync.delete_everywhere(voice_id)
    try:
        ok = voices.delete_voice(voice_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not ok:
        raise HTTPException(status_code=404, detail="找不到這個音色")
    return {"deleted": voice_id, **removed}


@app.get("/api/voices/{voice_id}/audio")
def api_voice_audio(voice_id: str, _=Depends(require_key)):
    try:
        path = voices.prompt_path(voice_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not path.exists():
        raise HTTPException(status_code=404, detail="找不到參考音檔")
    return FileResponse(path, media_type="audio/wav", filename=f"{voice_id}.wav")


# ------------------------------------------------------------ 音色推送
@app.get("/api/voices/{voice_id}/sync")
def api_voice_sync_status(voice_id: str, _=Depends(require_key)):
    if not voices.get_voice(voice_id):
        raise HTTPException(status_code=404, detail="找不到這個音色")
    return {"voice_id": voice_id, "sync": voice_sync.sync_status(voice_id)}


@app.post("/api/voices/{voice_id}/push")
def api_push_voice(voice_id: str, payload: dict = Body(default={}),
                   _=Depends(require_key)):
    """把這個音色的參考音上傳到各引擎。

    四包 gateway 的音色庫彼此不共用，不推過去的話每顆引擎會拿自己手上剛好有的
    音色去合成，「音色相似度」就是在比不同的東西。
    """
    voice = voices.get_voice(voice_id)
    if not voice:
        raise HTTPException(status_code=404, detail="找不到這個音色")
    if not voice.get("has_audio"):
        raise HTTPException(status_code=400, detail="這個音色沒有 prompt.wav，無法推送")
    if not voice.get("transcription"):
        raise HTTPException(
            status_code=400,
            detail="這個音色還沒有逐字稿。克隆模型少了逐字稿相似度會明顯下降，"
                   "請先按「自動辨識」或手動填寫。")

    force = bool(payload.get("force"))
    targets = payload.get("engines")
    job_id = jobs.create_job("push-voice", {"voice_id": voice_id})
    jobs.submit(job_id, lambda progress: voice_sync.push_voice(
        voice_id, engine_keys=targets, force=force, progress=progress))
    return {"job_id": job_id}


@app.post("/api/voices/push-all")
def api_push_all_voices(payload: dict = Body(default={}), _=Depends(require_key)):
    force = bool(payload.get("force"))
    job_id = jobs.create_job("push-voices")
    jobs.submit(job_id, lambda progress: voice_sync.push_all(force=force, progress=progress))
    return {"job_id": job_id}


# ================================================================== 評測
@app.get("/api/presets")
def api_presets(_=Depends(require_key)):
    """內建測試句：每一類都針對中文 TTS 的已知痛點。"""
    return {"presets": bench.PRESETS,
            "categories": bench.preset_categories(),
            "category_labels": bench.CATEGORY_LABELS}


def _resolve_engines(raw) -> list[str]:
    if isinstance(raw, str):
        raw = [raw]
    raw = raw or engines.enabled_keys()
    # 依登錄順序排，UI 與紀錄的欄位順序才會穩定
    engine_keys = [k for k in engines.keys() if k in raw]
    if not engine_keys:
        raise HTTPException(
            status_code=400,
            detail=f"engines 只接受 {engines.keys()}，至少要選一個")
    for key in engine_keys:
        if not engines.require(key).enabled:
            raise HTTPException(
                status_code=400,
                detail=f"{engines.label_for(key)} 沒有設定完整，目前停用中")
    return engine_keys


def _build_opts(payload: dict, engine_keys: list[str],
                progress=None) -> dict:
    """把請求的參數整理成引擎看得懂的 opts，順便處理音色與前置檢查。"""
    try:
        speed = float(payload.get("speed", 1.0))
    except (TypeError, ValueError):
        speed = 1.0

    mode = str(payload.get("mode", config.DEFAULT_TRIAL_MODE)).strip().lower()
    if mode not in bench.MODES:
        mode = config.DEFAULT_TRIAL_MODE
    # 只有一顆引擎就沒有對手，盲測不成立
    if len(engine_keys) < 2:
        mode = "open"

    try:
        repeats = int(payload.get("repeats", config.DEFAULT_REPEATS))
    except (TypeError, ValueError):
        repeats = config.DEFAULT_REPEATS

    opts = {
        "speed": min(max(speed, 0.5), 2.0),
        "mode": mode,
        "repeats": min(max(repeats, 1), config.MAX_REPEATS),
        "normalize_volume": bool(payload.get("normalize_volume", True)),
        "score": bool(payload.get("score", config.SCORE_BY_DEFAULT)) and config.ASR_ENABLED,
        "instructions": str(payload.get("instructions", "") or "").strip(),
        "category": payload.get("category") or "custom",
        "preset_id": payload.get("preset_id") or "",
        "voice_id": "", "voice_name": "", "voice_transcription": "",
        "voice_prompt_path": "", "_voice_remote_ids": {},
    }

    # 挑音色：只要有任何一顆引擎能用指定參考音，就值得挑一個 ——
    # 沒挑的話那幾顆會退回自己的預設音色，音色維度就白白放棄了
    wanted = str(payload.get("voice_id", "")).strip()
    needs = [k for k in engine_keys if engines.require(k).needs_voice]
    voice = voices.get_voice(wanted) if wanted else None

    if needs and not voice:
        raise HTTPException(
            status_code=404,
            detail=f"{'、'.join(engines.label_for(k) for k in needs)} 只能用克隆音色，"
                   "請先在「音色」分頁選一個。")
    if voice:
        if not voice.get("has_audio"):
            raise HTTPException(
                status_code=400,
                detail=f"音色「{voice.get('name')}」的 prompt.wav 遺失，請重新上傳參考音。")
        # 評測前確保參考音已經在各引擎上，而且不是舊版本
        opts.update(voice_sync.ensure_pushed(voice, engine_keys, progress))

    for key in engine_keys:
        try:
            engines.require(key).validate(opts)
        except ValueError as exc:
            raise HTTPException(
                status_code=400, detail=f"{engines.label_for(key)}：{exc}") from exc
    return opts


@app.post("/api/trials")
def api_create_trial(payload: dict = Body(...), _=Depends(require_key)):
    """跑一次評測：同一段文字丟給選定的引擎，回 job_id 供輪詢。"""
    text = str(payload.get("text", "")).strip()
    if not text:
        raise HTTPException(status_code=400, detail="請輸入要評測的文字")
    if visible_length(text) > config.MAX_TEXT_CHARS:
        raise HTTPException(
            status_code=400, detail=f"文字太長，單次上限 {config.MAX_TEXT_CHARS} 字")

    engine_keys = _resolve_engines(payload.get("engines"))
    opts = _build_opts(payload, engine_keys)

    job_id = jobs.create_job(
        "trial",
        {"engines": engine_keys, "chars": visible_length(text),
         "mode": opts["mode"], "repeats": opts["repeats"], "score": opts["score"]},
    )

    def _work(progress):
        trial = bench.run_trial(text, engine_keys, opts, progress)
        if not any(s.get("filename") for s in trial["samples"].values()):
            raise RuntimeError(
                "；".join(f"{engines.label_for(k)}：{v}" for k, v in trial["errors"].items())
                or "所有引擎都合成失敗")
        return bench.to_public(trial)

    jobs.submit(job_id, _work)
    return {"job_id": job_id, "engines": engine_keys, "mode": opts["mode"]}


@app.get("/api/trials")
def api_list_trials(limit: int = 100, batch_id: str = "", _=Depends(require_key)):
    items = bench.list_trials(limit=min(limit, 500))
    if batch_id:
        items = [t for t in items if t.get("batch_id") == batch_id]
    return {"items": [bench.to_public(t) for t in items]}


@app.get("/api/trials/pending")
def api_pending_trials(limit: int = 50, _=Depends(require_key)):
    """還沒投票的盲測 —— 批次跑完之後可以慢慢聽、慢慢投。"""
    items = [t for t in bench.list_trials(limit=config.TRIALS_LIMIT)
             if t.get("blind") and not t.get("revealed") and not t.get("vote")]
    return {"items": [bench.to_public(t) for t in items[:limit]], "total": len(items)}


@app.get("/api/trials/{trial_id}")
def api_get_trial(trial_id: str, _=Depends(require_key)):
    trial = bench.get_trial(trial_id)
    if not trial:
        raise HTTPException(status_code=404, detail="找不到這筆評測")
    return bench.to_public(trial)


@app.get("/api/trials/{trial_id}/audio/{slot}")
def api_trial_audio(trial_id: str, slot: str, _=Depends(require_key)):
    """盲測用的音檔端點。

    網址只有格子代號，不含引擎名稱 —— 檔名帶引擎代號的 /api/outputs/ 會洩漏答案，
    所以未揭曉的評測一律走這裡拿檔案。
    """
    trial = bench.get_trial(trial_id)
    if not trial:
        raise HTTPException(status_code=404, detail="找不到這筆評測")
    engine_key = trial.get("slots", {}).get(slot.upper())
    if not engine_key:
        raise HTTPException(status_code=404, detail=f"這筆評測沒有 {slot} 這一格")
    sample = trial["samples"].get(engine_key, {})
    path = config.OUTPUTS_DIR / (sample.get("filename") or "")
    if not sample.get("filename") or not path.exists():
        raise HTTPException(status_code=404, detail="找不到音檔")
    # 下載檔名也要中性，不然存到桌面就破功了
    return FileResponse(path, media_type="audio/wav", filename=f"{trial_id}_{slot.upper()}.wav")


@app.post("/api/trials/{trial_id}/vote")
def api_vote(trial_id: str, payload: dict = Body(...), _=Depends(require_key)):
    """投票並揭曉。

    對決模式送 `choice`（格子代號／tie／bad），排名模式送 `ranking`（格子由好到壞）。
    """
    ranking = payload.get("ranking")
    if ranking is not None and not isinstance(ranking, list):
        raise HTTPException(status_code=400, detail="ranking 要是一個陣列")
    try:
        trial = bench.vote(
            trial_id,
            choice=str(payload.get("choice", "")).strip(),
            ranking=ranking,
            note=str(payload.get("note", "")),
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="找不到這筆評測") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return bench.to_public(trial)


@app.post("/api/trials/{trial_id}/reveal")
def api_reveal(trial_id: str, _=Depends(require_key)):
    """不投票直接看答案（這筆就不列入排名統計）。"""
    try:
        return bench.to_public(bench.reveal(trial_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="找不到這筆評測") from exc


@app.delete("/api/trials/{trial_id}")
def api_delete_trial(trial_id: str, _=Depends(require_key)):
    if not bench.delete_trial(trial_id):
        raise HTTPException(status_code=404, detail="找不到這筆評測")
    return {"deleted": trial_id}


@app.delete("/api/trials")
def api_clear_trials(_=Depends(require_key)):
    return {"removed": bench.clear_trials()}


# ================================================================== 批次跑測
@app.post("/api/batches")
def api_create_batch(payload: dict = Body(default={}), _=Depends(require_key)):
    """把整批句子 × 全部引擎跑完，產出一份報表。"""
    engine_keys = _resolve_engines(payload.get("engines"))
    items = batch.resolve_items(
        preset_ids=payload.get("preset_ids"),
        categories=payload.get("categories"),
        custom_texts=payload.get("texts"),
    )
    if not items:
        raise HTTPException(status_code=400, detail="這次批次沒有任何句子")
    for item in items:
        if visible_length(item["text"]) > config.MAX_TEXT_CHARS:
            raise HTTPException(
                status_code=400,
                detail=f"「{item.get('label') or item['text'][:12]}」超過 "
                       f"{config.MAX_TEXT_CHARS} 字上限")

    # 批次預設不盲測：一次跑十幾句，客觀數據吃滿就好，
    # 要主觀資料就把 mode 設成 duel，跑完到競技場慢慢投。
    payload.setdefault("mode", "open")
    opts = _build_opts(payload, engine_keys)

    job_id = jobs.create_job(
        "batch",
        {"engines": engine_keys, "items": len(items),
         "mode": opts["mode"], "repeats": opts["repeats"]},
    )
    jobs.submit(job_id, lambda progress: batch.run_batch(items, engine_keys, opts, progress))
    return {"job_id": job_id, "item_count": len(items), "engines": engine_keys}


@app.get("/api/batches")
def api_list_batches(limit: int = 50, _=Depends(require_key)):
    return {"items": batch.list_reports(limit=limit)}


@app.get("/api/batches/{batch_id}")
def api_get_batch(batch_id: str, _=Depends(require_key)):
    data = batch.get_report(batch_id)
    if not data:
        raise HTTPException(status_code=404, detail="找不到這份報表")
    return data


@app.delete("/api/batches/{batch_id}")
def api_delete_batch(batch_id: str, _=Depends(require_key)):
    if not batch.delete_report(batch_id):
        raise HTTPException(status_code=404, detail="找不到這份報表")
    return {"deleted": batch_id}


# ================================================================== 排行榜
@app.get("/api/leaderboard")
def api_leaderboard(_=Depends(require_key)):
    return report.leaderboard()


@app.get("/api/export.csv")
def api_export_csv(limit: int = 500, _=Depends(require_key)):
    """一列一個 (評測 × 引擎)，丟進試算表或 pandas 都行。"""
    return PlainTextResponse(
        report.export_csv(limit=limit),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="tts-arena.csv"'},
    )


@app.post("/api/asr-baseline")
def api_asr_baseline(payload: dict = Body(default={}), _=Depends(require_key)):
    """量 ASR 自己的誤差底線 —— 沒有這個數字就沒辦法判讀 CER 的絕對水準。"""
    if not config.ASR_ENABLED:
        raise HTTPException(status_code=400, detail="未設定 ASR 服務")
    job_id = jobs.create_job("asr-baseline")

    def _work(progress):
        progress("辨識真人參考音…", 0.5)
        return report.measure_asr_baseline(str(payload.get("voice_id", "")).strip())

    jobs.submit(job_id, _work)
    return {"job_id": job_id}


# ================================================================== 任務／檔案
@app.get("/api/jobs/{job_id}")
def api_job(job_id: str, _=Depends(require_key)):
    job = jobs.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="找不到這個任務")
    return job


@app.get("/api/outputs/{filename}")
def api_output(filename: str, _=Depends(require_key)):
    """已揭曉的音檔直接下載（檔名含引擎代號，方便存檔比對）。"""
    path = config.OUTPUTS_DIR / Path(filename).name
    if not path.exists():
        raise HTTPException(status_code=404, detail="找不到音檔")
    return FileResponse(path, media_type="audio/wav", filename=path.name)


# ================================================================== 前端
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/favicon.ico")
def favicon():
    return RedirectResponse("/static/index.html", status_code=307)


@app.exception_handler(ValueError)
def value_error_handler(_request, exc: ValueError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.on_event("startup")
def _startup():
    log.info("TTS 評測平台啟動 — http://%s:%s", config.HOST, config.PORT)
    active = engines.enabled_engines()
    for adapter in engines.all_engines():
        log.info("引擎 %-16s → %s", adapter.label,
                 adapter.endpoint if adapter.enabled else "（未設定，停用）")
    log.info("ASR 評分   → %s", asr.speech_url() if config.ASR_ENABLED else "（停用）")

    if len(active) < 2:
        log.warning("目前只有 %d 個引擎可用，沒辦法做盲測對決。"
                    "在 engines.json 補上其他引擎的位址即可。", len(active))

    report_ = voices.repair_library()
    log.info("音色庫：%d 個音色（%s）", report_["total"], config.VOICES_DIR)
    if report_["recovered"]:
        log.warning("已重建 %d 個音色的 meta.json：%s",
                    len(report_["recovered"]), "、".join(report_["recovered"]))
    if report_["missing_audio"]:
        log.warning("%d 個音色缺少 prompt.wav：%s",
                    len(report_["missing_audio"]), "、".join(report_["missing_audio"]))

    if config.CHECK_REMOTE_ON_START and active:
        def _check(progress):
            out = {}
            for i, adapter in enumerate(active):
                progress(f"檢查 {adapter.label}…", i / len(active))
                out[adapter.key] = adapter.health(deep=False)
                if not out[adapter.key].get("ok"):
                    log.warning("%s 連線檢查失敗：%s",
                                adapter.label, out[adapter.key].get("error"))
                elif out[adapter.key].get("loaded") is False:
                    log.info("%s 連得上但模型還沒載入，按「暖機」再開始評測", adapter.label)
            clone = engines.clone_capable_keys()
            log.info("可用指定參考音的引擎：%s", clone or "（無）")
            return out

        jobs.submit(jobs.create_job("check"), _check)


def main() -> None:
    import uvicorn

    uvicorn.run("app.main:app", host=config.HOST, port=config.PORT,
                reload=False, log_level="info")


if __name__ == "__main__":
    main()
