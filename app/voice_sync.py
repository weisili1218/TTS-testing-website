"""把本機音色庫的參考音推送到各引擎，並記住對照表。

為什麼需要這個檔案
------------------
四包 gateway **各有各的音色庫**（`work/voices/`），彼此不共用。同一段參考音
沒有推到每一包，克隆比較就是假的 —— 每顆引擎會拿自己手上剛好有的音色去合成，
測出來的「音色相似度」在比不同的東西。

所以本機音色庫是唯一的真實來源，這裡負責：

    storage/voices/<local_id>/prompt.wav
        ├─ POST http://host:8001/v1/voices  ->  voice_a1b2...   (cosyvoice2)
        ├─ POST http://host:8002/v1/voices  ->  voice_c3d4...   (fun-cosyvoice3)
        └─ POST http://host:8004/v1/voices  ->  voice_e5f6...   (voxcpm2)

對照表存在 `storage/voice_map.json`：

    {"<local_id>": {"<engine_key>": {"remote_id": ..., "wav_sha": ..., ...}}}

`wav_sha` 與 `transcript_sha` 是用來偵測「本機改過了，遠端是舊的」。參考音重傳
或逐字稿改了，狀態就會變成 stale，下次評測前會自動重推（AUTO_PUSH_VOICE）。

不能克隆的引擎（qwen3-tts 只有 preset）會被標成 unsupported —— 不是錯誤，
是這顆引擎本來就沒有這個能力，UI 要據此把它排除在音色比較之外。
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from datetime import datetime, timezone
from typing import Optional

from . import config, engines, voices

log = logging.getLogger("tts.voicesync")

_lock = threading.Lock()

# 每個引擎在一個音色上的狀態
SYNCED = "synced"            # 推過了，而且跟本機一致
STALE = "stale"              # 推過了，但本機的參考音或逐字稿之後改過
MISSING = "missing"          # 還沒推
UNSUPPORTED = "unsupported"  # 這顆引擎不能克隆（例如只有 preset）
FAILED = "failed"            # 推過但失敗了


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def _wav_sha(local_id: str) -> str:
    path = voices.prompt_path(local_id)
    if not path.exists():
        return ""
    return _sha(path.read_bytes())


def _text_sha(text: str) -> str:
    return _sha((text or "").strip().encode("utf-8"))


# ---------------------------------------------------------------- 對照表
def _load_map() -> dict:
    if not config.VOICE_MAP_FILE.exists():
        return {}
    try:
        data = json.loads(config.VOICE_MAP_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001
        log.warning("voice_map.json 讀取失敗，當成空的處理")
        return {}


def _save_map(data: dict) -> None:
    config.VOICE_MAP_FILE.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _record(local_id: str, engine_key: str, entry: dict) -> None:
    with _lock:
        data = _load_map()
        data.setdefault(local_id, {})[engine_key] = entry
        _save_map(data)


def _forget(local_id: str, engine_key: Optional[str] = None) -> None:
    with _lock:
        data = _load_map()
        if engine_key is None:
            data.pop(local_id, None)
        elif local_id in data:
            data[local_id].pop(engine_key, None)
        _save_map(data)


def remote_ids(local_id: str) -> dict[str, str]:
    """這個本機音色在各引擎上的遠端 voice id（只回狀態是 synced 的）。

    刻意排除 stale：拿舊的參考音去比新的，比出來的東西沒有意義。
    """
    out: dict[str, str] = {}
    entries = _load_map().get(local_id, {})
    wav_sha = _wav_sha(local_id)
    voice = voices.get_voice(local_id) or {}
    text_sha = _text_sha(voice.get("transcription", ""))
    for engine_key, entry in entries.items():
        if not entry.get("remote_id"):
            continue
        if entry.get("wav_sha") != wav_sha or entry.get("transcript_sha") != text_sha:
            continue
        out[engine_key] = entry["remote_id"]
    return out


# ---------------------------------------------------------------- 推送
def _clone_targets(engine_keys: Optional[list[str]] = None) -> list:
    """能接受我們上傳參考音的引擎。"""
    out = []
    for adapter in engines.enabled_engines():
        if engine_keys is not None and adapter.key not in engine_keys:
            continue
        if getattr(adapter, "can_clone", False) and hasattr(adapter, "spec"):
            out.append(adapter)
    return out


def push_one(local_id: str, adapter, force: bool = False) -> dict:
    """把一個本機音色推到一顆引擎。回傳這次推送的結果。"""
    voice = voices.get_voice(local_id)
    if not voice:
        raise KeyError(local_id)
    wav = voices.prompt_path(local_id)
    if not wav.exists():
        return {"engine": adapter.key, "ok": False, "state": FAILED,
                "error": "本機音色缺少 prompt.wav"}

    wav_sha = _wav_sha(local_id)
    text_sha = _text_sha(voice.get("transcription", ""))
    existing = _load_map().get(local_id, {}).get(adapter.key, {})
    if (not force and existing.get("remote_id")
            and existing.get("wav_sha") == wav_sha
            and existing.get("transcript_sha") == text_sha):
        return {"engine": adapter.key, "ok": True, "state": SYNCED,
                "remote_id": existing["remote_id"], "skipped": True}

    # 換過參考音的話，先把遠端那個舊的刪掉，免得對方音色庫越積越多同名音色
    if existing.get("remote_id"):
        _delete_remote(adapter, existing["remote_id"])

    url = adapter.base_url + adapter.spec["voices_path"]
    data = {
        "name": voice.get("name") or local_id,
        "transcript": voice.get("transcription") or "",
        "default_engine": adapter.spec.get("model") or adapter.key,
    }
    language = str(adapter.spec.get("language") or "").strip()
    if language:
        data["language"] = language

    try:
        with wav.open("rb") as fh:
            resp = adapter._http().post(
                url,
                data=data,
                files={"file": (f"{local_id}.wav", fh, "audio/wav")},
                timeout=120,
            )
    except Exception as exc:  # noqa: BLE001
        entry = {"remote_id": "", "error": f"{exc.__class__.__name__}: {exc}",
                 "pushed_at": _now(), "wav_sha": wav_sha, "transcript_sha": text_sha}
        _record(local_id, adapter.key, entry)
        return {"engine": adapter.key, "ok": False, "state": FAILED, "error": entry["error"]}

    if resp.status_code >= 400:
        from .engines.base import describe_http_error

        err = describe_http_error(resp.status_code, resp.content, url)
        _record(local_id, adapter.key,
                {"remote_id": "", "error": err, "pushed_at": _now(),
                 "wav_sha": wav_sha, "transcript_sha": text_sha})
        return {"engine": adapter.key, "ok": False, "state": FAILED, "error": err}

    try:
        payload = resp.json()
    except Exception:  # noqa: BLE001
        payload = {}
    remote = payload.get("voice") or {}
    remote_id = str(remote.get("id") or "")
    warnings = [str(w) for w in (payload.get("warnings") or [])]

    if not remote_id:
        err = f"上傳成功但回應沒有 voice.id：{str(payload)[:200]}"
        _record(local_id, adapter.key,
                {"remote_id": "", "error": err, "pushed_at": _now(),
                 "wav_sha": wav_sha, "transcript_sha": text_sha})
        return {"engine": adapter.key, "ok": False, "state": FAILED, "error": err}

    entry = {
        "remote_id": remote_id,
        "remote_name": remote.get("name"),
        "duration_sec": remote.get("duration_sec"),
        "has_audio": remote.get("has_audio"),
        "warnings": warnings,
        "error": None,
        "pushed_at": _now(),
        "wav_sha": wav_sha,
        "transcript_sha": text_sha,
    }
    _record(local_id, adapter.key, entry)
    log.info("音色 %s 推送到 %s -> %s", local_id, adapter.key, remote_id)
    return {"engine": adapter.key, "ok": True, "state": SYNCED,
            "remote_id": remote_id, "warnings": warnings}


def push_voice(local_id: str, engine_keys: Optional[list[str]] = None,
               force: bool = False, progress=None) -> dict:
    """把一個本機音色推到所有（或指定的）可克隆引擎。"""
    targets = _clone_targets(engine_keys)
    results: list[dict] = []
    for i, adapter in enumerate(targets):
        if progress:
            progress(f"推送到 {adapter.label}…", i / max(len(targets), 1))
        try:
            results.append(push_one(local_id, adapter, force=force))
        except Exception as exc:  # noqa: BLE001
            log.exception("推送到 %s 失敗", adapter.key)
            results.append({"engine": adapter.key, "ok": False,
                            "state": FAILED, "error": f"{type(exc).__name__}: {exc}"})
    if progress:
        progress("推送完成", 1.0)
    return {"voice_id": local_id, "results": results,
            "ok": all(r.get("ok") for r in results) if results else True}


def push_all(force: bool = False, progress=None) -> dict:
    """把音色庫裡「有參考音也有逐字稿」的音色全部推一輪。"""
    usable = [v for v in voices.list_voices()
              if v.get("has_audio") and v.get("transcription")]
    out = []
    for i, v in enumerate(usable):
        if progress:
            progress(f"推送音色「{v['name']}」…", i / max(len(usable), 1))
        out.append(push_voice(v["id"], force=force))
    if progress:
        progress("全部推送完成", 1.0)
    return {"voices": out, "count": len(out)}


def _delete_remote(adapter, remote_id: str) -> bool:
    try:
        url = f"{adapter.base_url}{adapter.spec['voices_path']}/{remote_id}"
        resp = adapter._http().delete(url, timeout=30)
        return resp.status_code < 400
    except Exception as exc:  # noqa: BLE001
        log.warning("刪除 %s 上的遠端音色 %s 失敗：%s", adapter.key, remote_id, exc)
        return False


def delete_everywhere(local_id: str) -> dict:
    """本機音色刪掉時，順手把各引擎上的副本也刪掉。"""
    entries = _load_map().get(local_id, {})
    removed = []
    for engine_key, entry in entries.items():
        adapter = engines.get(engine_key)
        if adapter and entry.get("remote_id") and hasattr(adapter, "spec"):
            if _delete_remote(adapter, entry["remote_id"]):
                removed.append(engine_key)
    _forget(local_id)
    return {"voice_id": local_id, "removed_from": removed}


# ---------------------------------------------------------------- 狀態
def sync_status(local_id: str) -> dict[str, dict]:
    """這個音色在每顆啟用中引擎上的狀態，給 UI 顯示用。"""
    voice = voices.get_voice(local_id) or {}
    wav_sha = _wav_sha(local_id)
    text_sha = _text_sha(voice.get("transcription", ""))
    entries = _load_map().get(local_id, {})

    out: dict[str, dict] = {}
    for adapter in engines.enabled_engines():
        if not getattr(adapter, "can_clone", False):
            # 「探測到它不能克隆」跟「根本連不上所以不知道」是兩回事，
            # 都標成「不支援克隆」會讓人以為引擎沒問題、只是能力不夠。
            caps = adapter.capabilities() if hasattr(adapter, "capabilities") else {}
            if caps.get("error"):
                detail = f"連不上，無法判斷能力：{caps['error']}"
            elif caps.get("modes"):
                detail = f"只支援 {'／'.join(caps['modes'])}，不能用指定的參考音"
            else:
                detail = "這個引擎沒有克隆能力"
            out[adapter.key] = {"state": UNSUPPORTED, "label": adapter.label,
                                "detail": detail}
            continue
        entry = entries.get(adapter.key)
        if not entry or not entry.get("remote_id"):
            out[adapter.key] = {"state": FAILED if (entry or {}).get("error") else MISSING,
                                "label": adapter.label,
                                "detail": (entry or {}).get("error") or "尚未推送"}
            continue
        if entry.get("wav_sha") != wav_sha or entry.get("transcript_sha") != text_sha:
            out[adapter.key] = {"state": STALE, "label": adapter.label,
                                "remote_id": entry["remote_id"],
                                "detail": "本機的參考音或逐字稿改過，遠端還是舊的"}
            continue
        out[adapter.key] = {"state": SYNCED, "label": adapter.label,
                            "remote_id": entry["remote_id"],
                            "detail": f"已推送 {entry.get('pushed_at', '')}",
                            "warnings": entry.get("warnings") or []}
    return out


def library_status() -> list[dict]:
    """整個音色庫的推送狀態。"""
    out = []
    for v in voices.list_voices():
        status = sync_status(v["id"])
        synced = [k for k, s in status.items() if s["state"] == SYNCED]
        out.append({
            **v,
            "sync": status,
            "synced_engines": synced,
            "ready_for_timbre": len(synced) >= 2,
        })
    return out


# ---------------------------------------------------------------- 評測入口
def voice_opts(voice: dict) -> dict:
    """把音色庫的一筆資料攤成引擎看得懂的欄位（含各引擎的遠端 id）。"""
    return {
        "voice_id": voice["id"],
        "voice_name": voice.get("name", ""),
        "voice_transcription": voice.get("transcription", ""),
        "voice_prompt_path": str(voices.prompt_path(voice["id"])),
        "voice_language": voice.get("language", ""),
        "_voice_remote_ids": remote_ids(voice["id"]),
    }


def ensure_pushed(voice: dict, engine_keys: list[str], progress=None) -> dict:
    """評測前確保參考音已經在需要的引擎上。回傳更新過的 opts。

    這是「音色可比」的守門員：沒推成功的引擎，等一下 _pick_voice 會退回
    preset / default，UI 就會把它標成音色不可比 —— 不會安靜地比錯東西。
    """
    if not config.AUTO_PUSH_VOICE:
        return voice_opts(voice)

    status = sync_status(voice["id"])
    need = [k for k in engine_keys
            if status.get(k, {}).get("state") in (MISSING, STALE, FAILED)]
    if need:
        if progress:
            progress(f"把參考音推送到 {len(need)} 顆引擎…", 0.01)
        push_voice(voice["id"], engine_keys=need)
    return voice_opts(voice)
