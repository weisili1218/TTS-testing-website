"""音色庫：上傳的參考音檔 + 逐字稿，存在 storage/voices/<id>/。

每個音色資料夾內容：
    prompt.wav   16k 單聲道參考音
    meta.json    名稱、逐字稿、長度、建立時間、備註
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from . import config

log = logging.getLogger("tts.voices")

_lock = threading.Lock()

META_NAME = "meta.json"
PROMPT_NAME = "prompt.wav"


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _voice_dir(voice_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]{4,64}", voice_id or ""):
        raise ValueError("音色 ID 格式不正確")
    return config.VOICES_DIR / voice_id


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def _probe_duration(wav: Path) -> float:
    """直接從 wav 檔頭量長度，meta.json 壞掉時用得上。"""
    try:
        import soundfile as sf

        info = sf.info(str(wav))
        if info.samplerate:
            return round(info.frames / info.samplerate, 2)
    except Exception:  # noqa: BLE001
        pass
    return 0.0


def _rebuild_meta(d: Path, reason: str) -> dict:
    """meta.json 沒了或壞了，但 prompt.wav 還在：靠音檔本身把音色救回來。

    寧可給一個「未命名音色」讓使用者改名，也不要整個音色從清單消失。
    """
    wav = d / PROMPT_NAME
    try:
        created = (
            datetime.fromtimestamp(wav.stat().st_mtime, timezone.utc)
            .astimezone()
            .isoformat(timespec="seconds")
        )
    except OSError:
        created = _now()
    return {
        "id": d.name,
        "name": f"未命名音色 {d.name[:6]}",
        "transcription": "",
        "note": f"自動修復：{reason}；名稱與逐字稿需要重新填寫。",
        "duration": _probe_duration(wav),
        "source_filename": "",
        "created_at": created,
        "updated_at": _now(),
        "recovered": True,
    }


def _read_meta(d: Path) -> Optional[dict]:
    """讀一個音色資料夾。只要 prompt.wav 還在，就一定回得了東西。"""
    f = d / META_NAME
    has_audio = (d / PROMPT_NAME).exists()
    meta: Optional[dict] = None
    reason = ""

    if f.exists():
        try:
            loaded = json.loads(f.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                meta = loaded
            else:
                reason = "meta.json 內容不是物件"
        except Exception as exc:  # noqa: BLE001
            reason = f"meta.json 解析失敗（{exc.__class__.__name__}）"
    else:
        reason = "meta.json 遺失"

    if meta is None:
        if not has_audio:
            log.warning("音色 %s：%s 且沒有 %s，跳過", d.name, reason, PROMPT_NAME)
            return None
        log.warning("音色 %s：%s，改用參考音檔重建", d.name, reason)
        meta = _rebuild_meta(d, reason)
        try:  # 順手寫回，下次啟動就不用再修一遍
            with _lock:
                _write_meta(d, meta)
        except OSError as exc:
            log.warning("音色 %s 修復後寫回失敗：%s", d.name, exc)

    meta["id"] = d.name
    meta["has_audio"] = has_audio
    # 舊版或手改過的 meta 可能缺欄位，補齊避免前端 render 爆掉
    if not str(meta.get("name", "")).strip():
        meta["name"] = f"未命名音色 {d.name[:6]}"
    meta.setdefault("transcription", "")
    meta.setdefault("note", "")
    meta.setdefault("language", "")
    if not isinstance(meta.get("duration"), (int, float)):
        meta["duration"] = _probe_duration(d / PROMPT_NAME) if has_audio else 0.0
    meta.setdefault("created_at", _now())
    return meta


def _write_meta(d: Path, meta: dict) -> None:
    (d / META_NAME).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def list_voices() -> list[dict]:
    out: list[dict] = []
    if not config.VOICES_DIR.is_dir():
        log.warning("音色庫資料夾不存在，重建：%s", config.VOICES_DIR)
        config.VOICES_DIR.mkdir(parents=True, exist_ok=True)
        return out
    for d in sorted(config.VOICES_DIR.iterdir()):
        if not d.is_dir() or d.name.startswith("."):
            continue
        try:
            meta = _read_meta(d)
        except Exception as exc:  # noqa: BLE001
            log.warning("音色 %s 讀取失敗：%s", d.name, exc)
            continue
        if meta:
            out.append(meta)
    out.sort(key=lambda m: m.get("created_at", ""), reverse=True)
    return out


def repair_library() -> dict:
    """啟動時掃一次音色庫：救回壞掉的 meta，回報缺參考音的音色。"""
    items = list_voices()  # _read_meta 會順手把能修的修好
    recovered = [m["id"] for m in items if m.get("recovered")]
    missing_audio = [m["id"] for m in items if not m.get("has_audio")]
    return {
        "total": len(items),
        "recovered": recovered,
        "missing_audio": missing_audio,
    }


def get_voice(voice_id: str) -> Optional[dict]:
    d = _voice_dir(voice_id)
    return _read_meta(d) if d.is_dir() else None


def prompt_path(voice_id: str) -> Path:
    return _voice_dir(voice_id) / PROMPT_NAME


def create_voice(
    name: str,
    src_audio: Path,
    transcription: str = "",
    note: str = "",
) -> dict:
    """src_audio 為暫存的上傳檔；會轉成 16k wav 後存入音色庫。"""
    from .audio_utils import prepare_prompt_wav

    name = (name or "").strip() or "未命名音色"
    voice_id = new_id()
    d = _voice_dir(voice_id)
    d.mkdir(parents=True, exist_ok=False)

    try:
        duration = prepare_prompt_wav(src_audio, d / PROMPT_NAME)
        if duration < config.MIN_PROMPT_SECONDS:
            raise ValueError(
                f"參考音太短（{duration} 秒），建議 {config.MIN_PROMPT_SECONDS}–"
                f"{config.MAX_PROMPT_SECONDS} 秒的清晰單人語音"
            )
        if duration > config.MAX_PROMPT_SECONDS:
            raise ValueError(
                f"參考音太長（{duration} 秒），請裁剪到 {config.MAX_PROMPT_SECONDS} 秒以內"
            )
        meta = {
            "id": voice_id,
            "name": name,
            "transcription": (transcription or "").strip(),
            "note": (note or "").strip(),
            "duration": duration,
            "source_filename": Path(src_audio).name,
            "created_at": _now(),
            "updated_at": _now(),
        }
        with _lock:
            _write_meta(d, meta)
        meta["has_audio"] = True
        return meta
    except Exception:
        shutil.rmtree(d, ignore_errors=True)
        raise


def update_voice(voice_id: str, **fields) -> dict:
    d = _voice_dir(voice_id)
    meta = _read_meta(d)
    if not meta:
        raise KeyError(voice_id)
    for key in ("name", "transcription", "note", "language"):
        if key in fields and fields[key] is not None:
            meta[key] = str(fields[key]).strip()
    # 使用者親手改過名稱或逐字稿，就不再是自動修復的半成品
    if meta.pop("recovered", None) and ("name" in fields or "transcription" in fields):
        meta["note"] = ""
    meta["updated_at"] = _now()
    with _lock:
        _write_meta(d, meta)
    return meta


def delete_voice(voice_id: str) -> bool:
    d = _voice_dir(voice_id)
    if not d.is_dir():
        return False
    shutil.rmtree(d, ignore_errors=True)
    return True
