"""遠端 Whisper ASR 用戶端 —— 客觀評分用。

評「品質」最麻煩的是主觀，但至少「有沒有把字唸對」可以自動量：
把引擎合成出來的音檔丟回 ASR 轉成文字，跟原始輸入比字錯誤率（CER）。

    POST {ASR_BASE_URL}/speech-asr?language=zh
    multipart/form-data  file=<wav>     ->  {"success": true, "text": "..."}

CER 要怎麼讀
------------
這是「相對」指標，不是絕對品質分數：ASR 自己就會聽錯，所以即使是真人錄音
CER 也不會是 0。要判斷一個引擎的絕對水準，先跑一次 baseline()（拿音色庫裡
的真人參考音去 ASR），得到這個 ASR 的誤差底線，再拿引擎的 CER 去對照。
兩個引擎跑同一段文字、同一個 ASR，比大小才是公平的。
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
import httpx

from . import config
from .text_utils import char_error_rate

log = logging.getLogger("tts.asr")


def speech_url() -> str:
    return f"{config.ASR_BASE_URL}{config.ASR_SPEECH_PATH}"


def health_url() -> str:
    return f"{config.ASR_BASE_URL}{config.ASR_HEALTH_PATH}"


class ASRUnavailable(RuntimeError):
    """ASR 服務沒設定或連不上。評分失敗不該讓整個評測跟著失敗。"""


def check_remote() -> dict:
    if not config.ASR_ENABLED:
        raise ASRUnavailable("未設定 ASR_BASE_URL，客觀評分停用中")
    with httpx.Client(timeout=config.ASR_CONNECT_TIMEOUT) as cli:
        r = cli.get(health_url())
        r.raise_for_status()
        info = r.json()
    return {
        "ok": True,
        "base_url": config.ASR_BASE_URL,
        "endpoint": speech_url(),
        "model": info.get("model"),
    }


def status() -> dict:
    """狀態列用，連不上就回 error 字串，不丟例外。"""
    if not config.ASR_ENABLED:
        return {"enabled": False, "ok": False, "error": None}
    try:
        return {"enabled": True, **check_remote(), "error": None}
    except Exception as exc:  # noqa: BLE001
        return {
            "enabled": True,
            "ok": False,
            "base_url": config.ASR_BASE_URL,
            "endpoint": speech_url(),
            "error": f"{exc.__class__.__name__}: {exc}",
        }


def transcribe(wav_path: str | Path) -> dict:
    """把音檔送去 ASR，回傳 {'text': ..., 'language': ..., 'elapsed_seconds': ...}。"""
    if not config.ASR_ENABLED:
        raise ASRUnavailable("未設定 ASR_BASE_URL，客觀評分停用中")

    path = Path(wav_path)
    if not path.exists():
        raise ASRUnavailable(f"找不到要辨識的音檔：{path}")

    started = time.time()
    timeout = httpx.Timeout(config.ASR_TIMEOUT, connect=config.ASR_CONNECT_TIMEOUT)
    with httpx.Client(timeout=timeout) as cli:
        with path.open("rb") as fh:
            r = cli.post(
                speech_url(),
                params={"language": config.ASR_LANGUAGE} if config.ASR_LANGUAGE else None,
                files={"file": (path.name, fh, "audio/wav")},
            )
    r.raise_for_status()
    data = r.json()
    if not data.get("success", True):
        raise ASRUnavailable(f"ASR 回報失敗：{data.get('error') or '未知原因'}")

    return {
        "text": (data.get("text") or "").strip(),
        "language": data.get("language"),
        "elapsed_seconds": round(time.time() - started, 2),
    }


def score(reference_text: str, wav_path: str | Path) -> dict:
    """辨識 + 算 CER。這是評測流程真正會呼叫的入口。

    永遠回傳 dict、永遠不丟例外：ASR 掛掉只代表「這筆沒有客觀分數」，
    合成結果本身還是要留著給人聽。
    """
    try:
        asr = transcribe(wav_path)
    except Exception as exc:  # noqa: BLE001
        log.warning("ASR 評分失敗：%s", exc)
        return {
            "ok": False,
            "error": f"{exc.__class__.__name__}: {exc}",
            "asr_text": None,
            "cer": None,
        }

    metrics = char_error_rate(reference_text, asr["text"]) or {}
    return {
        "ok": True,
        "error": None,
        "asr_text": asr["text"],
        "asr_seconds": asr["elapsed_seconds"],
        "cer": metrics.get("cer"),
        "edits": metrics.get("edits"),
        "ref_chars": metrics.get("ref_chars"),
        "hyp_chars": metrics.get("hyp_chars"),
    }


def baseline(wav_path: str | Path, reference_text: str) -> dict:
    """ASR 誤差底線：拿真人參考音去辨識，看看這個 ASR 本身有多準。

    用來回答「引擎的 CER 0.12 到底算好還壞」——如果真人錄音也有 0.10，
    那 0.12 其實很接近上限了。
    """
    result = score(reference_text, wav_path)
    result["kind"] = "baseline"
    return result
