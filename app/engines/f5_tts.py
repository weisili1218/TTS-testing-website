"""F5-TTS（Speech Service）轉接器。

    POST {F5_BASE_URL}/speech-tts     JSON，audio_base64 = PCM 16-bit（無檔頭）
    GET  {F5_BASE_URL}/tts-health     健康檢查與可用引擎清單

已驗證過的兩件事
-----------------
1. ref_audio 只吃「F5 主機本機的檔案路徑」。純 base64 與 data URI 都會被回
   "reference audio not found"，所以本機音色庫送不過去。要用自訂參考音，
   得先把檔案放到那台機器上，再把路徑填進 F5_REF_AUDIO。
2. /tts-health 不可信。這個服務的 GPU 一旦掉進 cuFFT / cuBLAS 錯誤，之後每一次
   合成都會失敗且不會自己恢復，但健康檢查照樣回 available: true。
   所以 health(deep=True) 一定要實際合成一句才算數。
"""

from __future__ import annotations

import base64
import logging
import time
from typing import Optional

import httpx
import numpy as np

from .. import config
from .base import RemoteError, TTSAdapter, friendly_error

log = logging.getLogger("tts.f5")


class F5Adapter(TTSAdapter):
    key = "f5"
    label = "F5-TTS"
    needs_voice = False        # 讀不到本機檔案，用不了音色庫

    @property
    def enabled(self) -> bool:
        return bool(config.F5_BASE_URL)

    @property
    def endpoint(self) -> str:
        return f"{config.F5_BASE_URL}{config.F5_SPEECH_PATH}"

    @property
    def health_endpoint(self) -> str:
        return f"{config.F5_BASE_URL}{config.F5_HEALTH_PATH}"

    # ------------------------------------------------------------ 音色
    def _ref_audio(self, opts: dict) -> str:
        return (opts.get("f5_ref_audio") or config.F5_REF_AUDIO or "").strip()

    def uses_custom_reference(self, opts: dict) -> bool:
        return bool(self._ref_audio(opts))

    def voice_label(self, opts: dict) -> str:
        ref = self._ref_audio(opts)
        return f"F5 自訂參考音（{ref}）" if ref else "F5 內建預設音色"

    def describe(self) -> dict:
        return {
            **super().describe(),
            "voice_source": (
                f"自訂參考音：{config.F5_REF_AUDIO}"
                if config.F5_REF_AUDIO else "F5 內建預設音色"
            ),
            "uses_custom_ref": bool(config.F5_REF_AUDIO),
            "note": (
                "" if config.F5_REF_AUDIO else
                "F5 的 ref_audio 只接受它自己主機上的檔案路徑，"
                "本機音色庫送不過去。設定 F5_REF_AUDIO 就能改用指定的參考音。"
            ),
        }

    # ------------------------------------------------------------ 健康檢查
    def _shallow(self) -> dict:
        with httpx.Client(timeout=config.F5_CONNECT_TIMEOUT) as cli:
            r = cli.get(self.health_endpoint)
            r.raise_for_status()
            return r.json()

    def health(self, deep: bool = False) -> dict:
        if not self.enabled:
            return {"ok": False, "error": "未設定 F5_BASE_URL，停用中"}
        info: dict = {"endpoint": self.endpoint}
        try:
            raw = self._shallow()
            info.update(
                model=raw.get("model"),
                sample_rate=raw.get("sample_rate"),
                default_engine=raw.get("default_engine"),
                engines=[
                    {"name": e.get("name"), "label": e.get("label"),
                     "available": e.get("available")}
                    for e in raw.get("engines", [])
                ],
                ok=True, error=None,
            )
        except Exception as exc:  # noqa: BLE001
            return {**info, "ok": False, "engines": [],
                    "error": f"{exc.__class__.__name__}: {exc}"}

        if deep:
            # 淺層檢查會騙人（見檔頭），以實際合成結果為準
            smoke = self.smoke_test()
            info["smoke"] = smoke
            info["ok"] = smoke["ok"]
            info["error"] = smoke["error"]
        return info

    # ------------------------------------------------------------ 合成
    @staticmethod
    def _decode_pcm16(b64: str) -> np.ndarray:
        """audio_base64 是無檔頭的 PCM 16-bit little-endian，轉成 float32 -1..1。"""
        raw = base64.b64decode(b64)
        if len(raw) % 2:
            raw = raw[:-1]
        return (np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0).copy()

    def _payload(self, segment: str, opts: dict) -> dict:
        payload: dict = {
            "text": segment,
            "speed": float(opts.get("speed", 1.0)),
            "language": (opts.get("f5_language") or config.F5_LANGUAGE),
        }
        engine = (opts.get("f5_engine") or config.F5_ENGINE or "").strip()
        if engine:
            payload["engine"] = engine
        ref = self._ref_audio(opts)
        if ref:
            payload["ref_audio"] = ref
        ref_text = (opts.get("f5_ref_text") or config.F5_REF_TEXT or "").strip()
        if ref_text:
            payload["ref_text"] = ref_text
        return payload

    def request_segment(self, segment: str, opts: dict) -> tuple[np.ndarray, int]:
        timeout = httpx.Timeout(config.F5_TIMEOUT, connect=config.F5_CONNECT_TIMEOUT)
        last: Optional[Exception] = None

        for attempt in range(config.F5_MAX_RETRIES + 1):
            try:
                with httpx.Client(timeout=timeout) as cli:
                    r = cli.post(self.endpoint, json=self._payload(segment, opts))
                r.raise_for_status()
                data = r.json()
                if not data.get("success"):
                    # 服務自己回報的錯誤（GPU 壞掉、參考音路徑不存在…）重試也沒用
                    raise ValueError(
                        "F5-TTS 回報失敗：" + friendly_error(data.get("error") or "")
                    )
                b64 = data.get("audio_base64")
                if not b64:
                    raise ValueError("F5-TTS 沒有回傳音訊")
                return (self._decode_pcm16(b64),
                        int(data.get("sample_rate") or config.F5_SAMPLE_RATE))
            except ValueError:
                raise
            except Exception as exc:  # noqa: BLE001
                last = exc
                if attempt < config.F5_MAX_RETRIES:
                    log.warning("F5-TTS 第 %d 次失敗，重試：%s", attempt + 1, exc)
                    time.sleep(1.0)
        raise RemoteError(f"F5-TTS 請求失敗：{last}")
