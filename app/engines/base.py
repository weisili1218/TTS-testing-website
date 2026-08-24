"""TTS 引擎轉接層的基底類別與共用工具。

要加一個新引擎，只要在 app/engines/ 底下開一個檔案，繼承 TTSAdapter，
實作 `request_segment()`，然後在 app/engines/__init__.py 註冊它。UI 的
引擎選項、排行榜欄位、盲測配對都會自動跟著出現，不用改其他地方。

    class MyEngine(TTSAdapter):
        key = "myengine"
        label = "My Engine"

        @property
        def enabled(self):
            return bool(config.MY_BASE_URL)

        def request_segment(self, segment, opts):
            r = httpx.post(url, json={"text": segment})
            return decode_audio_bytes(r.content)      # -> (float32 波形, 取樣率)

為什麼共用流程要寫在基底類別
----------------------------
切段規則、逐段送出、計時起訖點、拼接與音量正規化全部由 synthesize() 統一處理，
子類別只負責「送一段、拿回音訊」。這不是為了少寫幾行 —— 而是評測的正確性要求：
兩個引擎如果用不同的切法或不同的計時起點，測出來的秒數就不能拿來比。
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Callable, Optional

import numpy as np

from .. import config
from .. import audio_metrics
from ..audio_utils import change_speed, concat_with_gap, peak_normalize
from ..text_utils import ensure_terminated, split_segments, visible_length

log = logging.getLogger("tts.engine")

Progress = Callable[[str, float], None]


class RemoteError(RuntimeError):
    """遠端引擎回了錯誤，或根本連不上。"""


# GPU 後端的服務常常把底層例外原樣丟回來，對使用者沒意義。
# 這裡把幾種「知道怎麼修」的狀況翻譯成可行動的說明。
_TRANSLATED_MARK = "（原始訊息："

_FATAL_HINTS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("cufft", "cublas", "cuda error", "device-side assert"),
     "服務端的 GPU 狀態異常，通常要重啟那台機器上的服務才會恢復"),
    (("out of memory", "oom", "cuda_error_out_of_memory"),
     "服務端的 GPU 記憶體不足，請縮短文字或重啟該服務"),
    (("reference audio not found", "no such file"),
     "服務端找不到參考音檔。若它只吃自己主機上的路徑，就得先把音檔放過去"),
)


def friendly_error(raw: str) -> str:
    """把服務端的例外字串翻譯成看得懂、知道下一步做什麼的說明。

    翻譯過的訊息會帶 _TRANSLATED_MARK，遇到就原樣放行 —— 這個函式可能在
    請求層與健康檢查層各經過一次，不擋一下會被包好幾層。
    """
    if _TRANSLATED_MARK in (raw or ""):
        return raw.strip()
    low = (raw or "").lower()
    for needles, hint in _FATAL_HINTS:
        if any(n in low for n in needles):
            return f"{hint}{_TRANSLATED_MARK}{raw.strip()[:160]}）"
    return raw.strip()[:300] or "未知原因"


def describe_http_error(status: int, body: bytes, endpoint: str = "") -> str:
    """把 HTTP 錯誤回應整理成一句話，盡量把伺服器講的原因挖出來。"""
    text = ""
    try:
        parsed = json.loads(body.decode("utf-8", "replace"))
        if isinstance(parsed, dict):
            err = parsed.get("error")
            if isinstance(err, dict):
                text = str(err.get("message") or err)
            else:
                text = str(parsed.get("detail") or parsed.get("message") or err or parsed)
    except Exception:  # noqa: BLE001
        text = body.decode("utf-8", "replace")
    text = (text or "").strip()[:400]

    hint = ""
    if status in (401, 403):
        hint = "（需要 API key）"
    elif status == 404:
        hint = f"（找不到端點：{endpoint}）" if endpoint else "（找不到端點）"
    elif status == 422:
        hint = "（送出的欄位對不上）"
    return f"HTTP {status}{hint}：{friendly_error(text)}"


class TTSAdapter:
    """一個參賽引擎。子類別最少只要覆寫 key / label / enabled / request_segment。"""

    key: str = ""
    label: str = ""

    #: 是否需要本機音色庫的參考音（zero-shot 音色複製）。
    #: True 的話，送出評測前會強制要求選一個有 prompt.wav 與逐字稿的音色。
    needs_voice: bool = False

    #: 服務端自己會處理 speed 參數嗎？False 的話由本機做變速不變調。
    remote_speed: bool = True

    def __init__(self) -> None:
        # 同一個引擎一次只送一段，避免壓垮遠端；也讓計時不受自身併發干擾
        self._lock = threading.Lock()

    # ------------------------------------------------------------ 基本資訊
    @property
    def enabled(self) -> bool:
        """設定齊全、應該出現在 UI 上嗎？"""
        return True

    @property
    def endpoint(self) -> str:
        return ""

    def voice_label(self, opts: dict) -> str:
        """這次評測實際用的是什麼音色 —— 比較結果能不能談音色，看這欄。"""
        return "引擎預設音色"

    def uses_custom_reference(self, opts: dict) -> bool:
        """有沒有用到我們指定的參考音。全部引擎都是 True 時，音色才可比。"""
        return False

    def validate(self, opts: dict) -> None:
        """送出前的檢查。有問題就丟 ValueError，訊息會直接顯示給使用者。"""

    def describe(self) -> dict:
        """給 /api/status 用的靜態描述。"""
        return {
            "key": self.key,
            "label": self.label,
            "enabled": self.enabled,
            "endpoint": self.endpoint,
            "needs_voice": self.needs_voice,
        }

    # ------------------------------------------------------------ 暖機
    def warmup(self) -> dict:
        """把模型載進 GPU。有專屬 warmup 端點的引擎應該覆寫這個。

        預設實作就是合成一句短句 —— 對沒有 warmup 端點的引擎，這是唯一能
        強迫它把模型載起來的方法。不先做，第一筆速度量到的是「載模型 + 合成」。
        """
        return self.smoke_test()

    # ------------------------------------------------------------ 健康檢查
    def health(self, deep: bool = False) -> dict:
        """回 {'ok': bool, 'error': str|None, ...}。

        deep=True 應該實際合成一句短句。淺層檢查（打健康端點）不一定可信 ——
        實測過有服務在 GPU 掛掉時，健康端點照樣回報一切正常。
        """
        if not self.enabled:
            return {"ok": False, "error": f"{self.label} 未設定，停用中"}
        if deep:
            return self.smoke_test()
        return {"ok": True, "error": None}

    def smoke_test(self, text: str = "測試。") -> dict:
        """實際合成一句短句，確認引擎真的能用。"""
        if not self.enabled:
            return {"ok": False, "error": f"{self.label} 未設定，停用中"}
        started = time.time()
        try:
            _audio, sr, info = self.synthesize(text, {"speed": 1.0, "normalize_volume": False},
                                               lambda *_a, **_k: None)
            return {"ok": True, "error": None, "sample_rate": sr,
                    "elapsed_seconds": round(time.time() - started, 2),
                    "duration_seconds": info.get("duration_seconds")}
        except Exception as exc:  # noqa: BLE001
            msg = friendly_error(str(exc))
            log.warning("%s 冒煙測試失敗：%s", self.label, msg)
            return {"ok": False, "error": msg,
                    "elapsed_seconds": round(time.time() - started, 2)}

    # ------------------------------------------------------------ 合成
    def prepare(self, opts: dict) -> None:
        """合成前的準備（例如把參考音編成 base64）。在計時開始前呼叫。"""

    def request_segment(self, segment: str, opts: dict) -> tuple[np.ndarray, int]:
        """送出一段文字，回傳 (float32 單聲道波形, 取樣率)。子類別必須實作。"""
        raise NotImplementedError

    def segments_for(self, text: str) -> list[str]:
        """切段。所有引擎共用同一套規則，切法一致速度才可比。"""
        content = ensure_terminated(text)
        if not config.SPLIT_SEGMENTS:
            return [content]
        return split_segments(content, config.MAX_CHARS_PER_SEGMENT) or [content]

    def synthesize(
        self,
        text: str,
        opts: dict,
        progress: Optional[Progress] = None,
    ) -> tuple[np.ndarray, int, dict[str, Any]]:
        """共用合成流程：切段 → 逐段送出 → 計時 → 拼接 → 正規化。

        計時只涵蓋「第一段送出」到「最後一段收完」，不含準備參考音與等鎖的時間，
        這樣不同引擎的秒數才是在量同一件事。
        """
        if not text.strip():
            raise ValueError("要合成的文字不可為空")

        self.validate(opts)
        segments = self.segments_for(text)
        speed = float(opts.get("speed", 1.0))

        if progress:
            progress("準備中…", 0.02)
        self.prepare(opts)

        with self._lock:
            started = time.time()
            first_chunk_at: Optional[float] = None
            chunks: list[np.ndarray] = []
            sample_rate = 0

            for i, seg in enumerate(segments):
                if progress:
                    progress(f"送出第 {i + 1}/{len(segments)} 段…", 0.05 + 0.9 * (i / len(segments)))
                wav, sr = self.request_segment(seg, opts)
                if chunks and sr != sample_rate:
                    raise RemoteError(f"各段取樣率不一致（{sample_rate} vs {sr}），無法拼接")
                sample_rate = sr
                chunks.append(wav)
                if first_chunk_at is None:
                    first_chunk_at = time.time()

            elapsed = round(time.time() - started, 2)
            first_chunk = round(first_chunk_at - started, 2) if first_chunk_at else None

        audio = concat_with_gap(chunks, sample_rate, config.SEGMENT_GAP_SECONDS)

        # 服務端不吃 speed 的話，本機做變速不變調（不計入合成耗時，那不是引擎的成本）
        if not self.remote_speed and abs(speed - 1.0) > 1e-3:
            if progress:
                progress("調整語速…", 0.96)
            audio = change_speed(audio, speed)

        # 客觀指標一定要在正規化**之前**算 —— 正規化會把每個引擎的峰值都拉到
        # 同一個數字，peak / RMS / 削波比例全部失去比較意義。
        metrics = audio_metrics.analyze(audio, sample_rate, visible_length(text))

        if opts.get("normalize_volume", True):
            audio = peak_normalize(audio)

        if progress:
            progress("完成", 1.0)
        return audio, sample_rate, {
            "audio_metrics": metrics,
            "audio_warnings": audio_metrics.warnings_for(metrics),
            "duration_seconds": round(len(audio) / sample_rate, 2) if sample_rate else 0.0,
            "elapsed_seconds": elapsed,
            "first_chunk_seconds": first_chunk,
            "segments": len(segments),
            "requests": len(segments),
            "processed_text": "\n".join(segments),
            "sample_rate": sample_rate,
            "endpoint": self.endpoint,
        }
