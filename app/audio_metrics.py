"""合成音檔的客觀訊號指標 —— 不需要任何遠端服務，純 numpy 算得出來。

CER 量的是「有沒有唸對字」，主觀投票量的是「好不好聽」，但兩者都看不到一些
很具體的毛病。這裡補上那一塊：

    靜音比例太高      -> 引擎在段落之間塞了大量空白，實際語音沒那麼長
    頭尾靜音很長      -> 即時應用會感覺延遲，而且 RTF 會被灌水（分母變大）
    削波比例高        -> 輸出爆音，音量沒收好
    有效頻寬遠低於奈奎斯特 -> **宣稱 48kHz，實際內容只到 12kHz**
    語速異常          -> 唸太快（吃字）或唸太慢（拖）

最後兩個特別重要。取樣率是引擎自己宣告的（voxcpm2 說 48000），但宣告不等於
內容真的有那個頻寬 —— 從 16k 模型升採樣上去也會「是」48kHz。用 FFT 量 99%
能量落在哪裡，才知道實際解析度。

指標要在**音量正規化之前**算，不然 peak/RMS 全部會被壓成同一個值，
比較就沒意義了。呼叫點在 engines/base.py 的 synthesize()。
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

log = logging.getLogger("tts.metrics")

# 判定「這一幀算不算靜音」的門檻：比整段最大幀能量低這麼多 dB 就算靜音
SILENCE_REL_DB = 40.0
# 靜音門檻的絕對下限，避免整段都很小聲時把全部判成有聲
SILENCE_ABS_DBFS = -60.0
FRAME_MS = 20.0


def _dbfs(x: float) -> Optional[float]:
    return round(20.0 * float(np.log10(max(x, 1e-10))), 2) if x > 0 else None


def _frame_rms(wav: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    """切成不重疊的短幀，回每幀的 RMS。"""
    hop = max(1, int(sr * FRAME_MS / 1000.0))
    n = len(wav) // hop
    if n < 1:
        return np.asarray([float(np.sqrt(np.mean(wav ** 2)))] if wav.size else [0.0]), hop
    frames = wav[: n * hop].reshape(n, hop)
    return np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1)), hop


def _effective_bandwidth(wav: np.ndarray, sr: int, energy_ratio: float = 0.99) -> Optional[float]:
    """能量累積到 energy_ratio 的頻率（Hz）—— 內容實際佔到多寬。

    對「16k 模型升採樣成 48k 輸出」這種情況，這個值會停在 8kHz 附近，
    而不是接近 24kHz 的奈奎斯特頻率。
    """
    if wav.size < 1024 or sr <= 0:
        return None
    # 取中段最響的一秒來算，避開頭尾靜音
    n = min(len(wav), sr)
    if len(wav) > n:
        rms, hop = _frame_rms(wav, sr)
        if rms.size:
            center = int(np.argmax(rms)) * hop
            start = max(0, min(center - n // 2, len(wav) - n))
        else:
            start = (len(wav) - n) // 2
    else:
        start = 0
    seg = wav[start : start + n].astype(np.float64)
    if not np.any(seg):
        return None

    seg = seg * np.hanning(len(seg))
    spec = np.abs(np.fft.rfft(seg)) ** 2
    total = float(spec.sum())
    if total <= 0:
        return None
    cumulative = np.cumsum(spec) / total
    idx = int(np.searchsorted(cumulative, energy_ratio))
    idx = min(idx, len(spec) - 1)
    freqs = np.fft.rfftfreq(len(seg), d=1.0 / sr)
    return round(float(freqs[idx]), 1)


def analyze(wav: np.ndarray, sr: int, char_count: int = 0) -> dict:
    """算一整組客觀指標。任何一項失敗都不該讓評測跟著失敗，所以整段包起來。"""
    try:
        return _analyze(wav, sr, char_count)
    except Exception as exc:  # noqa: BLE001
        log.warning("音訊指標計算失敗：%s", exc)
        return {"error": f"{type(exc).__name__}: {exc}"}


def _analyze(wav: np.ndarray, sr: int, char_count: int) -> dict:
    wav = np.asarray(wav, dtype=np.float32).ravel()
    if wav.size == 0 or sr <= 0:
        return {"error": "音檔是空的"}

    duration = len(wav) / sr
    peak = float(np.max(np.abs(wav)))
    rms = float(np.sqrt(np.mean(wav.astype(np.float64) ** 2)))

    # --- 靜音分析 ---
    frame_rms, hop = _frame_rms(wav, sr)
    frame_db = 20.0 * np.log10(np.maximum(frame_rms, 1e-10))
    ceiling = float(np.max(frame_db)) if frame_db.size else -100.0
    threshold = max(ceiling - SILENCE_REL_DB, SILENCE_ABS_DBFS)
    voiced = frame_db > threshold

    frame_seconds = hop / sr
    voiced_seconds = float(np.sum(voiced)) * frame_seconds
    silence_ratio = 1.0 - (float(np.sum(voiced)) / len(voiced)) if len(voiced) else 0.0

    lead = tail = 0.0
    if np.any(voiced):
        first = int(np.argmax(voiced))
        last = len(voiced) - 1 - int(np.argmax(voiced[::-1]))
        lead = first * frame_seconds
        tail = (len(voiced) - 1 - last) * frame_seconds

    # --- 削波 ---
    clipped = int(np.sum(np.abs(wav) >= 0.999))

    bandwidth = _effective_bandwidth(wav, sr)
    nyquist = sr / 2.0

    return {
        "duration_seconds": round(duration, 3),
        # 音量（一定要在正規化之前算，不然全部都會是同一個峰值）
        "peak_dbfs": _dbfs(peak),
        "rms_dbfs": _dbfs(rms),
        "crest_db": round(_dbfs(peak) - _dbfs(rms), 2)
                    if peak > 0 and rms > 0 else None,
        "dc_offset": round(float(np.mean(wav)), 5),
        "clip_ratio": round(clipped / len(wav), 6),
        "clipped_samples": clipped,
        # 靜音
        "voiced_seconds": round(voiced_seconds, 3),
        "silence_ratio": round(silence_ratio, 4),
        "lead_silence_seconds": round(lead, 3),
        "tail_silence_seconds": round(tail, 3),
        # 頻寬：宣稱的取樣率 vs 內容實際佔到的頻寬
        "sample_rate": sr,
        "nyquist_hz": round(nyquist, 1),
        "effective_bandwidth_hz": bandwidth,
        "bandwidth_ratio": round(bandwidth / nyquist, 3)
                           if bandwidth and nyquist else None,
        # 語速：用「有聲時間」算才準，把段落之間的空白排除掉
        "chars_per_voiced_second": round(char_count / voiced_seconds, 2)
                                    if char_count and voiced_seconds > 0.05 else None,
    }


# ------------------------------------------------------------------ 判讀
# 給 UI 用的警訊規則。門檻取得寬鬆，只標「明顯不對勁」的，避免滿江紅。
def warnings_for(metrics: dict) -> list[str]:
    if not metrics or metrics.get("error"):
        return []
    out: list[str] = []

    ratio = metrics.get("bandwidth_ratio")
    bw = metrics.get("effective_bandwidth_hz")
    if ratio is not None and ratio < 0.5 and bw:
        out.append(
            f"實際頻寬只到 {bw / 1000:.1f} kHz，遠低於宣稱取樣率的奈奎斯特頻率"
            f"（{metrics['nyquist_hz'] / 1000:.1f} kHz）—— 多半是低取樣率模型升採樣上來的"
        )

    silence = metrics.get("silence_ratio")
    if silence is not None and silence > 0.35:
        out.append(f"靜音佔了 {silence:.0%}，音檔長度會灌水，RTF 要打折看")

    lead = metrics.get("lead_silence_seconds") or 0
    if lead > 0.4:
        out.append(f"開頭有 {lead:.2f} 秒靜音，即時應用會感覺延遲")

    tail = metrics.get("tail_silence_seconds") or 0
    if tail > 0.6:
        out.append(f"結尾拖了 {tail:.2f} 秒靜音")

    clip = metrics.get("clip_ratio")
    if clip is not None and clip > 0.0005:
        out.append(f"有 {clip:.2%} 的取樣點削波，輸出音量沒收好")

    dc = metrics.get("dc_offset")
    if dc is not None and abs(dc) > 0.01:
        out.append(f"直流偏移 {dc:+.3f}，波形沒有以零為中心")

    rms_db = metrics.get("rms_dbfs")
    if rms_db is not None and rms_db < -35:
        out.append(f"整體音量偏小（RMS {rms_db} dBFS）")

    cps = metrics.get("chars_per_voiced_second")
    if cps is not None:
        if cps > 9.0:
            out.append(f"語速偏快（每秒 {cps} 字），注意有沒有吃字")
        elif cps < 2.5:
            out.append(f"語速偏慢（每秒 {cps} 字）")
    return out
