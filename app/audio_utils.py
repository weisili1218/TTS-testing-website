"""音訊處理小工具：格式轉換、長度檢查、語速調整、拼接。"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf


class AudioError(RuntimeError):
    pass


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def load_mono(path: str | Path, target_sr: int) -> np.ndarray:
    """把任意格式音檔讀成單聲道 float32 numpy array。

    優先用 librosa（有裝的話），失敗才退回 ffmpeg。
    """
    path = str(path)
    try:
        import librosa

        wav, _ = librosa.load(path, sr=target_sr, mono=True)
        return wav.astype(np.float32)
    except Exception as exc:  # noqa: BLE001
        if not _ffmpeg_available():
            raise AudioError(
                f"無法讀取音檔 {Path(path).name}：{exc}。"
                "請安裝 ffmpeg（macOS：brew install ffmpeg）後再試。"
            ) from exc

    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "conv.wav"
        cmd = [
            "ffmpeg", "-nostdin", "-loglevel", "error", "-y",
            "-i", path, "-ac", "1", "-ar", str(target_sr), str(out),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 or not out.exists():
            raise AudioError(f"ffmpeg 轉檔失敗：{proc.stderr.strip()[:400]}")
        wav, _ = sf.read(str(out), dtype="float32", always_2d=False)
    return np.asarray(wav, dtype=np.float32)


def save_wav(path: str | Path, wav: np.ndarray, sr: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(wav, dtype=np.float32), sr, subtype="PCM_16")


def trim_silence(wav: np.ndarray, top_db: float = 35.0) -> np.ndarray:
    """去掉頭尾靜音；失敗就原樣回傳。"""
    try:
        import librosa

        trimmed, _ = librosa.effects.trim(wav, top_db=top_db)
        if trimmed.size > 0:
            return trimmed.astype(np.float32)
    except Exception:  # noqa: BLE001
        pass
    return wav


def peak_normalize(wav: np.ndarray, peak: float = 0.95) -> np.ndarray:
    m = float(np.max(np.abs(wav))) if wav.size else 0.0
    if m < 1e-6:
        return wav
    return (wav / m * peak).astype(np.float32)


def change_speed(wav: np.ndarray, speed: float) -> np.ndarray:
    """變速不變調（phase vocoder）。speed > 1 = 講快一點。"""
    if abs(speed - 1.0) < 1e-3 or wav.size == 0:
        return wav
    try:
        import librosa

        return librosa.effects.time_stretch(wav, rate=float(speed)).astype(np.float32)
    except Exception:  # noqa: BLE001
        return wav


def concat_with_gap(chunks: list[np.ndarray], sr: int, gap_seconds: float) -> np.ndarray:
    chunks = [c for c in chunks if c is not None and c.size > 0]
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    if len(chunks) == 1:
        return chunks[0]
    gap = np.zeros(int(sr * max(gap_seconds, 0.0)), dtype=np.float32)
    out: list[np.ndarray] = []
    for i, c in enumerate(chunks):
        if i:
            out.append(gap)
        out.append(c)
    return np.concatenate(out).astype(np.float32)


def duration_seconds(wav: np.ndarray, sr: int) -> float:
    return round(float(wav.size) / sr, 2) if sr else 0.0


def decode_audio_bytes(data: bytes) -> tuple[np.ndarray, int]:
    """把 HTTP 回傳的音檔位元組解成 (單聲道 float32, 取樣率)。

    wav / flac / ogg 直接用 soundfile；mp3 / aac 這類再退回 ffmpeg。
    """
    import io

    if not data:
        raise AudioError("遠端回傳的音檔是空的")

    try:
        wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
        wav = np.asarray(wav, dtype=np.float32)
        if wav.ndim > 1:
            wav = wav.mean(axis=1).astype(np.float32)
        return wav, int(sr)
    except Exception as exc:  # noqa: BLE001  可能是 mp3 之類
        sf_error = exc

    if not _ffmpeg_available():
        raise AudioError(
            f"無法解碼遠端回傳的音檔：{sf_error}。"
            "請把 TTS_RESPONSE_FORMAT 設成 wav，或安裝 ffmpeg（brew install ffmpeg）。"
        ) from sf_error

    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "remote.bin"
        out = Path(td) / "decoded.wav"
        src.write_bytes(data)
        cmd = [
            "ffmpeg", "-nostdin", "-loglevel", "error", "-y",
            "-i", str(src), "-ac", "1", str(out),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 or not out.exists():
            raise AudioError(f"ffmpeg 解碼遠端音檔失敗：{proc.stderr.strip()[:400]}")
        wav, sr = sf.read(str(out), dtype="float32", always_2d=False)
    return np.asarray(wav, dtype=np.float32), int(sr)


def prepare_prompt_wav(src: str | Path, dst: str | Path) -> float:
    """把使用者上傳的音檔轉成音色庫用的 16k 單聲道 wav，回傳長度（秒）。

    放在這裡而不是某個引擎裡：音色庫是平台共用的，不屬於任何一個引擎。
    """
    from . import config

    wav = load_mono(src, config.PROMPT_SAMPLE_RATE)
    wav = trim_silence(wav)
    wav = peak_normalize(wav, 0.9)
    save_wav(dst, wav, config.PROMPT_SAMPLE_RATE)
    return round(len(wav) / config.PROMPT_SAMPLE_RATE, 2)
