"""四包 TTS gateway 的轉接器（OpenAI 相容，可多實例）。

cosyvoice2 / fun-cosyvoice3 / qwen3-tts / voxcpm2 四包用的是**同一份 gateway
app.py**，所以對外 API 只有一份形狀，差別只在 port 與底下那顆引擎宣告支援哪些
mode。因此這裡不是「四個轉接器」，而是**一個轉接器、四個實例** —— 每個實例讀
自己的一份設定（見 config.ENGINE_SPECS 與 engines.json）。

    POST {base}/v1/audio/speech   {"input","model","voice","speed",...} -> 音檔位元組
    GET  {base}/v1/models         -> 能力宣告（modes / presets / sample_rate / loaded）
    POST {base}/v1/warmup         -> 把模型載進 GPU（冷啟動要 30–90 秒）
    GET  {base}/v1/voices         -> 音色清單（自建 clone/design + 內建 preset）

能力一律用探測的，不寫死
------------------------
`modes`、`presets`、`sample_rate`、`needs_ref_audio` 全部來自 `/v1/models`。
gateway 自己就是這樣設計的（`ENGINE_MODES` 環境變數決定路由），所以之後把
qwen3-tts 換成會克隆的 Base checkpoint，這裡一行都不用改，探測到 `clone`
就會自動改用音色庫的參考音。

音色怎麼決定（`_pick_voice`）
-----------------------------
1. 支援 clone、而且這個本機音色已經推送過去了 -> 用遠端 voice id（**音色可比**）
2. 支援 preset -> 用設定的 speaker（`preset_voice`）（音色不可比）
3. 支援 default -> 不帶 voice，讓模型自己生（音色不可比）
4. 都不行 -> 報錯，訊息告訴使用者要先推送音色

第 1 條是整個平台能不能談「音色相似度」的關鍵，推送邏輯在 app/voice_sync.py。
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Optional

import numpy as np

from ..audio_utils import decode_audio_bytes
from .base import RemoteError, TTSAdapter, describe_http_error, friendly_error

log = logging.getLogger("tts.gateway")

# gateway 回 503 通常是「引擎還在載模型」，值得等久一點再試
_LOADING_BACKOFF = (3.0, 8.0, 15.0)


class GatewayAdapter(TTSAdapter):
    """一包 gateway（一個 port）。設定從 spec 進來，能力從 /v1/models 探測。"""

    def __init__(self, spec: dict) -> None:
        super().__init__()
        self.spec = spec
        self.key = spec["key"]
        self.label = spec["label"]
        self._client = None
        self._client_lock = threading.Lock()
        self._caps: dict = {}
        self._caps_at: float = 0.0

    # ------------------------------------------------------------ 基本資訊
    @property
    def enabled(self) -> bool:
        return bool(self.spec.get("base_url")) and bool(self.spec.get("enabled", True))

    @property
    def base_url(self) -> str:
        return self.spec["base_url"]

    @property
    def endpoint(self) -> str:
        return self.base_url + self.spec["speech_path"]

    @property
    def remote_speed(self) -> bool:  # type: ignore[override]
        """服務端會不會真的用 speed。False 的話由基底類別做本機變速。"""
        return self.spec.get("speed_mode") == "remote"

    def _url(self, path_field: str) -> str:
        return self.base_url + self.spec[path_field]

    # ------------------------------------------------------------ HTTP
    def _http(self):
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    import httpx

                    api_key = str(self.spec.get("api_key") or "").strip()
                    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
                    self._client = httpx.Client(
                        timeout=httpx.Timeout(
                            float(self.spec["timeout"]),
                            connect=float(self.spec["connect_timeout"]),
                        ),
                        headers=headers,
                        follow_redirects=True,
                    )
        return self._client

    # ------------------------------------------------------------ 能力探測
    def capabilities(self, refresh: bool = False) -> dict:
        """從 /v1/models 讀這顆引擎宣告的能力。帶快取，避免每次合成都打一次。

        注意 gateway 自己對引擎 /health 有 300 秒快取，冷啟動時 `presets` 會是
        空的而且會被快取住 —— 所以 warmup() 打完 /v1/warmup 後要清我們這層快取，
        gateway 那層它自己會清。
        """
        ttl = float(self.spec.get("caps_ttl", 60))
        if not refresh and self._caps and (time.time() - self._caps_at) < ttl:
            return self._caps
        if not self.enabled:
            return {}

        caps: dict = {}
        try:
            import httpx

            # 能力探測是狀態查詢，不是合成 —— 連不上就要很快放棄，
            # 不然引擎關機時每次開頁面都要等滿合成用的連線逾時。
            r = self._http().get(
                self._url("models_path"),
                timeout=httpx.Timeout(float(self.spec.get("caps_timeout", 6)),
                                      connect=float(self.spec.get("caps_connect_timeout", 3))))
            if r.status_code >= 400:
                caps = {"error": describe_http_error(r.status_code, r.content,
                                                     self._url("models_path"))}
            else:
                data = r.json()
                items = data.get("data", []) if isinstance(data, dict) else data
                wanted = str(self.spec.get("model") or self.key)
                row = None
                if isinstance(items, list) and items:
                    row = next((m for m in items
                                if isinstance(m, dict)
                                and (m.get("id") == wanted or m.get("engine") == wanted)),
                               items[0] if isinstance(items[0], dict) else None)
                if row:
                    caps = {
                        "id": row.get("id"),
                        "engine": row.get("engine"),
                        "status": row.get("status"),
                        "loaded": bool(row.get("loaded")),
                        "sample_rate": row.get("sample_rate"),
                        "modes": [str(m) for m in (row.get("modes") or [])],
                        "needs_ref_audio": bool(row.get("needs_ref_audio")),
                        "presets": [str(p) for p in (row.get("presets") or [])],
                        "error": row.get("error"),
                    }
                else:
                    caps = {"error": "/v1/models 沒有回傳任何引擎"}
        except Exception as exc:  # noqa: BLE001
            caps = {"error": f"{exc.__class__.__name__}: {exc}"}

        self._caps = caps
        self._caps_at = time.time()
        return caps

    @property
    def modes(self) -> list[str]:
        return self.capabilities().get("modes") or []

    def supports(self, mode: str) -> bool:
        return mode in self.modes

    @property
    def can_clone(self) -> bool:
        """能不能用我們指定的參考音做 zero-shot 克隆 —— 音色可不可比看這個。"""
        return self.supports("clone")

    # ------------------------------------------------------------ 音色決策
    def _remote_voice_id(self, opts: dict) -> Optional[str]:
        """這個本機音色在這顆引擎上的遠端 voice id（推送過才有）。"""
        mapping = opts.get("_voice_remote_ids") or {}
        return mapping.get(self.key)

    def _pick_voice(self, opts: dict) -> tuple[Optional[str], str, str]:
        """決定要送哪個 voice。回 (voice 欄位值, 音色類型, 給人看的說明)。"""
        remote_id = self._remote_voice_id(opts)
        if self.can_clone and remote_id:
            name = opts.get("voice_name") or remote_id
            return remote_id, "clone", f"{name}（克隆）"

        if self.supports("preset"):
            preset = str(self.spec.get("preset_voice") or "").strip()
            if not preset:
                available = self.capabilities().get("presets") or []
                preset = available[0] if available else ""
            if preset:
                return preset, "preset", f"{preset}（內建 speaker）"
            # preset 清單是空的：多半是還沒 warmup，gateway 快取住了空值
            return None, "preset", "內建 speaker（尚未 warmup，清單是空的）"

        if self.supports("default"):
            return None, "default", "模型自動生成音色"

        if self.can_clone and not remote_id:
            raise ValueError(
                f"{self.label} 只能用克隆音色，但音色「{opts.get('voice_name') or '—'}」"
                "還沒推送到這顆引擎。請到「音色」分頁按「推送到各引擎」。"
            )
        raise ValueError(f"{self.label} 沒有宣告任何可用的 mode（/v1/models 回的 modes 是空的）")

    # ------------------------------------------------------------ 必填欄位
    def _required_fields(self) -> set[str]:
        """這顆引擎的 API 規定不能省略的欄位（見 config 的 required_fields）。"""
        return set(self.spec.get("required_fields") or [])

    def _resolve_language(self, opts: dict) -> str:
        """語言：引擎設定優先，其次沿用音色上填的語言。"""
        return str(self.spec.get("language") or opts.get("voice_language") or "").strip()

    def _check_required(self, voice_value: Optional[str], language: str) -> None:
        """必填欄位少了就在本機擋下來，附上要去哪裡補。

        等遠端回錯誤才知道的話，訊息只會是一句 validation error，看不出要填哪裡；
        批次跑到一半才炸掉更難救。
        """
        required = self._required_fields()
        missing: list[str] = []
        if "voice" in required and not str(voice_value or "").strip():
            missing.append("voice（clone 引擎請先到「音色」分頁推送；"
                           "preset 引擎請設定 preset_voice）")
        if "language" in required and not language:
            missing.append("language（語言名稱，例如 Chinese）")
        if not missing:
            return
        raise ValueError(
            f"{self.label} 的 API 規定 " + "、".join(sorted(required))
            + " 都必填，目前缺少：" + "；".join(missing)
            + f"。請在 engines.json 的 {self.key} 補上對應欄位。"
        )

    @property
    def needs_voice(self) -> bool:  # type: ignore[override]
        """要不要強制先挑一個本機音色。

        只有 clone 的引擎才非挑不可；有 preset / default 可退的就不強制 ——
        不然 qwen3-tts 明明用內建 speaker，卻被逼著選一個它根本用不到的音色。
        """
        if not self.enabled:
            return False
        modes = self.modes
        if not modes:
            return False
        return "clone" in modes and not ({"preset", "default"} & set(modes))

    def voice_label(self, opts: dict) -> str:
        try:
            return self._pick_voice(opts)[2]
        except Exception:  # noqa: BLE001
            return "—"

    def uses_custom_reference(self, opts: dict) -> bool:
        """有沒有真的用到我們指定的那段參考音。音色可比性就是看這個。"""
        try:
            return self._pick_voice(opts)[1] == "clone"
        except Exception:  # noqa: BLE001
            return False

    def validate(self, opts: dict) -> None:
        if not self.enabled:
            raise ValueError(f"{self.label} 未設定 base_url，停用中")
        voice_value, kind, _ = self._pick_voice(opts)
        if kind == "clone" and not str(opts.get("voice_transcription") or "").strip():
            # 沒逐字稿 CosyVoice 會退到 cross-lingual、VoxCPM2 少掉 ultimate cloning，
            # 兩者相似度都明顯下降 —— 這會讓音色比較變成在比「有沒有填逐字稿」。
            raise ValueError(
                f"音色「{opts.get('voice_name') or '—'}」沒有逐字稿。"
                "克隆模型少了逐字稿相似度會明顯下降，請先按「自動辨識」或手動填寫。"
            )
        if kind == "preset" and not voice_value:
            raise ValueError(
                f"{self.label} 要用內建 speaker，但清單是空的。"
                "先按「暖機」把模型載進 GPU（preset 清單是載入時才填的）。"
            )
        # 必填欄位在這裡就擋掉，不要等批次跑到一半才炸
        self._check_required(voice_value, self._resolve_language(opts))

    def describe(self) -> dict:
        caps = self.capabilities()
        return {
            "key": self.key,
            "label": self.label,
            "enabled": self.enabled,
            "endpoint": self.endpoint,
            "base_url": self.base_url,
            "model": self.spec.get("model"),
            "kind": "gateway",
            "needs_voice": self.needs_voice,
            "modes": caps.get("modes") or [],
            "can_clone": "clone" in (caps.get("modes") or []),
            "presets": caps.get("presets") or [],
            "preset_voice": self.spec.get("preset_voice") or "",
            "sample_rate": caps.get("sample_rate"),
            "loaded": caps.get("loaded"),
            "status": caps.get("status"),
            "speed_mode": self.spec.get("speed_mode"),
            "honors_speed": self.remote_speed,
            "language": self.spec.get("language") or "",
            "required_fields": sorted(self._required_fields()),
            "notes": self.spec.get("notes") or "",
            "caps_error": caps.get("error"),
        }

    # ------------------------------------------------------------ 暖機
    def warmup(self) -> dict:
        """打 gateway 的 /v1/warmup 把模型載進 GPU。

        第一次合成要等 30–90 秒載模型，不先做這件事，第一筆速度數字量到的
        是「載模型 + 合成」，跟其他引擎完全不能比。
        """
        if not self.enabled:
            return {"ok": False, "error": f"{self.label} 未設定，停用中"}
        started = time.time()
        try:
            r = self._http().post(self._url("warmup_path"), timeout=300)
            if r.status_code >= 400:
                return {"ok": False,
                        "error": describe_http_error(r.status_code, r.content,
                                                     self._url("warmup_path"))}
            payload = r.json() if "json" in (r.headers.get("content-type") or "") else {}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"{exc.__class__.__name__}: {exc}",
                    "elapsed_seconds": round(time.time() - started, 2)}

        # warmup 會清掉 gateway 的能力快取，我們這層也要跟著清，
        # 不然剛載好的 preset 清單還是會讀到舊的空值
        caps = self.capabilities(refresh=True)
        return {
            "ok": True,
            "error": None,
            "elapsed_seconds": round(time.time() - started, 2),
            "loaded": caps.get("loaded"),
            "sample_rate": caps.get("sample_rate"),
            "presets": caps.get("presets") or [],
            "raw": payload,
        }

    # ------------------------------------------------------------ 健康檢查
    def health(self, deep: bool = False) -> dict:
        if not self.enabled:
            return {"ok": False, "error": f"{self.label} 未設定 base_url，停用中"}

        info: dict[str, Any] = {"endpoint": self.endpoint, "base_url": self.base_url}
        caps = self.capabilities(refresh=True)
        info.update({
            "modes": caps.get("modes") or [],
            "presets": caps.get("presets") or [],
            "sample_rate": caps.get("sample_rate"),
            "loaded": caps.get("loaded"),
            "status": caps.get("status"),
        })

        if caps.get("error"):
            return {**info, "ok": False, "error": caps["error"]}
        if caps.get("status") == "unreachable":
            return {**info, "ok": False,
                    "error": f"gateway 連不到引擎：{caps.get('error') or '未知原因'}"}
        if not caps.get("loaded"):
            info["hint"] = "模型還沒載入（status=idle），按「暖機」載入後再測速度"

        if deep:
            smoke = self.smoke_test()
            info["smoke"] = smoke
            info["ok"] = smoke["ok"]
            info["error"] = smoke["error"]
            return info

        info.update(ok=True, error=None)
        return info

    def smoke_test(self, text: str = "測試。") -> dict:
        """實際合成一句短句。clone-only 的引擎要先借一個音色，其餘不用。"""
        if not self.enabled:
            return {"ok": False, "error": f"{self.label} 未設定，停用中"}

        opts: dict = {"speed": 1.0, "normalize_volume": False}
        if self.needs_voice:
            from .. import voice_sync, voices

            usable = [v for v in voices.list_voices()
                      if v.get("has_audio") and v.get("transcription")]
            if not usable:
                return {"ok": False,
                        "error": f"{self.label} 需要克隆音色，但音色庫裡沒有可用的音色"}
            opts.update(voice_sync.voice_opts(usable[0]))

        started = time.time()
        try:
            _audio, sr, info = self.synthesize(text, opts, lambda *_a, **_k: None)
            return {"ok": True, "error": None, "sample_rate": sr,
                    "elapsed_seconds": round(time.time() - started, 2),
                    "duration_seconds": info.get("duration_seconds")}
        except Exception as exc:  # noqa: BLE001
            msg = friendly_error(str(exc))
            log.warning("%s 冒煙測試失敗：%s", self.label, msg)
            return {"ok": False, "error": msg,
                    "elapsed_seconds": round(time.time() - started, 2)}

    # ------------------------------------------------------------ 合成
    def _build_payload(self, segment: str, opts: dict) -> dict:
        voice_value, _kind, _label = self._pick_voice(opts)

        payload: dict[str, Any] = {
            "model": self.spec.get("model") or self.key,
            "input": segment,
            "response_format": self.spec.get("response_format", "wav"),
        }
        if voice_value:
            payload["voice"] = voice_value

        # speed：只有服務端真的會用的時候才送，否則基底類別會在本機做變速。
        # 送一個對方會忽略的 speed 比不送更糟 —— 看起來設定生效了，其實沒有。
        if self.remote_speed:
            payload["speed"] = round(float(opts.get("speed", 1.0)), 3)

        language = self._resolve_language(opts)
        self._check_required(voice_value, language)
        if language:
            payload["language"] = language

        # instructions 預設關掉：CosyVoice 收到就會走 inference_instruct2，
        # 不再走 zero-shot，音色相似度跟不給時不一樣 —— 那就不是同一個受測條件了。
        instructions = str(opts.get("instructions") or "").strip()
        if instructions and self.spec.get("allow_instructions"):
            payload["instructions"] = instructions

        for k, v in (self.spec.get("extra_fields") or {}).items():
            payload.setdefault(k, v)
        return payload

    def request_segment(self, segment: str, opts: dict) -> tuple[np.ndarray, int]:
        import httpx

        payload = self._build_payload(segment, opts)
        client = self._http()
        attempts = max(1, int(self.spec.get("max_retries", 2)) + 1)
        last: Optional[Exception] = None

        for attempt in range(attempts):
            try:
                resp = client.post(self.endpoint, json=payload)
            except httpx.HTTPError as exc:
                last = exc
                if attempt + 1 < attempts:
                    log.warning("%s 送出失敗（第 %d 次），重試：%s", self.label, attempt + 1, exc)
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise RemoteError(
                    f"連不到 {self.endpoint}：{type(exc).__name__}: {exc}") from exc

            if resp.status_code == 503:
                # gateway 的 503 = 連不到引擎，通常是還在載模型。等久一點再試。
                if attempt + 1 < attempts:
                    wait = _LOADING_BACKOFF[min(attempt, len(_LOADING_BACKOFF) - 1)]
                    log.warning("%s 回 503（引擎可能還在載模型），%.0f 秒後重試", self.label, wait)
                    time.sleep(wait)
                    last = RemoteError("引擎還在載入")
                    continue
                raise RemoteError(
                    f"{self.label} 持續回 503：gateway 連不到引擎，通常是模型還在載入。"
                    "先按「暖機」，或確認該容器已經起來。")

            if resp.status_code >= 400:
                msg = describe_http_error(resp.status_code, resp.content, self.endpoint)
                if resp.status_code >= 500 and attempt + 1 < attempts:
                    log.warning("%s 遠端 5xx（第 %d 次），重試：%s", self.label, attempt + 1, msg)
                    time.sleep(1.5 * (attempt + 1))
                    last = RemoteError(msg)
                    continue
                raise RemoteError(msg)

            if "json" in (resp.headers.get("content-type") or "").lower():
                raise RemoteError("遠端回傳 JSON 而不是音檔：" + resp.text.strip()[:400])
            return decode_audio_bytes(resp.content)

        raise RemoteError(str(last) if last else "遠端請求失敗")
