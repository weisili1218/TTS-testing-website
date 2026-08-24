"""引擎登錄表 —— 從設定動態建出來，不是寫死的清單。

四包 gateway（cosyvoice2 / fun-cosyvoice3 / qwen3-tts / voxcpm2）用的是同一份
app.py，所以它們共用一個轉接器類別 `GatewayAdapter`，只是各自吃一份設定。
要加第五顆模型，改 `engines.json` 就好，Python 一行都不用動。

介面完全不同的引擎（例如舊的 F5-TTS Speech Service）才需要自己的轉接器檔案：

  1. 在這個目錄開一個檔案，繼承 base.TTSAdapter，實作 request_segment()
  2. 在下面 _LEGACY_BUILTIN 加一行

停用中的引擎（設定沒填齊）不會出現在 UI 上，但仍留在登錄表裡 ——
排行榜要看得到它以前跑過的歷史紀錄。
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from .. import config
from .base import RemoteError, TTSAdapter, describe_http_error, friendly_error
from .f5_tts import F5Adapter
from .gateway import GatewayAdapter

log = logging.getLogger("tts.engines")

# 介面自成一格、需要專屬轉接器的引擎
_LEGACY_BUILTIN: list[type[TTSAdapter]] = [
    F5Adapter,
]

_REGISTRY: "dict[str, TTSAdapter]" = {}


def register(adapter: TTSAdapter) -> TTSAdapter:
    if not adapter.key:
        raise ValueError("引擎必須有 key")
    if adapter.key in _REGISTRY:
        raise ValueError(f"引擎 key 重複：{adapter.key}")
    _REGISTRY[adapter.key] = adapter
    return adapter


def _build_registry() -> None:
    # 順序 = UI 上的顯示順序：先設定檔裡的 gateway，再舊式轉接器
    for spec in config.ENGINE_SPECS:
        try:
            register(GatewayAdapter(spec))
        except ValueError as exc:
            log.error("引擎 %s 登錄失敗：%s", spec.get("key"), exc)
    for cls in _LEGACY_BUILTIN:
        try:
            register(cls())
        except ValueError as exc:
            log.error("引擎 %s 登錄失敗：%s", cls.__name__, exc)


_build_registry()

log.info("已登錄引擎：%s（啟用中：%s）",
         list(_REGISTRY),
         [k for k, a in _REGISTRY.items() if a.enabled])


def all_engines() -> list[TTSAdapter]:
    """全部登錄的引擎，含停用中的。"""
    return list(_REGISTRY.values())


def enabled_engines() -> list[TTSAdapter]:
    """設定齊全、可以拿來評測的引擎。"""
    return [a for a in _REGISTRY.values() if a.enabled]


def get(key: str) -> Optional[TTSAdapter]:
    return _REGISTRY.get(key)


def require(key: str) -> TTSAdapter:
    adapter = _REGISTRY.get(key)
    if adapter is None:
        raise ValueError(f"不認識的引擎：{key}")
    return adapter


def keys() -> list[str]:
    return list(_REGISTRY)


def enabled_keys() -> list[str]:
    return [a.key for a in enabled_engines()]


def label_for(key: str) -> str:
    """引擎的顯示名稱。查不到就回 key 本身 ——

    排行榜要能顯示「已經被拿掉的引擎」留下的歷史紀錄，不能因為查不到就爆掉。
    """
    adapter = _REGISTRY.get(key)
    return adapter.label if adapter else key


def prefetch_capabilities(refresh: bool = False) -> None:
    """並行把每顆引擎的能力宣告抓回快取。

    /api/status 要列出每顆引擎的 modes / 取樣率 / 有沒有載入，逐顆同步去打的話，
    只要有一顆關機就要等滿它的連線逾時 —— 五顆引擎全關機時開頁面會等好幾十秒。
    並行之後最久就是單顆的逾時。
    """
    targets = [a for a in _REGISTRY.values()
               if a.enabled and hasattr(a, "capabilities")]
    if not targets:
        return
    with ThreadPoolExecutor(max_workers=min(len(targets), 8)) as pool:
        futures = [pool.submit(a.capabilities, refresh) for a in targets]
        for future in futures:
            try:
                future.result(timeout=20)
            except Exception as exc:  # noqa: BLE001
                log.debug("能力探測失敗：%s", exc)


def clone_capable_keys() -> list[str]:
    """能用我們指定的參考音做克隆的引擎 —— 音色維度只在這群之間可比。"""
    return [a.key for a in enabled_engines() if getattr(a, "can_clone", False)]


def reload_from_config() -> list[str]:
    """重讀 engines.json 並重建登錄表。改完設定不用重啟服務。"""
    config.ENGINE_SPECS = config.load_engine_specs()
    _REGISTRY.clear()
    _build_registry()
    log.info("引擎清單已重載：%s", list(_REGISTRY))
    return list(_REGISTRY)


__all__ = [
    "TTSAdapter", "GatewayAdapter", "RemoteError",
    "describe_http_error", "friendly_error",
    "register", "all_engines", "enabled_engines", "get", "require",
    "keys", "enabled_keys", "label_for", "clone_capable_keys",
    "prefetch_capabilities", "reload_from_config",
]
