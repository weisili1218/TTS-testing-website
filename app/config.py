"""全域設定：路徑、引擎清單、評測參數。

這個專案是 TTS 評測平台：同一段文字送進多個引擎，比「速度」「吐字」「音色」。
引擎是可插拔且**可多實例**的 —— 四包 gateway 用的是同一份 OpenAI 相容 API，
所以不需要四個轉接器，只要四份設定。

引擎清單從哪裡來（依序，先找到的贏）
------------------------------------
1. `engines.json`（專案根目錄，可用 ENGINES_FILE 換路徑）—— 建議用這個
2. 環境變數 `TTS_ENGINES="key=url,key=url,..."` —— 快速上線用
3. 舊版的 `OPENAI_TTS_BASE_URL` / `F5_BASE_URL` —— 向後相容，不動舊設定也能跑

其餘設定都可以用環境變數覆寫（或寫在專案根目錄的 .env）。
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

log = logging.getLogger("tts.config")

try:  # .env 支援（非必要）
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass


def _env_bool(key: str, default: bool = False) -> bool:
    val = os.getenv(key)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_json(key: str, default: Any) -> Any:
    raw = os.getenv(key, "").strip()
    if not raw:
        return default
    try:
        return json.loads(raw)
    except Exception:  # noqa: BLE001
        log.warning("%s 不是合法的 JSON，忽略", key)
        return default


# ---------------------------------------------------------------- 專案路徑
PROJECT_ROOT = Path(__file__).resolve().parent.parent

STORAGE_DIR = Path(os.getenv("STORAGE_DIR", str(PROJECT_ROOT / "storage"))).resolve()
VOICES_DIR = STORAGE_DIR / "voices"          # 本機音色庫：每個音色一個資料夾
OUTPUTS_DIR = STORAGE_DIR / "outputs"        # 評測產生的音檔
TMP_DIR = STORAGE_DIR / "tmp"
REPORTS_DIR = STORAGE_DIR / "reports"        # 批次跑測的報表

for _d in (STORAGE_DIR, VOICES_DIR, OUTPUTS_DIR, TMP_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ================================================================ 引擎清單
# 四包 gateway 通常跑在同一台 GPU 主機上，只有 port 不同。把主機位址集中成一個
# 變數，engines.json 裡就可以寫 http://${TTS_HOST}:8001，換機器只改這一行。
TTS_HOST = os.getenv("TTS_HOST", "127.0.0.1").strip()

ENGINES_FILE = Path(os.getenv("ENGINES_FILE", str(PROJECT_ROOT / "engines.json")))

# 所有 gateway 共用的預設值。單一引擎可以在 engines.json 裡各自覆寫。
GATEWAY_DEFAULTS: dict[str, Any] = {
    # 端點路徑（四包 gateway 都是這組）
    "speech_path": "/v1/audio/speech",
    "models_path": "/v1/models",
    "warmup_path": "/v1/warmup",
    "voices_path": "/v1/voices",
    "health_path": "/healthz",

    "api_key": os.getenv("TTS_API_KEY", "").strip(),
    "model": "",                  # 留空 = 用 key 當 model 名稱
    "response_format": "wav",     # wav 最省事，不用 ffmpeg 轉

    # speed 只有 CosyVoice 兩顆真的會傳進 inference，其餘會忽略。
    # "remote" = 交給服務端；"local" = 本機變速不變調（不計入合成耗時）。
    # 預設 local：寧可自己做，也不要「調了沒反應」造成引擎之間語速不一致。
    "speed_mode": "local",

    # 宣告 preset mode 的引擎要用哪個內建 speaker（四包目前都沒有 preset）
    "preset_voice": "",
    # 語言名稱（例如 Chinese）。留空則沿用音色上填的設定
    "language": "",

    # 這顆引擎的 API 規定哪些欄位「一定要有值」。空的就不送是 OpenAI 的慣例，
    # 但不是每顆都這麼寬鬆 —— 列在這裡的欄位會在 validate() 就先擋下來，
    # 不會等送出去才收到一句看不出要補哪裡的遠端 validation error。
    "required_fields": [],

    # instructions（語氣指示）預設不送 —— CosyVoice 收到就會離開 zero-shot
    # 路徑改走 inference_instruct2，音色相似度會跟不給時不一樣，等於偷換受測條件。
    "allow_instructions": False,

    "timeout": float(os.getenv("TTS_TIMEOUT", "600")),
    "connect_timeout": float(os.getenv("TTS_CONNECT_TIMEOUT", "10")),
    "max_retries": int(os.getenv("TTS_MAX_RETRIES", "2")),

    # gateway 對引擎 /health 有 300 秒快取；我們自己也快取一份能力宣告
    "caps_ttl": float(os.getenv("TTS_CAPS_TTL", "60")),
    # 能力探測的逾時要比合成短很多 —— 引擎關機時開頁面不該卡住
    "caps_timeout": float(os.getenv("TTS_CAPS_TIMEOUT", "6")),
    "caps_connect_timeout": float(os.getenv("TTS_CAPS_CONNECT_TIMEOUT", "3")),

    "extra_fields": {},
    "notes": "",
    "enabled": True,
}

# 常見引擎的預設值（只在 engines.json 沒寫的欄位才套用）。
# 這不是「寫死能力」—— modes / presets / sample_rate 一律從 /v1/models 探測，
# 這裡只放 API 探測不到的東西：顯示名稱、speed 有沒有效、preset 要用誰。
KNOWN_ENGINE_HINTS: dict[str, dict] = {
    "cosyvoice2":     {"label": "CosyVoice2",    "speed_mode": "remote"},
    "fun-cosyvoice3": {"label": "FunCosyVoice3", "speed_mode": "remote"},
    "qwen3-tts":      {"label": "Qwen3-TTS",     "speed_mode": "local",
                       "language": "Chinese",
                       "required_fields": ["language"]},
    "voxcpm2":        {"label": "VoxCPM2",       "speed_mode": "local"},
}


def _expand(value: str) -> str:
    """展開 base_url 裡的 ${VAR} —— 四包 gateway 通常在同一台主機上，
    寫成 http://${TTS_HOST}:8001 就只要改一個環境變數，不用四行各改一次。"""
    def _sub(m):
        name = m.group(1)
        return os.getenv(name, "") or (TTS_HOST if name == "TTS_HOST" else "")
    return re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", _sub, value or "")


def _normalize_spec(raw: dict) -> dict | None:
    key = str(raw.get("key", "")).strip()
    base_url = _expand(str(raw.get("base_url", ""))).strip().rstrip("/")
    if not key:
        log.warning("引擎設定缺少 key，跳過：%s", raw)
        return None

    spec = dict(GATEWAY_DEFAULTS)
    spec.update(KNOWN_ENGINE_HINTS.get(key, {}))
    spec.update({k: v for k, v in raw.items() if v is not None})

    spec["key"] = key
    spec["base_url"] = base_url
    spec["label"] = str(spec.get("label") or key)
    spec["model"] = str(spec.get("model") or key)
    for path_field in ("speech_path", "models_path", "warmup_path",
                       "voices_path", "health_path"):
        spec[path_field] = "/" + str(spec[path_field]).lstrip("/")
    if spec.get("speed_mode") not in ("remote", "local"):
        spec["speed_mode"] = "local"
    if not isinstance(spec.get("extra_fields"), dict):
        spec["extra_fields"] = {}
    req = spec.get("required_fields")
    if isinstance(req, str):
        req = [req]
    spec["required_fields"] = [str(f).strip() for f in (req or []) if str(f).strip()]
    return spec


def _specs_from_file() -> list[dict]:
    if not ENGINES_FILE.exists():
        return []
    try:
        data = json.loads(ENGINES_FILE.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        log.error("%s 解析失敗（%s），改用環境變數", ENGINES_FILE, exc)
        return []
    if isinstance(data, dict):
        data = data.get("engines", [])
    if not isinstance(data, list):
        log.error("%s 的內容應該是一個陣列", ENGINES_FILE)
        return []
    return [s for s in (_normalize_spec(d) for d in data if isinstance(d, dict)) if s]


def _specs_from_env() -> list[dict]:
    """TTS_ENGINES="cosyvoice2=http://h:8001,qwen3-tts=http://h:8003" """
    raw = os.getenv("TTS_ENGINES", "").strip()
    if not raw:
        return []
    out: list[dict] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            log.warning("TTS_ENGINES 的 %r 缺少 key=url 格式，跳過", item)
            continue
        key, url = item.split("=", 1)
        spec = _normalize_spec({"key": key.strip(), "base_url": url.strip()})
        if spec:
            out.append(spec)
    return out


def _specs_legacy() -> list[dict]:
    """舊版單一 OPENAI_TTS_* 設定 —— 不改 .env 也能繼續跑。"""
    base = os.getenv("OPENAI_TTS_BASE_URL", "").strip().rstrip("/")
    if not base:
        return []
    spec = _normalize_spec({
        "key": "openai",
        "label": os.getenv("OPENAI_TTS_LABEL", "OpenAI 相容 TTS").strip(),
        "base_url": base,
        "model": os.getenv("OPENAI_TTS_MODEL", "tts-1"),
        "api_key": os.getenv("OPENAI_TTS_API_KEY", "").strip(),
        "speech_path": os.getenv("OPENAI_TTS_SPEECH_PATH", "/audio/speech"),
        "models_path": os.getenv("OPENAI_TTS_MODELS_PATH", "/models"),
        "voices_path": "/voices",
        "warmup_path": "/warmup",
        "speed_mode": "remote",
        "response_format": os.getenv("OPENAI_TTS_RESPONSE_FORMAT", "wav").strip().lower(),
        "extra_fields": _env_json("OPENAI_TTS_EXTRA_FIELDS", {}),
    })
    return [spec] if spec else []


def load_engine_specs() -> list[dict]:
    specs = _specs_from_file()
    source = str(ENGINES_FILE)
    if not specs:
        specs = _specs_from_env()
        source = "TTS_ENGINES 環境變數"
    if not specs:
        specs = _specs_legacy()
        source = "OPENAI_TTS_*（舊版設定）"
    if specs:
        log.info("引擎清單來源：%s（%d 個）", source, len(specs))
    else:
        log.warning("沒有任何引擎設定。複製 engines.example.json 成 engines.json 再填位址。")
    return specs


ENGINE_SPECS = load_engine_specs()


# ------------------------------------------------------------ 引擎：F5-TTS
# 舊的 F5 Speech Service（介面跟四包 gateway 不一樣，所以留獨立轉接器）。
# 留空字串 = 停用，不會出現在 UI 上。
F5_BASE_URL = os.getenv("F5_BASE_URL", "").strip().rstrip("/")
F5_SPEECH_PATH = "/" + os.getenv("F5_SPEECH_PATH", "/speech-tts").lstrip("/")
F5_HEALTH_PATH = "/" + os.getenv("F5_HEALTH_PATH", "/tts-health").lstrip("/")
F5_ENGINE = os.getenv("F5_ENGINE", "").strip()
F5_LANGUAGE = os.getenv("F5_LANGUAGE", "zh").strip()
F5_SAMPLE_RATE = int(os.getenv("F5_SAMPLE_RATE", "24000"))

# F5 的 ref_audio 只接受「它自己那台機器上的檔案路徑」，純 base64 會被回
# "reference audio not found"。要讓 F5 也參加音色比較，就得先把 wav 複製過去。
F5_REF_AUDIO = os.getenv("F5_REF_AUDIO", "").strip()
F5_REF_TEXT = os.getenv("F5_REF_TEXT", "").strip()

F5_TIMEOUT = float(os.getenv("F5_TIMEOUT", "180"))
F5_CONNECT_TIMEOUT = float(os.getenv("F5_CONNECT_TIMEOUT", "10"))
F5_MAX_RETRIES = int(os.getenv("F5_MAX_RETRIES", "1"))

# 啟動時先確認各引擎連得上
CHECK_REMOTE_ON_START = _env_bool("CHECK_REMOTE_ON_START", True)


# ---------------------------------------------------------------- 合成參數
# 長文要不要先切段、逐段送出。所有引擎共用這個設定 ——
# 切法一致，測出來的速度才可以互相比較。
SPLIT_SEGMENTS = _env_bool("SPLIT_SEGMENTS", True)

PROMPT_SAMPLE_RATE = 16000   # 參考音檔需要的取樣率

MAX_CHARS_PER_SEGMENT = int(os.getenv("MAX_CHARS_PER_SEGMENT", "60"))
SEGMENT_GAP_SECONDS = float(os.getenv("SEGMENT_GAP_SECONDS", "0.18"))

# 參考音檔長度限制（秒）。gateway 的規則是 < 2 秒直接 400、> 30 秒給 warning，
# 這裡對齊它，讓問題在本機就被擋下來而不是等上傳到一半才失敗。
MIN_PROMPT_SECONDS = float(os.getenv("MIN_PROMPT_SECONDS", "2"))
MAX_PROMPT_SECONDS = float(os.getenv("MAX_PROMPT_SECONDS", "30"))

MAX_TEXT_CHARS = int(os.getenv("MAX_TEXT_CHARS", "2000"))


# ---------------------------------------------------------------- 伺服器
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8080"))
API_KEY = os.getenv("API_KEY", "").strip()


# ------------------------------------------------------- 遠端 ASR（客觀評分）
# 把合成音檔轉回文字、算字錯誤率（CER）：
#     POST {ASR_BASE_URL}/speech-asr   multipart file=<wav>   ->  {"text": "..."}
# 沒設就沿用 F5_BASE_URL；兩個都空則停用客觀評分。
ASR_BASE_URL = (os.getenv("ASR_BASE_URL", "").strip() or F5_BASE_URL).rstrip("/")
ASR_SPEECH_PATH = "/" + os.getenv("ASR_SPEECH_PATH", "/speech-asr").lstrip("/")
ASR_HEALTH_PATH = "/" + os.getenv("ASR_HEALTH_PATH", "/asr-health").lstrip("/")
ASR_LANGUAGE = os.getenv("ASR_LANGUAGE", "zh").strip()
ASR_TIMEOUT = float(os.getenv("ASR_TIMEOUT", "120"))
ASR_CONNECT_TIMEOUT = float(os.getenv("ASR_CONNECT_TIMEOUT", "10"))
ASR_ENABLED = bool(ASR_BASE_URL)

SCORE_BY_DEFAULT = _env_bool("SCORE_BY_DEFAULT", True)


# ---------------------------------------------------------------- 評測平台
TRIALS_LIMIT = int(os.getenv("TRIALS_LIMIT", "500"))

# 每次評測隨機決定誰先跑，避免「先跑的比較冷、後跑的比較熱」造成系統性偏差
RANDOMIZE_RUN_ORDER = _env_bool("RANDOMIZE_RUN_ORDER", True)

# 預設的盲測模式：duel（隨機抽兩個對決）/ ranking（全部排名）/ open（不盲測）
DEFAULT_TRIAL_MODE = os.getenv("DEFAULT_TRIAL_MODE", "duel").strip().lower()

# 同一句重複跑幾次來看穩定性（1 = 不重複）
DEFAULT_REPEATS = int(os.getenv("DEFAULT_REPEATS", "1"))
MAX_REPEATS = int(os.getenv("MAX_REPEATS", "10"))

# Elo 參數。K 越大越快反應近期表現，但也越抖。
ELO_K = float(os.getenv("ELO_K", "24"))
ELO_START = float(os.getenv("ELO_START", "1500"))
# Elo 信賴區間的 bootstrap 次數（0 = 不算）
ELO_BOOTSTRAP = int(os.getenv("ELO_BOOTSTRAP", "200"))

# 音色推送：把本機音色庫的參考音上傳到各引擎，讓克隆比較公平
VOICE_MAP_FILE = STORAGE_DIR / "voice_map.json"
AUTO_PUSH_VOICE = _env_bool("AUTO_PUSH_VOICE", True)
