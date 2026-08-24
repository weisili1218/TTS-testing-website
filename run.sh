#!/usr/bin/env bash
# 啟動 TTS 評測平台
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

export PYTHONUTF8=1
export TOKENIZERS_PARALLELISM=false

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"

PY="${PYTHON:-python}"
if [ -x "$ROOT/.venv-console/bin/python" ]; then
  PY="$ROOT/.venv-console/bin/python"
fi
if ! "$PY" -c "import fastapi, httpx, soundfile" >/dev/null 2>&1; then
  echo "找不到必要套件，請先執行： pip install -r requirements-console.txt" >&2
  exit 1
fi

if [ ! -f "$ROOT/engines.json" ]; then
  echo "⚠ 找不到 engines.json。先複製一份再填位址：" >&2
  echo "    cp engines.example.json engines.json" >&2
  exit 1
fi

echo "TTS 評測平台 → http://${HOST}:${PORT}"
# 直接把展開後的位址印出來 —— 主機填錯是最常見的問題，
# 讓它在啟動的第一行就看得見，而不是等到第一次評測失敗才發現。
"$PY" - <<'PYEOF'
from app import config
print(f"  主機 TTS_HOST = {config.TTS_HOST}")
for spec in config.ENGINE_SPECS:
    print(f"  {spec['label']:16s} → {spec['base_url'] or '（未設定，停用）'}")
print(f"  {'F5-TTS':16s} → {config.F5_BASE_URL or '（未設定，停用）'}")
print(f"  {'ASR 評分':14s} → {config.ASR_BASE_URL or '（未設定，停用）'}")
PYEOF

exec "$PY" -m uvicorn app.main:app --host "$HOST" --port "$PORT"
