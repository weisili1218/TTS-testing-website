"""非同步工作佇列。

一次評測要依序跑完兩個引擎再加 ASR 評分，可能好幾十秒，所以 API 不卡住：
POST 後回傳 job_id，前端輪詢 /api/jobs/{id} 看進度。

executor 固定單一執行緒 —— 這不只是為了不壓垮遠端，更是評測的正確性要求：
兩個引擎跑在同一台機器上，若同時有別的評測在跑，測出來的秒數就不可比了。
"""

from __future__ import annotations

import threading
import traceback
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Callable, Optional

_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts")
_jobs: "OrderedDict[str, dict]" = OrderedDict()
_lock = threading.Lock()
_MAX_JOBS = 200

def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


# ------------------------------------------------------------------ jobs
def create_job(kind: str, meta: Optional[dict] = None) -> str:
    job_id = uuid.uuid4().hex[:12]
    with _lock:
        _jobs[job_id] = {
            "id": job_id,
            "kind": kind,
            "status": "queued",
            "progress": 0.0,
            "message": "排隊中…",
            "result": None,
            "error": None,
            "created_at": _now(),
            "meta": meta or {},
        }
        while len(_jobs) > _MAX_JOBS:
            _jobs.popitem(last=False)
    return job_id


def update_job(job_id: str, **fields) -> None:
    with _lock:
        job = _jobs.get(job_id)
        if job:
            job.update(fields)


def get_job(job_id: str) -> Optional[dict]:
    with _lock:
        job = _jobs.get(job_id)
        return dict(job) if job else None


def submit(job_id: str, fn: Callable[[Callable[[str, float], None]], Any]) -> None:
    """把工作丟進單執行緒佇列；fn 會收到一個 progress(message, ratio) callback。"""

    def _progress(message: str, ratio: float) -> None:
        update_job(job_id, message=message, progress=round(float(ratio), 3))

    def _run() -> None:
        update_job(job_id, status="running", message="開始處理…", started_at=_now())
        try:
            result = fn(_progress)
            update_job(
                job_id,
                status="done",
                progress=1.0,
                message="完成",
                result=result,
                finished_at=_now(),
            )
        except Exception as exc:  # noqa: BLE001
            update_job(
                job_id,
                status="error",
                message="發生錯誤",
                error=f"{type(exc).__name__}: {exc}",
                traceback=traceback.format_exc(limit=6),
                finished_at=_now(),
            )

    _executor.submit(_run)


def queue_depth() -> int:
    with _lock:
        return sum(1 for j in _jobs.values() if j["status"] in ("queued", "running"))
