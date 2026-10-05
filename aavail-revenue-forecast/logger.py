import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_LOG_PATH = os.environ.get("AAVAIL_LOG_PATH", os.path.join("logs", "runtime.jsonl"))


def read_records(path, n=None):
    path = Path(path)
    if not path.exists():
        return []
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if n is not None and n > 0:
        records = records[-n:]
    return records


class RuntimeLogger:

    def __init__(self, path=None):
        self.path = Path(path or DEFAULT_LOG_PATH)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def log(self, endpoint, inputs=None, prediction=None, runtime=None,
            status="ok", model_version=None, note=None):
        record = {
            "unique_id": uuid.uuid4().hex,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "endpoint": endpoint,
            "input": inputs,
            "prediction": prediction,
            "runtime_seconds": None if runtime is None else round(float(runtime), 6),
            "model_version": model_version,
            "status": status,
            "note": note,
        }
        line = json.dumps(record, default=str)
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        return record["unique_id"]

    def read(self, n=None):
        return read_records(self.path, n)

    def tail_text(self, n=100):
        return "\n".join(json.dumps(r) for r in self.read(n))
