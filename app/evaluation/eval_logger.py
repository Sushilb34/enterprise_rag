import json
import os
from datetime import datetime

from app.core.config import get_settings

settings = get_settings()

class EvaluationLogger:
    """
    Lightweight evaluation logger for RAG pipeline.
    Stores per-query metrics in JSONL(JSON per Line) format.
    """

    def __init__(self):
        self.log_path = settings.LOG_PATH

        # ensure directory exists
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def log(self, data: dict):
        """
        Append a single evaluation record.
        """
        data["timestamp"] = datetime.utcnow().isoformat()

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data) + "\n")