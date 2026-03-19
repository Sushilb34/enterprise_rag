import json
import os
from datetime import datetime


class EvaluationLogger:
    """
    Lightweight evaluation logger for RAG pipeline.
    Stores per-query metrics in JSONL(JSON per Line) format.
    """

    def __init__(self, log_path: str = "logs/evaluation.jsonl"):
        self.log_path = log_path

        # ensure directory exists
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def log(self, data: dict):
        """
        Append a single evaluation record.
        """
        data["timestamp"] = datetime.utcnow().isoformat()

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data) + "\n")