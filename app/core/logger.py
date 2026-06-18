import sys

from loguru import logger
from app.core.config import get_settings

settings = get_settings()

# Ensure the console stream can render unicode (exception tracebacks use
# box-drawing chars; answers may contain em-dashes/emojis). On Windows the
# default console encoding is cp1252, which otherwise crashes the print sink.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

logger.remove()

# Logging / PII policy:
#   - INFO and above carry only metadata (lengths, IDs, sources, scores) — no raw
#     user queries or document bodies. Safe for console / production monitoring.
#   - Full query text and retrieved document content are logged at DEBUG.
#   - The project log file captures DEBUG so a developer can read the full detail
#     in the logs/ folder; the console stays at the configured LOG_LEVEL so
#     production stdout isn't flooded with (potentially PII-bearing) content.
#   - Files rotate at 10 MB and are retained 10 days, then dropped.
logger.add(
    sink="logs/rag_system.log",
    level="DEBUG",
    rotation="10 MB",
    retention="10 days",
    compression="zip",
)

logger.add(
    sink=lambda msg: print(msg, end=""),
    level=settings.LOG_LEVEL
)

def get_logger():
    return logger