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

logger.add(
    sink="logs/rag_system.log",
    level=settings.LOG_LEVEL,
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