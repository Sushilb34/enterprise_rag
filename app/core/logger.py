from loguru import logger
from app.core.config import get_settings

settings = get_settings()

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