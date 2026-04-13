import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

_RUN_LOG_FILE = None


def _get_run_log_file() -> Path:
    global _RUN_LOG_FILE

    if _RUN_LOG_FILE is None:
        log_dir = Path(os.getenv("LOG_DIR", "logs"))
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        _RUN_LOG_FILE = log_dir / f"rag_service_{timestamp}.log"

    return _RUN_LOG_FILE


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Create or return a logger that writes to one shared log file
    for the current app run. A new file is created every restart.
    """
    log_level = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    log_file = _get_run_log_file()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    logger.setLevel(getattr(logging, log_level, logging.INFO))
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False

    return logger