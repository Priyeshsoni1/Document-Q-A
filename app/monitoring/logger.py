import logging
import sys
from pathlib import Path


def setup_logger() -> logging.Logger:
    """Configure application-wide logging."""

    logger = logging.getLogger("rag_app")

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(name)s | "
            "%(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(
        sys.stdout
    )
    console_handler.setFormatter(formatter)

    # File handler
    log_directory = Path("logs")
    log_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_handler = logging.FileHandler(
        log_directory / "application.log",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


logger = setup_logger()