import logging
import sys

def setup_logger(name=None, log_file="app.log"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.hasHandlers():  # <- QUAN TRỌNG
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s - %(name)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.encoding = "utf-8"

        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        logger.propagate = False

    return logger

logger=setup_logger("video2text-ocr")