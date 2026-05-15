import os
import sys
from loguru import logger

def setup_logger(bot_name: str, log_dir: str) -> None:
    os.makedirs(log_dir, exist_ok=True)
    logger.remove()
    logger.add(sys.stdout, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{name}</cyan> | {message}")
    logger.add(
        os.path.join(log_dir, f"{bot_name}.log"),
        rotation="1 week",
        retention="1 month",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name} | {message}",
    )
