import logging

from aislib.misc_utils import get_logger as _get_logger


def get_logger(name: str) -> logging.Logger:
    logger = _get_logger(name=name)
    logger.propagate = False

    return logger
