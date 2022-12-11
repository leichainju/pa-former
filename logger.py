

import sys
import logging
import functools
from termcolor import colored


@functools.lru_cache()
def build_logger(log_file, name='', console_out=True):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # logger format
    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    green_term = colored('[%(asctime)s %(name)s]', 'green')
    yellow_term = colored('(%(filename)s %(lineno)d)', 'yellow')
    color_fmt = green_term + yellow_term + ': %(levelname)s %(message)s'

    # for console output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if console_out else logging.CRITICAL)
    console_handler.setFormatter(
        logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(console_handler)

    # for file output
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    return logger
