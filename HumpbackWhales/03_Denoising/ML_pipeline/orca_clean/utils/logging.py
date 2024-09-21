"""
Module: logging.py
Authors: Christian Bergler
License: GNU General Public License v3.0
Institution: Friedrich-Alexander-University Erlangen-Nuremberg, Department of Computer Science, Pattern Recognition Lab
Last Access: 26.04.2022
"""

import logging
import logging.handlers
import os
import queue


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

"""
Logger in order to track training, validation, and testing of the network
"""
class Logger(metaclass=Singleton):
    def __init__(self, name, debug=False, log_dir=None, do_log_name=False):
        level = logging.DEBUG if debug else logging.INFO
        fmt = "%(asctime)s"
        if do_log_name:
            fmt += "|%(name)s"
        fmt += "|%(levelname).1s|%(message)s"
        formatter = logging.Formatter(fmt=fmt, datefmt="%H:%M:%S")

        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        handlers = [sh]

        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
            fs = logging.FileHandler(os.path.join(log_dir, name + ".log"))
            fs.setFormatter(formatter)
            handlers.append(fs)

        self._queue = queue.Queue(1000)
        self._handler = logging.handlers.QueueHandler(self._queue)
        self._listener = logging.handlers.QueueListener(self._queue, *handlers)

        self._logger = logging.getLogger(name)
        self._logger.setLevel(level)
        self._logger.addHandler(self._handler)
        self._logger.propagate = False
        self._listener.start()

    def close(self):
        self._listener.stop()
        self._handler.close()

    def debug(self, msg, *args, **kwargs):
        self._logger.debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self._logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self._logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self._logger.error(msg, *args, **kwargs)