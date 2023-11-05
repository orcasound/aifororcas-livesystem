"""
Module: checkpoints.py
Authors: Christian Bergler
License: GNU General Public License v3.0
Institution: Friedrich-Alexander-University Erlangen-Nuremberg, Department of Computer Science, Pattern Recognition Lab
Last Access: 26.04.2022
"""

import os
import glob
import queue
import threading

import torch


"""
Checkpoint handling functionality (save, store, and load checkpoints)
"""
class CheckpointHandler:
    def __init__(
        self,
        checkpoint_dir: str,
        prefix: str = "",
        max_checkpoints: int = 5,
        logger=None,
    ):
        self.max_checkpoints = max_checkpoints
        self.checkpoint_dir = checkpoint_dir
        if self.checkpoint_dir is None:
            return
        if not prefix.endswith("_"):
            prefix += "_"
        self.prefix = prefix
        if not os.path.isdir(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self._logger = logger
        if self._logger is not None:
            self._logger.debug("Starting checkpoint writer thread")
        self._queue = queue.Queue()
        self._should_stop = threading.Event()
        self._worker = threading.Thread(target=self._write_worker, daemon=True)
        self._worker.start()

    def _write_worker(self):
        while not self._should_stop.is_set():
            try:
                checkpoint_dict = self._queue.get()
                if checkpoint_dict is None:
                    break
                checkpoint_name = os.path.join(
                    self.checkpoint_dir,
                    "{}epoch_{:05d}.checkpoint".format(
                        self.prefix, checkpoint_dict["trainState"]["epoch"]
                    ),
                )
                if self._logger is not None:
                    self._logger.debug(
                        "Writing checkpoint_dict to {}".format(checkpoint_name)
                    )
                torch.save(checkpoint_dict, checkpoint_name)
                checkpoints = glob.glob(
                    os.path.join(self.checkpoint_dir, self.prefix + "*.checkpoint")
                )
                if len(checkpoints) > self.max_checkpoints:
                    checkpoints.sort()
                    for i in range(len(checkpoints) - self.max_checkpoints):
                        os.remove(checkpoints[i])
            except Exception as e:
                if self._logger is not None:
                    self._logger.error(
                        "Failed to write checkpoint to {}.".format(checkpoint_name)
                    )
                    self._logger.error(str(e))
            finally:
                self._queue.task_done()
        if self._logger is not None:
            self._logger.info("Shutting down checkpoint_dict writer thread")

    def write(self, checkpoint_dict: dict, prefix=None):
        if self.checkpoint_dir is None:
            return
        self._queue.put(checkpoint_dict)

    def read_latest(self):
        if self.checkpoint_dir is None:
            return
        checkpoints = glob.glob(
            os.path.join(self.checkpoint_dir, self.prefix + "*.checkpoint")
        )
        if len(checkpoints) == 0:
            self._logger.info("No checkpoints found in {}".format(self.checkpoint_dir))
            return None
        checkpoints.sort()
        if self._logger is not None:
            self._logger.info("Restoring checkpoint {}".format(checkpoints[-1]))
        return torch.load(checkpoints[-1], map_location="cpu")

    def _shutdown_worker(self):
        self._should_stop.set()
        if self._worker.is_alive():
            self._queue.put(None)
            self._worker.join()

    def __del__(self):
        if self._logger is not None:
            self._logger.info("Shutting down CheckpointHandler")
        self._shutdown_worker()
