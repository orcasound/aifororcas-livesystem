"""
Module: FileIO.py
Authors: Christian Bergler
License: GNU General Public License v3.0
Institution: Friedrich-Alexander-University Erlangen-Nuremberg, Department of Computer Science, Pattern Recognition Lab
Last Access: 26.04.2022
"""

import os
import queue
import platform

import torch.multiprocessing as mp


def _default_read_fn(fn):
    with open(fn, "rb") as f:
        return f.read()

def _default_write_fn(fn, data):
    with open(fn, "wb") as f:
        f.write(data)

"""
Asynchronous file reader
"""
class AsyncFileReader(object):
    def __init__(self, n_readers=1, read_fn=_default_read_fn, n_retries=3):
        self._read_queue = mp.Queue()
        self._out_queue = mp.Queue()
        self._manager = mp.Manager()
        self._buf = self._manager.dict()

        self._read_fn = read_fn
        self._read_workers = [
            mp.Process(
                target=self.__read_worker,
                args=(self._read_queue, self._out_queue),
                daemon=True,
            )
            for _ in range(n_readers)
        ]

        if platform.system() != "Windows":
            for w in self._read_workers:
                w.start()
        self.n_retries = n_retries

    def __call__(self, file_name):
        self._read_queue.put(file_name)
        n_tries = 0
        while file_name not in self._buf:
            try:
                if n_tries > self.n_retries:
                    return _default_read_fn(file_name)
                fn, data = self._out_queue.get(timeout=.5)
                if fn == file_name:
                    return data
                self._buf[fn] = data
            except queue.Empty:
                n_tries += 1
        return self._buf.pop(file_name)

    def __read_worker(self, in_queue, out_queue):
        while True:
            try:
                fn = in_queue.get()
                if fn is None:
                    print("Stopping read worker with pid", os.getpid())
                    break
                out_queue.put((fn, self._read_fn(fn)))
            except FileNotFoundError:
                out_queue.put((fn, None))
            except (KeyboardInterrupt, SystemExit):
                break
            except Exception:
                import traceback

                print(traceback.format_exc())
                out_queue.put((fn, None))

"""
Asynchronous file writer
"""
class AsyncFileWriter(object):
    def __init__(self, write_fn=_default_write_fn, n_writers=1):
        self._write_queue = mp.Queue()
        self._write_fn = write_fn
        self._write_workers = [
            mp.Process(
                target=self.__write_worker, args=(self._write_queue,), daemon=True
            )
            for _ in range(n_writers)
        ]

        if platform.system() != "Windows":
            for w in self._write_workers:
                w.start()

    def __call__(self, file_name, data):
        self._write_queue.put((file_name, data))

    def __write_worker(self, in_queue):
        while True:
            try:
                fn, data = in_queue.get()
                if data is None:
                    print("Stopping write worker with pid", os.getpid())
                    break
                self._write_fn(fn, data)
            except (KeyboardInterrupt, SystemExit):
                break
            except Exception:
                import traceback

                print(traceback.format_exc())
