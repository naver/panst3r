# Copyright (C) 2025-present Naver Corporation. All rights reserved.

import threading
from contextlib import contextmanager
from itertools import tee

from tqdm import tqdm as _tqdm

class TqdmRouter:
    def __init__(self):
        self._lock = threading.RLock()
        self._wrapper = None

    @contextmanager
    def wrap_tqdm(self, new_fn):
        """
        Temporarily replace tqdm.tqdm with new_fn inside a context.
        Automatically resets on exit.
        """
        with self._lock:
            self._wrapper= new_fn

        try:
            yield
        finally:
            with self._lock:
                self._wrapper = None

    def __call__(self, iterable, total=None, *args, **kwargs):
        it = _tqdm(iterable, total=total, *args, **kwargs)
        if self._wrapper is not None:
            it = self._wrapper(it, total=it.total, *args, **kwargs)

        return it

tqdm = TqdmRouter()
