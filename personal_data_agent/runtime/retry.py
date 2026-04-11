from __future__ import annotations

import random
import time
from typing import Callable, TypeVar

from .errors import ToolExecutionError


T = TypeVar("T")


def with_retry(func: Callable[[], T], max_retries: int = 3, base_delay: float = 0.5) -> T:
    """Retry retriable ToolExecutionError with exponential backoff + jitter."""
    attempt = 0
    while True:
        try:
            return func()
        except ToolExecutionError as exc:
            attempt += 1
            if (not exc.retriable) or attempt > max_retries:
                raise
            sleep_s = (base_delay * (2 ** (attempt - 1))) + random.uniform(0, 0.2)
            time.sleep(sleep_s)
