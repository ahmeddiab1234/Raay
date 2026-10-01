"""Queue transports for the micro-batch consumer: memory and Redis."""

from __future__ import annotations

import queue
import threading
import time
from typing import Any, Protocol

import redis
from loguru import logger
from redis import exceptions as redis_exceptions

from raay.inference.consumer_messages import (
    ReviewMessage,
    decode_message,
    encode_message,
    encode_result,
)


class QueueBackend(Protocol):
    """A pull queue for inbound reviews plus a result sink."""

    def drain(self, max_items: int, max_wait_ms: int) -> list[ReviewMessage]: ...
    def push(self, messages: list[ReviewMessage]) -> int: ...
    def push_results(self, results: list[dict[str, Any]]) -> int: ...


class MemoryQueue:
    """Thread-safe queue over a Python ``queue.Queue`` (tests + benchmark mode).

    ``drain`` honours the micro-batch trigger exactly: return up to
    ``max_items``, or block for at most ``max_wait_ms`` total, whichever comes
    first.
    """

    def __init__(self) -> None:
        self._input: queue.Queue[str] = queue.Queue()
        self._results: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def drain(self, max_items: int, max_wait_ms: int) -> list[ReviewMessage]:
        deadline = time.monotonic() + max_wait_ms / 1000.0
        out: list[ReviewMessage] = []
        while len(out) < max_items:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                payload = self._input.get(timeout=min(remaining, 0.05))
            except queue.Empty:
                break
            message = decode_message(payload)
            if message is not None:
                out.append(message)
        return out

    def push(self, messages: list[ReviewMessage]) -> int:
        for message in messages:
            self._input.put(encode_message(message))
        return len(messages)

    def push_results(self, results: list[dict[str, Any]]) -> int:
        with self._lock:
            self._results.extend(results)
        return len(results)

    def results(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._results)


class RedisQueue:
    """Redis list transport: ``BLPOP`` input key, ``RPUSH`` output key.

    Elements are strings (see :func:`encode_message`). ``drain`` slices the
    block timeout into ``poll_ms`` windows so a slow trickle of items still
    yields early once ``max_items`` fill the batch without waiting the whole
    window after the deadline.
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379",
        input_key: str = "reviews",
        result_key: str = "reviews-results",
        poll_ms: int = 250,
        client: Any = None,
    ) -> None:
        self._client = (
            client
            if client is not None
            else redis.from_url(url, decode_responses=True, socket_timeout=3.0)
        )
        self._input_key = input_key
        self._result_key = result_key
        self._poll_ms = poll_ms
        self._transient_logged = False

    def drain(self, max_items: int, max_wait_ms: int) -> list[ReviewMessage]:
        deadline = time.monotonic() + max_wait_ms / 1000.0
        out: list[ReviewMessage] = []
        while len(out) < max_items:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            block_ms = max(1, int(min(remaining, self._poll_ms / 1000.0) * 1000))
            try:
                pair = self._client.blpop(self._input_key, timeout=block_ms)
            except (redis_exceptions.TimeoutError, redis_exceptions.ConnectionError):
                # Transient client-side socket timeout over NAT/port-forwarding;
                # retry the same poll window instead of failing the whole drain.
                if not self._transient_logged:
                    logger.warning(
                        "Redis blpop timed out; treating as an empty poll window."
                    )
                    self._transient_logged = True
                time.sleep(0.01)
                continue
            if pair is None:
                continue
            message = decode_message(str(pair[1]))
            if message is not None:
                out.append(message)
        return out

    def push(self, messages: list[ReviewMessage]) -> int:
        if not messages:
            return 0
        return int(
            self._client.rpush(self._input_key, *[encode_message(m) for m in messages])
        )

    def push_results(self, results: list[dict[str, Any]]) -> int:
        if not results:
            return 0
        return int(
            self._client.rpush(self._result_key, *[encode_result(r) for r in results])
        )

    def count_results(self) -> int:
        return int(self._client.llen(self._result_key))
