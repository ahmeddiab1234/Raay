"""Queue transport tests: the memory trigger and the Redis round-trip.

Hermetic (AGENTS.md rule: unit tests only) -- no Redis server, no ONNX graph.
The micro-batch trigger is count-or-timeout, so the memory queue's
``drain`` timing is asserted directly.
"""

import time

from consumer_helpers import FakeRedis

from raay.inference.batch_consumer import (
    MemoryQueue,
    RedisQueue,
    ReviewMessage,
    encode_message,
)


class TestMemoryQueue:
    def test_drain_returns_full_batch_immediately(self):
        q = MemoryQueue()
        q.push([ReviewMessage(text=f"t{i}") for i in range(3)])
        t0 = time.monotonic()
        items = q.drain(max_items=5, max_wait_ms=1000)
        elapsed = time.monotonic() - t0
        assert len(items) == 3
        assert [m.text for m in items] == ["t0", "t1", "t2"]
        assert elapsed < 0.2

    def test_drain_waits_until_deadline_when_empty(self):
        q = MemoryQueue()
        t0 = time.monotonic()
        items = q.drain(max_items=5, max_wait_ms=120)
        elapsed = time.monotonic() - t0
        assert items == []
        assert 0.03 <= elapsed <= 1.0

    def test_drain_returns_partial_before_deadline(self):
        q = MemoryQueue()
        q.push([ReviewMessage(text="a")])
        t0 = time.monotonic()
        items = q.drain(max_items=3, max_wait_ms=1000)
        elapsed = time.monotonic() - t0
        assert len(items) == 1
        assert elapsed < 0.2

    def test_results_are_published_in_order(self):
        q = MemoryQueue()
        q.push_results([{"text": "a", "label": "pos"}, {"text": "b", "label": "neg"}])
        assert q.results() == [
            {"text": "a", "label": "pos"},
            {"text": "b", "label": "neg"},
        ]

    def test_results_return_list_is_a_copy(self):
        q = MemoryQueue()
        q.push_results([{"text": "a", "label": "pos"}])
        out = q.results()
        out.append({"text": "b", "label": "neg"})
        assert len(q.results()) == 1


class TestRedisQueue:
    def test_push_drain_roundtrip_through_fake_client(self):
        fake = FakeRedis()
        q = RedisQueue(client=fake)
        q.push(
            [
                ReviewMessage(text="نص واحد"),
                ReviewMessage(text="ثاني", id="id-2"),
            ]
        )
        items = q.drain(max_items=10, max_wait_ms=100)
        assert [m.text for m in items] == ["نص واحد", "ثاني"]
        assert items[1].id == "id-2"

    def test_results_land_in_result_list(self):
        fake = FakeRedis()
        q = RedisQueue(client=fake)
        n = q.push_results([{"text": "a", "label": "pos", "score": 0.9}])
        assert n == 1
        assert fake.llen("reviews-results") == 1
        assert "pos" in fake._lists["reviews-results"][0]

    def test_drain_respects_max_items(self):
        fake = FakeRedis()
        q = RedisQueue(client=fake)
        payloads = [encode_message(ReviewMessage(text=f"t{i}")) for i in range(5)]
        fake._lists["reviews"].extend(payloads)
        items = q.drain(max_items=2, max_wait_ms=100)
        assert len(items) == 2
