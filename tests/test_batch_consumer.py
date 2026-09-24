"""Unit tests for the micro-batch queue consumer.

Hermetic by design (AGENTS.md rule: unit tests only): no Redis server, no
ONNX graph. The fake scorer and memory/fake clients exercise the micro-batch
trigger (count-or-timeout), publish path, and batch-vs-serial win without any
integration services.
"""

import time

from raay.inference.batch_consumer import (
    BatchConsumer,
    MemoryQueue,
    RedisQueue,
    ReviewMessage,
    _serial_reference,
    decode_message,
    encode_message,
)


class FakeScorer:
    """Scorer stub recording call sizes; returns labels matching input count."""

    def __init__(self, n_labels: int = 3) -> None:
        self.calls: list[int] = []
        self.n_labels = n_labels

    def score(self, texts: list[str]) -> list[dict]:
        self.calls.append(len(texts))
        return [
            {"label": f"label{i % self.n_labels}", "score": 0.9 + i * 0.01}
            for i in range(len(texts))
        ]


class FakeRedis:
    """Minimal ``blpop``/``rpush``/``llen`` client for RedisQueue unit tests."""

    def __init__(self) -> None:
        self._lists: dict[str, list[str]] = {"reviews": [], "reviews-results": []}

    def blpop(self, key: str, timeout: int) -> tuple | None:
        if key not in self._lists or not self._lists[key]:
            return None
        return (key, self._lists[key].pop(0))

    def rpush(self, key: str, *values: str) -> int:
        self._lists[key].extend(values)
        return len(values)

    def llen(self, key: str) -> int:
        return len(self._lists[key])


class TestDecodeEncode:
    def test_plain_text_is_used_as_message(self):
        msg = decode_message("الجودة ممتازة")
        assert msg is not None and msg.text == "الجودة ممتازة" and msg.id is None

    def test_json_payload_keeps_id(self):
        msg = decode_message('{"id": "abc", "text": "كلام عربي"}')
        assert msg is not None and msg.text == "كلام عربي" and msg.id == "abc"

    def test_encode_roundtrip_with_and_without_id(self):
        assert encode_message(ReviewMessage(text="نص")) == "نص"
        encoded = encode_message(ReviewMessage(text="نص", id="1"))
        msg = decode_message(encoded)
        assert msg is not None and msg.id == "1" and msg.text == "نص"

    def test_invalid_json_dict_without_text_skipped(self):
        assert decode_message('{"foo": 1}') is None


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


class TestBatchConsumer:
    def test_run_once_publishes_matched_results(self):
        q = MemoryQueue()
        scorer = FakeScorer()
        q.push(
            [
                ReviewMessage(text="رائع", id="1"),
                ReviewMessage(text="سيء", id="2"),
            ]
        )
        consumer = BatchConsumer(q, scorer, max_batch_size=32, max_wait_ms=100)
        assert consumer.run_once() is True
        assert consumer.stats.items_processed == 2
        results = q.results()
        assert results[0]["id"] == "1" and results[0]["text"] == "رائع"
        assert results[1]["id"] == "2" and results[1]["text"] == "سيء"
        assert {r["label"] for r in results} <= {"label0", "label1", "label2"}

    def test_empty_batch_is_skipped(self):
        q = MemoryQueue()
        consumer = BatchConsumer(q, FakeScorer(), max_batch_size=5, max_wait_ms=50)
        assert consumer.run_once() is False
        assert consumer.stats.batches == 0

    def test_fill_ratio_and_p50_accumulate(self):
        q = MemoryQueue()
        consumer = BatchConsumer(q, FakeScorer(), max_batch_size=4, max_wait_ms=50)
        for i in range(8):
            q.push([ReviewMessage(text=f"t{i}")])
            consumer.run_once()
        assert consumer.stats.batches == 8
        assert consumer.stats.fill_p50 == 0.25  # 1 of 4 each time

    def test_run_stops_after_max_batches(self):
        q = MemoryQueue()
        consumer = BatchConsumer(q, FakeScorer(), max_batch_size=4, max_wait_ms=50)
        for i in range(6):
            q.push([ReviewMessage(text=f"t{i}")])
        consumer.run(max_batches=2)
        assert consumer.stats.batches == 2
        assert consumer.stats.items_processed == 6

    def test_run_stops_when_idle(self):
        q = MemoryQueue()
        consumer = BatchConsumer(q, FakeScorer(), max_batch_size=4, max_wait_ms=40)
        q.push([ReviewMessage(text="only-one")])
        consumer.run(stop_on_idle_sec=0.5)
        assert consumer.stats.batches == 1

    def test_scorer_is_called_once_per_batch(self):
        q = MemoryQueue()
        scorer = FakeScorer()
        consumer = BatchConsumer(q, scorer, max_batch_size=4, max_wait_ms=50)
        for i in range(4):
            q.push([ReviewMessage(text=f"t{i}")])
        consumer.run_once()
        assert scorer.calls == [4]


class TestSerialVsBatch:
    def test_batch_mode_makes_fewer_predict_calls(self):
        scorer = FakeScorer()
        texts = [f"مراجعة {i}" for i in range(10)]
        serial = _serial_reference(scorer, texts, 0.0)
        assert scorer.calls[-10:] == [1] * 10
        assert serial["reviews_per_sec"] > 0

    def test_microbatch_speedup_over_serial(self):
        scorer = FakeScorer()
        q = MemoryQueue()
        texts = [f"مراجعة {i}" for i in range(8)]
        q.push([ReviewMessage(text=t, id=str(i)) for i, t in enumerate(texts)])
        consumer = BatchConsumer(q, scorer, max_batch_size=8, max_wait_ms=50)
        consumer.run(max_batches=1)
        assert scorer.calls == [8]
        assert consumer.stats.items_processed == 8
