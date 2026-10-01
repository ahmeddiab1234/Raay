"""Consumer-loop and serial-reference tests.

Hermetic (AGENTS.md rule: unit tests only): a fake scorer and the in-process
memory queue stand in for the INT8 graph and Redis.
"""

from consumer_helpers import FakeScorer

from raay.inference.batch_consumer import (
    BatchConsumer,
    MemoryQueue,
    ReviewMessage,
    _serial_reference,
)


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
