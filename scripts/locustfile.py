"""Locust load-test tasks for the BentoML ``/predict`` endpoint.

Each virtual user POSTs batches of Arabic reviews to the same endpoint shape
the production clients use. Drive it through ``scripts/locust_run.py`` (which
starts/tears down the BentoML server per variant), or run standalone:

    uv run bentoml serve src/raay/serving/serve.py:svc --port 3030 &
    uv run locust -f scripts/locustfile.py --host http://127.0.0.1:3030 \
        --headless -u 8 -r 2 --run-time 90s --html reports/locust.html
"""

from __future__ import annotations

import random

from locust import HttpUser, between, task

_SAMPLE_REVIEWS: tuple[str, ...] = (
    "هذا المنتج ممتاز والجودة عالية جدا",
    "المنتج وصل متأخر والجودة رديئة",
    "الطلبية وصلت بسرعة والحاجة تمام جدا شكرا",
    "حسبي الله ونعم الوكيل ياخي الجودة خايسة",
    "المنتج محايد شكله عادي",
    "الخدمة ممتازة وسعر مناسب لكن التوصيل بطيء",
)


class SentimentUser(HttpUser):
    """Simulates clients sending 3-review classification calls."""

    wait_time = between(0.5, 2.0)

    @task
    def predict(self) -> None:
        self.client.post(
            "/predict",
            json={"texts": [random.choice(_SAMPLE_REVIEWS) for _ in range(3)]},
            headers={"Content-Type": "application/json"},
        )
