# PR #5 — Serving & Operations

## Summary

This phase turned the fine-tuned model into a running service and put operational controls around it. The BentoML REST service was hardened for real endpoint semantics, baked into a self-contained container, and put behind a compose stack. On top of that, three operational layers were added: a Redis micro-batch consumer for near-real-time scoring, a nightly batch re-scoring job with an Evidently PSI drift gate, and a 95/5 canary rollout wired to the MLflow registry alias.

Serving uses the INT8 ONNX graph at `p50 12.0 ms` per single call on CPU, which met the observed target without a GPU, so TensorRT was deferred.

## What Was Done

### 1. Hardened REST Service

`src/raay/serving/serve.py` gained two middlewares to make the API behave predictably under clients:

- `HealthRouteMiddleware` short-circuits `GET /health` and returns `200 {"status": "healthy"}` directly. It intercepts the path *before* BentoML's Starlette `Mount`, which would otherwise 307-redirect `/health` to `/health/` and break the container healthcheck.
- `Validation422Middleware` rewrites BentoML 1.4.39's pydantic validation bodies from `400` to `422`, matching the documented contract, while leaving genuine `400`s untouched.

Resulting endpoint contract:

| Request                                         | Result                                                                           |
| ----------------------------------------------- | -------------------------------------------------------------------------------- |
| `GET /health`                                 | `200 {"status": "healthy"}` (no redirect)                                      |
| `{"texts": ["…"]}`                           | `200` with per-text `label` + `score`                                      |
| `{"texts": []}`                               | `200 {"predictions": []}` by design (short-circuits before `np.concatenate`) |
| `{"texts": 5}` / `{}` / `{"texts": null}` | `422` with pydantic detail                                                     |

### 2. Containerized Serving

`bentofile.yaml` ships a trimmed serving-only dependency set (CPU torch via `--extra-index-url .../whl/cpu`) and bakes `models/onnx/model_int8.onnx` plus the baseline tokenizer, so the image is self-contained. The compose stack publishes three ports:

| Service                   | Host → Container    | Role                                                    |
| ------------------------- | -------------------- | ------------------------------------------------------- |
| `raay-sentiment`        | `8000` → `3000` | Production worker (INT8)                                |
| `raay-sentiment-canary` | none                 | Canary worker (distilled FP32), reached by service name |
| `raay-nginx`            | `8081` → `8081` | 95/5 canary front                                       |
| `queue-redis`           | `6379` → `6379` | Queue for the micro-batch consumer                      |

**Resolved blocker — src-layout import failure.** The first image died at import with `No module named 'raay'`. BentoML copies include globs into the bento `src/` directory while preserving source paths, so `src/raay/**` landed at `/home/bentoml/bento/src/src/raay`, which is invisible to `PYTHONPATH=/home/bentoml/bento/src`. Fixed with `python.is_src_layout: true` in `bentofile.yaml` (strips the `src/` prefix) plus `PYTHONPATH`, which makes `src/raay/**` resolve to `/home/bentoml/bento/src/raay`. Compose then points `RAAY_ONNX_PATH` / `RAAY_TOKENIZER_DIR` at the baked `/home/bentoml/bento/src/models/**`.

The healthcheck uses python `urllib` with a 60 s `start_period`. Compose deliberately does **not** bind `env_file: .env`, which still holds dead Kaggle paths.

### 3. Near-Real-Time Micro-Batch Consumer

`src/raay/inference/batch_consumer.py` polls the Redis `reviews` list, drains up to `--max-batch 32` or `--max-wait-ms 100` (whichever fires first), and scores the whole micro-batch in **one** in-process INT8 ORT call, publishing `{"id"?, "text", "label", "score"}` to `reviews-results`. Batching is deliberately client-side — there is no `batchable=True` in `serve.py`.

### 4. Nightly Batch Re-Scoring & Drift Gate

`src/raay/inference/batch_score.py` re-scores a daily panel and gates on distribution drift:

| Mode               | Reads                          | Writes                                                                                         |
| ------------------ | ------------------------------ | ---------------------------------------------------------------------------------------------- |
| `init-reference` | `data/processed/test.csv`    | `data/scoring/reference/reference.csv` (fixed seed)                                          |
| `make-input`     | `data/processed/test.csv`    | `data/scoring/input/{date}.csv` (seed = CRC32 of the date → idempotent)                     |
| `score`          | the day's input                | `data/scoring/output/{date}.csv` (3-class probs + `predicted_label` / `predicted_score`) |
| `drift`          | reference vs. the day's output | `reports/drift/{date}.json`                                                                  |

The gate is Evidently PSI over `predicted_label` and `positive`: `< 0.1` PASS, `< 0.2` WARN, `≥ 0.2` FAIL, with the overall verdict taken as the worst column. Every mode logs to the `raay_batch` MLflow experiment.

### 5. Airflow Orchestration

`airflow/dags/raay_nightly_batch_scoring.py` runs daily at **03:00 UTC** (`catchup=False`) as three `BashOperator`s shelling out to `python -m raay.inference.batch_score`: `materialize_daily_input → score_daily_batch → run_drift_check`.

Airflow is intentionally **host-isolated and not a project dependency** (`uv tool install "apache-airflow==2.10.5"`), keeping the served image lean. `LocalExecutor` is hard-blocked on SQLite, so the runtime uses `SequentialExecutor`; the executor is only changed while the scheduler is stopped, because the scheduler's own `/proc/<pid>/environ` — not the config file — is the source of truth at runtime.

### 6. Canary Rollout

`deploy/nginx_canary.conf` adds an nginx front on `:8081` splitting traffic **95/5** at the request level between the two workers. `scripts/canary_promote.py` provides `declare` (register the canary graph, idempotent), `promote`, and `rollback`; both mutating modes health-gate `:8000` and `:8081` before flipping the `Production` alias with `archive_existing_versions=True`.

Both variants live under the **same** registered model — the canary worker binds `distilled.onnx` + its external `.onnx.data` + `models/distilled/final` read-only, so the alias acts as the promotion gate rather than the source of the 5 % graph. The canary graph is a real second model, not theater: its logits genuinely differ from INT8 on the same input.

## Results

**Serving latency** (`reports/serving_benchmark.json`, 500 samples, single call):

| Batch |        p50 |        p95 | p50 per item |
| ----: | ---------: | ---------: | -----------: |
|     1 |    12.0 ms |    35.9 ms |      12.0 ms |
|    32 | 1 845.7 ms | 2 675.6 ms |      57.7 ms |

**Micro-benchmark** (`reports/queue_benchmark.json`, 128 reviews, batch 32, 20 ms modeled overhead):

| Metric                           |             Value |
| -------------------------------- | ----------------: |
| Micro-batches                    | 4 (avg fill 32.0) |
| Call amortization                |            32.0× |
| Speedup vs serial, pure CPU      |            0.25× |
| Speedup vs serial, with overhead |            0.50× |

**Drift verdicts:**

| Date       | Verdict | `predicted_label` PSI | `positive` PSI |
| ---------- | ------- | ----------------------: | ---------------: |
| 2026-09-23 | PASS    |                  0.0003 |           0.0193 |
| 2026-09-24 | PASS    |                  0.0034 |           0.0348 |

## Honest Caveats

- **Micro-batching does not beat serial scoring on this box.** INT8 is latency-optimal at batch 1 here (12.0 ms vs 57.7 ms per review at batch 32), so `pure_cpu` speedup of 0.25× is expected, not a bug. Batching earns its keep through *call amortization* (128 reviews → 4 `session.run` calls) and would win outright on batch-throughput-bound backends (GPU / TensorRT) or with high per-call HTTP overhead.
- **The canary latency signal is not real.** Both workers share 2 cores, so the 5 % slice's latency sawtooth is CPU contention, not a canary signal. The behavioral gate is the nightly PSI drift job, not nginx latency. Weights are request-level, and promotions land on worker restart via alias resolution.
- **The canary path is not exercised live in CI.** `tests/test_canary_nginx.py` parses the config as text and drives the promote/rollback logic against a fake MLflow client; there is no `nginx -t` in the suite, so a syntactically valid-but-wrong config would not be caught.
- **Airflow runs single-worker.** `SequentialExecutor` executes tasks in the scheduler process — fine for a nightly job, not a throughput pattern.

## Artifacts

- `src/raay/serving/serve.py`, `src/raay/inference/batch_consumer.py`, `src/raay/inference/batch_score.py`
- `bentofile.yaml`, `docker-compose.yml`, `deploy/nginx_canary.conf`, `scripts/canary_promote.py`
- `airflow/dags/raay_nightly_batch_scoring.py`
- `reports/serving_benchmark.json`, `reports/queue_benchmark.json`, `reports/locust_{fp32,int8}_stats.csv`
- `reports/drift/2026-09-23.json`, `reports/drift/2026-09-24.json`
- `tests/test_serving.py` (13), `tests/test_batch_consumer.py` (20), `tests/test_batch_score.py` (14), `tests/test_canary_nginx.py` (14) — 61 of the 96 hermetic tests

## How to Verify

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy src && uv run pytest

# service contract (dev worker, registry-resolved Production alias)
uv run bentoml serve src/raay/serving/serve.py:svc --port 3000
curl -s -w " HTTP %{http_code}\n" http://127.0.0.1:3000/health
curl -s -H 'Content-Type: application/json' -d '{"texts": 5}'      http://127.0.0.1:3000/predict   # 422
curl -s -H 'Content-Type: application/json' -d '{"texts": []}'     http://127.0.0.1:3000/predict   # 200 []
curl -s -H 'Content-Type: application/json' -d '{"texts":["الجودة ممتازة"]}' http://127.0.0.1:3000/predict

# container + canary
uv run bentoml build -f bentofile.yaml
uv run bentoml containerize raay-sentiment:latest --image-tag raay-sentiment:latest
docker compose up -d && docker compose ps
curl -s http://localhost:8000/health && curl -s http://localhost:8081/health

# drift gate
uv run python -m raay.inference.batch_score --mode drift --date 2026-09-24
```
