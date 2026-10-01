# PR #7 — Monitoring & drift response

Phase 6 makes the model *notice* the world moved, then *do something*: an input drift gate over engineered features, a prediction drift gate that names which half moved, a retrain trigger that dispatches to GitHub, and a CS feedback loop that turns corrections into a train-only sidecar.

Every number here was measured. The most useful results are negative — **three of the brief's assumptions did not survive contact with the data**, and each one changed the design:

| Assumption                                          | Reality                                                                                | Consequence                                                                                                                                                                                                     |
| --------------------------------------------------- | -------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| class prior is 45/35/20                             | **57.6/37.3/5.1**, identical across all splits                                   | clean panel scores PSI**0.41 (FAIL)** against the brief vs **0.021 (PASS)** against the real prior — the gate would have failed nightly on a healthy model. Read from `train.csv`, not hardcoded |
| rising`oov_rate` is the earliest new-slang signal | **exactly 0.0** for all 1000 reference rows *and* the injected new-slang panel | AraBERT's 64k WordPiece vocab covers this corpus, franco-Arabizi included; nothing reaches`[UNK]`. Reports `SKIPPED` with the reason; `oov_bucket` is gated instead                                       |
| any reference panel will do                         | **size sets the noise floor**                                                    | at 200 rows the tail PCs false-alarmed at**0.20/0.32/0.22**; at 1000 the same three read **0.024/0.042/0.037**. pc8–pc10 carry ~1.3% of the variance combined                                      |

Suite went **505 → 703 passing** (198 new), all hermetic.

## What Was Done

**1. Engineered input drift** — `drift_features.py` widens the nightly gate to **17 columns**: `predicted_label`, `positive`, `confidence_score`, `text_length`, `oov_bucket`, `oov_rate`, `dialect_label`, `embedding_pc1..10`. An `AraBertEmbedder` loads the **same** tokenizer+encoder the model was fine-tuned with, and PCA — **fitted once on the reference and frozen** to `pca_basis.joblib`, never refit — projects to 10 components so day-over-day scores are comparable. The encoder loads for **only two modes** (`init-reference`, `drift`); `make-input`/`score` must not pay ~2 GB for a model they never use.

**2. Prediction drift + triage** — `prediction_drift.py` watches the **outputs**: class mix PSI'd against **two** references (the training label prior and the reference panel's own predictions), mean confidence vs the frozen reference, plus a rolling z-score. `classify_triage` names which half moved:

| input   | output | triage             |
| ------- | ------ | ------------------ |
| PASS    | PASS   | `stable`         |
| FAIL    | PASS   | `world_changed`  |
| PASS    | FAIL   | `model_degraded` |
| FAIL    | FAIL   | `ambiguous`      |
| missing | any    | `indeterminate`  |

A missing input report returns `indeterminate` rather than assuming the inputs held. `escalate` (confidence falling **AND** mix rising) stays its own field, never folded into `overall`.

**3. Retrain trigger** — `retrain_trigger.py` turns both reports into a decision, precedence **`manual` > `psi_breach` > `scheduled` > `none`**, logging `trigger_reason` to `raay_batch`. `configs/seasonal_events.yaml` holds the calendar; `dispatch_retrain` POSTs the GitHub `dispatches` API with stdlib `urllib.request`, token from file/env **never argv**. `promote_model.py` stamps the reason onto the promoted version **only at the Production flip** — a rejected candidate, a `--dry-run`, or a clean night emits no tags.

**4. Feedback loop** — `feedback_service.py` (standalone Starlette, not BentoML — it is stateful and has no model) takes `POST /feedback`, idempotent on a content hash, and **refuses to start without a token**: an unauthenticated write into the training corpus is a poisoning vector. `--mode review` → `overrides.csv` (the **only** writer; humans type adjudications there) → `--mode merge` → `train_feedback.csv`. Only `corroborated` (≥2 **distinct** agents) and `adjudicated` train.

## Results

**The gate was validated with a null and a positive control — this is what makes it trustworthy.**

| control                                                                | result                                                                                                                  |
| ---------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| **Null** — two independent 1000-row panels vs one reference     | max PSI**0.0416 / 0.0546** — noise floor ~2× below the 0.1 WARN line; FAIL band is ~4× the noise               |
| **Positive** — half the panel replaced with francophone-Arabizi | **all 16 non-`oov_bucket` columns FAIL**; dialect PSI 1.47, arabizi **4.5%→51.5%**, `text_length` 1.84 |

Class-mix PSI across four clean days: 0.0285 / 0.0179 / 0.0117 / 0.021 — ~5× below WARN. **The ~0.02 floor is the model's Neutral weakness, not drift** (predicts 2.9% against a 5.1% prior, recall 0.139), so thresholds stay at 0.1/0.2 — do not tighten below it.

**`--min-history-days` defaults to 7, not 3.** At 3 the four real panels produce a monotone decline and fire at **z = −2.85** on what is provably sampling noise; a 3-point rolling std (0.0025) is a fragile denominator.

**Three deliberate non-firers**, each a false alarm avoided: WARN does not fire (action is at 0.2–0.25); `SKIPPED`/`ERROR` columns do not fire (`oov_rate` → `SKIPPED` is not a breach); unconfirmed calendar events do not fire — Ramadan/Eid are moon-sighting-set, so `confirmed: false` is the arming switch.

**Dispatch token scope, measured not assumed:** a read-only fine-grained PAT → **403** `Resource not accessible by personal access token`; **Contents: read and write** → **204**. GitHub's endpoint page documents only the *classic* `repo` scope.

**Feedback rehearsal** (synthetic day, 4 assertions): 2 corroborating agents on one negative correction + 1 confirmation + 1 lone agent → `review` returns 1 would-be-trainable, `merge` writes **1 row** with `corroborated_by: agent.01;agent.02`. `production_error_rate` overall 0.75, with an unobserved class at `rate: null` — never `0.0`.

## Findings that changed the design

**A 204 does not prove that a workflow ran.** `repository_dispatch` only triggers a workflow present on the **default branch**. The original live probe returned 204 but saw zero runs after two minutes; its accompanying claim that `main` had no `.github/` directory is stale. The current `main` tree contains `.github/workflows/{ci,cd,promote,retrain}.yml`. Their triggers are different: CI runs on `dev` pushes/PRs and manual dispatch; CD runs on `main` pushes and manual dispatch; promote and retrain include schedules and manual dispatch, while retrain also handles `repository_dispatch`. The old zero-run observation does not establish that the current schedules or CD trigger are inert; check the Actions runs for present-day execution status.

`retrain.yml`'s refresh early-exit was **left intact on purpose**: a `psi_breach` means the raw data didn't change, so the refresh is correctly a no-op and forcing past it would reach `git commit` with nothing staged. The receiver is a separate `notify` job (`if: repository_dispatch`, `issues: write`) opening an idempotent `Retrain candidate:` issue, with no `--label` because `gh issue create --label` fails outright if the label is missing. A structural test pins that `notify` cannot retrain.

**Train-only is load-bearing, not a preference.** Merging into `normalized.csv` re-runs `split`, changes `test.csv`, and makes `promote_model.py:check_frozen_split` exit 2 with **no report** — collapsing the 14-gate promotion story. Verified: md5 `5d18bebe1ecdbbcccab6dca9e311630f` unchanged across `dvc repro feedback`.

**Five QA-ladder bugs the tests found, none visible by reading the code.** Each passed a manual spot-check and would have quietly corrupted training data:

1. **Corroboration ate itself.** Text-only dedup marked the *second* agent's assertion a `duplicate`, overwriting `corroborated` — the batch trained on **nothing**. Corroboration needs two rows on one text; dedup was removing the evidence.
2. **One review → two identical training rows**, weighting that hard negative 2× in the loss (5× if five agents flagged it). Corroboration is *evidence for a label*, not extra data. Now one row with every voter in `corroborated_by`, and the Neutral cap counts reviews, not votes.
3. **Adjudication was read per row.** A person adjudicates a *review*, not one assertion — so a group where only the second row carried the ruling merged the ruling **and** the bare corroboration: two training rows for one human decision. Conflicting rulings are now no ruling at all.
4. **`--mode merge` read the raw captures and rewrote `overrides.csv`**, reverting every hand-typed adjudication to `single_agent` on the next DVC run — on a git-tracked, hashed file, so DVC would record the vandalism. Two bugs from one line: unreproducible from declared deps *and* the human step discarded.
5. **`merge_reviewed` overwrote its own history**, dropping every previously validated hard negative (since `train.py` concatenates the whole sidecar). Now appends, ordered by content hash so the file stays byte-stable.

**Smaller ones, same discipline:** the 422 path would have **500'd** (pydantic's `errors()` `ctx` holds a non-serializable `ValueError`) — found by a status-code assertion; `ci.yml`/`retrain.yml` ran a **bare `dvc repro`**, sweeping in `feedback` and churning `dvc.lock` under a metrics table explaining nothing, now scoped to `preprocess split`; the sixth Airflow task was an **island** that could merge while the drift chain had failed; `docs/labeling_guidelines.md` had the **label encoding inverted** until v1.1 (read Positive=2/Neutral=1/Negative=0; every graph uses positive=0/negative=1/neutral=2) — the exact failure mode `promote_model.py`'s `label order` gate defends against; cold-start metrics are **omitted** rather than logged as a healthy-looking `0.0`; a failed dispatch **exits 1** (an alarm the operator asked for that didn't arrive) while a missing token exits 0.

## Honest Caveats

- **Workflow execution status was not re-verified by the original 204 probe.** The workflow files are present on `main`; GitHub only runs scheduled workflows from the default branch, and actual run status should be checked in Actions. Do not use the historical “no `.github/` on `main`” observation as a current caveat.
- **Both drift panels are seeded draws from `data/processed/test.csv`.** A PASS means "matches the training distribution" and the null floor is optimistic — what is validated is the **mechanism**, not production traffic. Two triage cells (`world_changed`, `model_degraded`) aren't constructible from one pool at all and exist only as a matrix test.
- **The seasonal trigger cannot be validated** — no row carries a timestamp, so whether firing before Ramadan helped is unknowable here.
- **The feedback loop has never run on real traffic.** No CS tool, no agent roster, so every number in `feedback_metrics.json` is a rehearsal; with one agent id, review correctly returns 100% `single_agent` and nothing trains. `train_feedback.csv` ships header-only, so no claim is made about its effect on F1. `production_error_rate` is uninterpretable if the tool posts only disputes. The leakage guard is tested against a synthetic `test.csv`, and rapidfuzz 0.9 won't catch a lexically-rewritten near-duplicate.
- **The rolling confidence baseline needs ~2 weeks** of nightly runs; until then only the frozen-reference delta carries weight. `oov_rate` only becomes live on text genuinely outside this corpus.

## Artifacts

| Piece                         | Path                                                                                                |
| ----------------------------- | --------------------------------------------------------------------------------------------------- |
| Engineered drift              | `src/raay/inference/drift_features.py`                                                            |
| Prediction drift              | `src/raay/inference/prediction_drift.py`                                                          |
| Retrain trigger               | `src/raay/inference/retrain_trigger.py`                                                           |
| Seasonal calendar             | `configs/seasonal_events.yaml`                                                                    |
| Feedback sidecar / QA ladder  | `src/raay/serving/feedback_service.py`, `src/raay/data/feedback.py`                             |
| Nightly DAG (6 chained tasks) | `airflow/dags/raay_nightly_batch_scoring.py`                                                      |
| Dispatch receiver             | `.github/workflows/retrain.yml` (`notify` job)                                                  |
| Frozen basis + reference      | `data/scoring/reference/{pca_basis.joblib,reference_engineered.csv}`                              |
| Reports (git-tracked)         | `reports/{drift,prediction_drift,retrain_trigger}/{date}.json`, `reports/feedback_metrics.json` |

198 new tests, all hermetic: `test_feedback` 67 · `test_drift_features` 51 · `test_feedback_service` 32 · `test_prediction_drift` 31 · `test_retrain_trigger` 30 (plus 11 structural DAG, 27 CI-workflow, 24 retrain-workflow). Shared fakes in `tests/conftest.py`.

## How to Verify

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy src && uv run pytest

# step 1 — fit the frozen basis once, then the nightly chain
uv run python -m raay.inference.batch_score --mode init-reference --samples 1000
uv run python -m raay.inference.batch_score --mode make-input --date 2026-10-03 --samples 1000
uv run python -m raay.inference.batch_score --mode score      --date 2026-10-03 --min-samples 1000
uv run python -m raay.inference.batch_score --mode drift      --date 2026-10-03

# step 2 — outputs, against both the panel and the train.csv label prior
uv run python -m raay.inference.batch_score --mode predict-drift --date 2026-10-03

# step 3 — the decision; --force-reason manual exercises the dispatch path
uv run python -m raay.inference.retrain_trigger --date 2026-10-03

# step 4 — merge must NOT read raw/
uv run python -m raay.data.feedback --mode review --raw data/feedback/raw/2026-10-03.csv \
    --reviewed data/feedback/reviewed/overrides.csv
uv run python -m raay.data.feedback --mode merge --reviewed data/feedback/reviewed/overrides.csv \
    --merged data/processed/train_feedback.csv
```

The invariant that makes step 4 safe: `md5sum data/processed/test.csv` is `5d18bebe…` before and after `uv run dvc repro feedback`.

The full execution record — every negative control and every reproduced bug — is in [`commands_run_p7.txt`](commands_run_p7.txt) (steps 1–3) and [`commands_run_p8.txt`](commands_run_p8.txt) (step 4).
