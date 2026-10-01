# PR6 — CI/CD & automation on Dagshub

Branch: `ci-cd-automation` → `dev`

## What this adds

| Piece | Path |
| --- | --- |
| CI workflow (3 jobs) | `.github/workflows/ci.yml` |
| CD workflow (publish on `main`) | `.github/workflows/cd.yml` |
| Promotion workflow (gate → human → re-gate) | `.github/workflows/promote.yml` |
| Retrain workflow (Step 6, weekly/dispatch) | `.github/workflows/retrain.yml` |
| ±0.005 metrics drift gate (+ `--ignore` refresh mode) | `scripts/ci_metrics_gate.py` |
| New split metrics output | `reports/split_metrics.json` (from `src/raay/data/split.py`) |
| Pre-commit tool pin parity | `.pre-commit-config.yaml` (ruff `v0.5.6`→`v0.16.4`, mypy `v1.10.1`→`v2.3.1`) |
| Tests | gate 28→35, split 6, ci_workflow 16, retrain_workflow 19 (new) |

## The shared gate (`scripts/ci_metrics_gate.py`)

A script, not `dvc metrics diff`: DVC 3.67's CLI has no `--fail-on-diff` and **swallows load errors** into an empty, exit-0 diff that looks like a clean pass. The script calls the DVC **API**, whose result carries a per-revision `errors` mapping — an unreadable baseline is always a failure, and the base rev is resolved up front so a shallow clone/typo'd `origin/dev` fails loudly.

Verdicts: numeric ≤0.005 PASS; numeric >0.005, added/removed key, non-numeric, or unreadable revision FAIL; whole file with no baseline = NEW (PASS). At ±0.005 the **integer counts match exactly** (40046→40047 fails); only proportions may move.

`--ignore '<regex>'` removes matching metrics from the comparison entirely (`ignored` count in the report). Exists only for the retrain refresh gate (`--ignore '.*_size$'`); default `None` keeps CI byte-identical. A key with *no* baseline (a brand-new label/dialect class) is never ignored and always fails.

Base rev = `git merge-base origin/<base_ref> HEAD` (drift on dev after forking isn't blamed on the PR); manual runs/first push fall back to `HEAD~1`.

## CI on `dev`

```
lint      ruff check . + ruff format --check . + mypy src
test      pytest --cov --junitxml
pipeline  dvc pull data/raw/Final_Data.csv.dvc → dvc repro → dvc status clean
          → ci_metrics_gate.py --threshold 0.005 → PR comment
          → git diff --exit-code dvc.lock
```

`pipeline` needs lint+test, so a lint failure never spends the ~90 s repro. **Fork PRs** skip the DVC/MLflow steps via a *job-level* `if` (`head.repo.full_name == github.repository`) — a per-step `if` on the comment still ran DVC with an empty token and died on an auth error that looks nothing like "not allowed".

### Traps hit wiring CI

1. **`dvc pull <stage>` doesn't pull stage dependencies** — it left `data/raw/Final_Data.csv` missing and `dvc repro` failed. CI pulls the raw CSV by pointer and rebuilds the rest.
2. **`dvc push` can lie**: it uploads only what's in `.dvc/cache`; the cache copy of `Final_Data.csv` was gone, so push said "up to date" while the remote had nothing (fresh clone: `Checkout failed… Is your cache up to date?`). Rehydrated the cache object and pushed the pointer.
3. **Old CML flag style is gone** — `cml comment create <path>` with `REPO_TOKEN`, not the `--comment … --publish …` form. `vega: false` avoids libcairo/libpango.

## CD on `main` (Step 2)

`cd.yml`: `lint` → `test` → `docker-build` (`packages: write`). `main` is a fast-forward of `dev`, so the repro+gate already ran on the PR — CD deliberately does **not** re-run them. `cancel-in-progress: false`.

- **Model comes from DVC, not the registry**: Dagshub MLflow *experiments* work from CI but `mlflow.search_registered_models()` is empty (registry exists only in local `mlflow.db`), so `models:/ArabicSentiment/Production` is unresolvable on a fresh runner. The 4 serving pointers are the only runner-acquirable source: `model_int8.onnx` (136,119,118 B, md5 `687d587004c63610…`) + the 3 `models/baseline/final/*.json` files. `model_version` is derived (`int8-<first 12 of md5>`), never hand-kept, and re-checked with `md5sum`/`stat` before building.
- **The version ships twice**: BentoML `labels:` are *Bento* labels (the template emits no `LABEL`), so `--label` carries OCI metadata for `docker inspect` and `--build-arg RAAY_MODEL_VERSION` the value the service reports on `/health`/`/predict`. Inspect asserts all three + smoke greps both routes.
- **Deploy**: `RAAY_IMAGE=ghcr.io/…:<sha> docker compose pull raay-sentiment && docker compose up -d --no-build` — without `--no-build` compose rebuilds the repo image; never deploy by `latest`.

### Local-rehearsal findings

- **`/health` passing is not evidence the container works**: with only the version baked, `serve.py` still defaulted to the registry alias → `/health` healthy while `/predict` 500'd. `RAAY_ONNX_PATH`/`RAAY_TOKENIZER_DIR` are now declared in the bentofile.
- **`awk '{print $2}'` can't parse a DVC pointer** (`- md5: <hash>` → `$2` is literally `md5:`), which would have produced `int8-md5:` baked into every image. Now splits on the colon + asserts 32 chars.
- The `.gitignore` needs the exact 4-rule `models/baseline/` dance (git won't descend into excluded dirs, DVC refuses to pull a git-ignored pointer), and the `.gitignore` + `.dvc` files must land in one commit.

## Staging with auto-rollback (Step 3)

Fourth CD job (`environment: staging`), logic in `scripts/deploy_staging.py`: pull → health wait → **revision-label cross-check** → 3 Arabic predictions with confidence floors → `{"texts":[1]}` must 422 → only then promote `last_known_good`. Any failure rolls back.

- **The two delete-prone assertions**: the running container's revision label must equal the target sha (else a cached/hand-edited container reports success while old code serves), and the 422 probe (a pydantic bump that drops `list[str]` still looks healthy and returns plausible labels — nothing else would notice). Both verified by breaking them on purpose.
- **Rollback ordering**: state file (outside the checkout at `/var/lib/raay-staging/deploy-state.json`) → the running container's revision label → fail with no rollback (a first deploy has nothing to return to). `latest` is never a fallback.
- **Rehearsal without a VM**: `--registry localhost:5050 --repository <repo> --skip-login` against local `registry:2` + a stand-in image built deliberately broken (`FAKE_MODE=flip|wrongversion|unhealthy`). Exercised flip/schema/unhealthy/lost-state/first-deploy paths.
- **Traps**: the known-hosts guard matched its own prose (comments contain `ssh-ed25519`) and an earlier `^`-anchored pattern rejected every real line — strip comments first; a `zip(CASES, {set})` gave non-deterministic scores → tuple.

## Promotion gate (Step 4)

`scripts/promote_model.py` is the **only code allowed to move `Production`**. A checklist can't do this because the decisions are comparisons — a model predicting `positive` for everything scores 0.85 accuracy (the same headline as the serving model) with macro F1 0.42, so accuracy is not gated on its own; F1, per-class recall, and absolute Neutral/Negative recall floors are.

| gate | compares | default |
| --- | --- | --- |
| `frozen_test_split` | md5 of `data/processed/test.csv` vs `dvc.lock` | exact |
| `full_split_evaluated` | rows scored vs in split | all 7209 |
| `label_order_matches_production` | `id2label` | identical |
| `f1_macro_not_regressed` / `_above_floor` | vs Production / vs `eval_baseline.json` | −0.005 / −0.01 |
| `accuracy_not_regressed` | vs Production | −0.01 |
| `recall_<class>` (3) | vs Production | −0.02 |
| `recall_{neutral,negative}_absolute_floor` | vs constant | 0.10 / 0.50 |
| `latency_p95_within_budget` | vs Production, null-controlled | +10% |
| `model_size_within_limit` | vs Production | no growth |
| `onnx_parity` | `reports/onnx_int8_parity.json` | 1.0 / ≤0.5 |

Decision + every value + thresholds + split md5 → `reports/promotion_<version>.json`. A drifted split or missing artifact is **exit 2 with no report** (a report from the wrong data is worse than none).

- **`floor_tolerance=0.01` is measured, not round**: Production (0.6369) sits 0.0038 *below* `eval_baseline.json` (0.6407), so zero tolerance rejects the model already serving; a test pins both directions.
- **Latency had to be A/B + null-controlled + quiet-session** (three attempts disproved by running the gate against itself): sequential same-file was 21% apart, interleave+rotate 15%, then three series / control gap still misattributed (Production series agreed 4.08% while spread was 14.4%). Noise = spread across all three series. Two default ORT thread pools in one process **spin** and invented 6.3% (single pool 1.2%); `latency_session_options()` disables spinning — spread went 38% → **0.39%**. Pinning `intra_op_num_threads=1` is 3x slower (242 vs 86 ms) — a test fails if tried. `latency_warmup=10`, `latency_runs=50`.
- **The `AutoConfig` pin in `load_graph` is load-bearing**: ORT sessions have no `.config`, so without it the evaluator reads alphabetical id2label (negative/neutral/positive) against ids positive=0/negative=1/neutral=2 — no exception, plausible confusion matrix, labels silently swapped. Tested at the effect level (M11 passed without it).
- **Refusals**: `--eval-limit` can never promote (`head(800)` has too few Neutral); `Candidate` alias is set first but `Production` is the last statement behind every gate; promote refuses multi-graph versions and external-`.onnx.data` graphs.
- **Two jobs, one human**: `gate` measures + `--dry-run`; `promote` (required reviewers) **re-runs** the gate at flip time — approval covers the numbers read, so they must still hold. Triggers: dispatch + weekly, not PR (a PR-triggered promotion runs with a stranger's DVC token). Pulls the `split` *stage* (the test CSV has no pointer of its own).
- **Exit codes**: 0 passed (`passed_not_promoted` for rehearsals), 1 gate failed (model problem), 2 couldn't decide (broken pipeline).

### What mutation testing found after the first green run

- The measuring job **could promote**: `--dry-run` was never passed to `promote()`. Now stops at the alias; report records *why* (`dry_run` vs `no_registry_client`).
- The promote job **would fail at flip time**: it used the gate job's *path* output (dead on another runner). The composite action resolves the graph per job from the *version* — only the version crosses jobs; re-resolving per job would race a promotion landing in between.
- A **drifted split cost 20 minutes**: the frozen-split check now runs *before* any scoring and exits 2.

### Verification

66 hermetic tests (`test_promote_model.py`) + 29 structural (`test_promote_workflow.py`); mutation-checked (26 tool + 22 workflow mutations, all caught). The full-split run independently reproduced `eval_int8.json` to 4 dp (F1 0.6369, acc 0.8503, rec 0.9082/0.8599/0.1230), accepts the current graph, rejects the distilled student. The latency gate run against itself says honest `INCONCLUSIVE` (same-graph noise 38.5%) on this 2-core box instead of blaming the model.

## Retrain hooks (Step 6)

`.github/workflows/retrain.yml` (cron `12 4 * * 1` + dispatch + `repository_dispatch[retrain]`) re-runs `dvc repro` on newly versioned raw data:

```
checkout dev → uv sync → creds→.dvc/config.local → dvc pull Final_Data.csv.dvc
→ dvc repro → dvc status clean → git diff --exit-code dvc.lock
    ≡ unchanged ⇒ notice + exit 0 (cheap no-op)
→ gate --base HEAD --targets reports/split_metrics.json
       --ignore '.*_size$' --threshold $REFRESH_THRESHOLD
    PASS  ⇒ push retrain/data-<date> (dvc.lock + 2 metrics jsons) + PR into dev
    FAIL  ⇒ ::error:: + exit 1
upload reports/data_refresh_* (always) + step summary
```

- **One escape hatch, refresh-only**: ±0.005 makes integer counts exact-match, but sizes/counts *should* move on a refresh — so `--ignore '.*_size$'`. Proportions still compare strictly; a brand-new label/dialect class is **not** ignored and fails. Default `None` keeps CI byte-identical (7 new gate tests).
- **Scheduling note**: GitHub only schedules cron from the default branch (`main` here), so the file lives on `main` but checks out `ref: dev` — data changes land on dev and reach main via normal promotion.
- **PR body** carries the gate table and states CI's own gate will flag the count rows red there by design.
- **`repository_dispatch` is a receiver, no producer yet**: the seam for the nightly Airflow PSI job (no GitHub token on that host); threshold/reason overridable via `client_payload`.
- **Out of scope**: fine-tuning still needs GPU/Kaggle (`scripts/kaggle_train_runs.py`); this job only hands the human a re-split + PR.
- **Housekeeping this PR carried**: the pre-commit ruff **v0.5.6**/mypy **v1.10.1** pins fought the project's 0.16.4/2.3.1 forever, re-reverting 3 committed Step-4 test files on every commit; pins now match.

Verification: 19 structural tests in `test_retrain_workflow.py`; full local gate green (432 collected); live probe of the new flags = clean no-op pass. The workflow itself has never run on a real repo (no `DAGSHUB_TOKEN` on this checkout) — structurally validated, not end-to-end.

## Required repository secrets

- `DAGSHUB_USER` — Dagshub username
- `DAGSHUB_TOKEN` — Dagshub token (**rotate the existing one**: written to plaintext in `docs/commands_run_p6.txt`)

## Known gaps

- Push to ghcr.io and the staging/production environments have never run against a real host/`production` environment (no staging VM, no `production` reviewers, no registered candidate); registry-side graph resolution is the least-tested part.
- First CI run reports the 30 split metrics as NEW and passes; they become a real baseline after merge.
- A full promotion run is ~25 min here (score 7209 rows ×2 + timed series), feels slow on schedule.
- The weekly promote schedule re-checks whatever `Production` points at — a re-gate reminder, not a drift detector (that's the nightly PSI job); the latency gate cannot pass on this 2-core box by design, so it must run on a real runner.
- `.dvc/config` still lists a dead `storage` remote (`core.remote = dagshub` wins; harmless).
