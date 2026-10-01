
# Raay — Arabic Review Sentiment: Label Guidelines

**Current version: `v1.1`.** The `guideline_version` column on every labeled row records which version produced it, so a labeling-process change stays distinguishable from data drift.

## 1. Labels

The class order is the one every graph in this repo uses, **derived from `LABELS` in `src/raay/enums/constants.py` and `labels:` in `configs/train.yaml`** — both `["positive", "negative", "neutral"]`, i.e. `positive=0`, `negative=1`, `neutral=2`:

| Label    | Code  | Definition                                                                                                                                                   |
| -------- | ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Positive | `0` | Reviewer is overall satisfied with the product, seller, or delivery. Praise outweighs complaints.                                                            |
| Negative | `1` | Reviewer is overall dissatisfied. Complaints outweigh praise.                                                                                                |
| Neutral  | `2` | Purely factual or descriptive text, mixed sentiment with no clear overall lean, or sentiment unrelated to the product itself (e.g., "Delivered on Tuesday"). |

> **v1.0 → v1.1 correction.** This table previously read Positive=2 / Neutral=1 / Negative=0. That was wrong, and the mistake was live: `scripts/promote_model.py` has a `label order vs Production` gate precisely because an ONNX session carries no config, so a consumer that falls back to alphabetical id2label gets `(negative, neutral, positive)` against the real `(positive, negative, neutral)` — no exception, a plausible confusion matrix, Negative and Neutral silently swapped. Any code or spreadsheet built from the v1.0 table inherits that swap. **The order is read from the constants module, never retyped from this table**; if the two ever disagree, `constants.py` wins and this table is the bug.

Keep the Arabic UI-facing strings (`سلبي` / `محايد` / `إيجابي`) in a separate display-mapping module, not in the label column. `raay.serving.feedback_service` accepts the **string** labels only and rejects integers, so a v1.0-era integer posting cannot silently mean a different class.

## 2. Scope of "sentiment"

Label the sentiment **toward the product/purchase experience**, not toward the platform, unless the review is explicitly about the platform (e.g. "Noon's app crashes") — in that case still label by the reviewer's expressed satisfaction.

## 3. Edge cases (resolve consistently)

- **Sarcasm** ("رائع، وصل مكسور 👏"): label by true intent (Negative here), not literal words.
- **Mixed reviews** ("المنتج حلو بس التوصيل تأخر كتير"): label by the *dominant* clause; if truly balanced, use Neutral.
- **Dialect & code-switching** (Egyptian, Gulf, Levantine, franco-arabe like "كويس sh no مش عارف"): label normally; do not down-weight dialectal text.
- **Ramadan / seasonal vocabulary** ("مناسب للسحور", "هدية العيد حلوة"): treat as normal positive/negative signal, not a special class — ~~but tag the row with `season=ramadan` in metadata for drift monitoring (see Phase 4)~~ — **not implemented**; no column carries `season`. Recorded here so the promise is not read as shipped.
- **Star rating vs. text mismatch** (5 stars, negative text or vice versa): trust the **text**, not the star rating. Flag mismatches (`rating_text_conflict=true`) for adjudication review.
- **Empty / non-Arabic / emoji-only reviews**: exclude from the labeled set; route to a `filtered_out` bucket with reason code.
- **Short reviews** (<3 tokens, e.g. "تمام", "زبالة"): still labelable — lexical polarity is usually unambiguous. Label normally.

## 4. Annotation process

1. Each review labeled independently by **2 annotators**.
2. Disagreements go to a **3rd senior adjudicator**; adjudicator's label is final.
3. Track inter-annotator agreement with **Cohen's Kappa**; target κ ≥ 0.75. Below that, guidelines are ambiguous — revise this doc, not just retrain annotators.
4. Re-annotate a random 5% sample every batch as a QA spot-check.

## 5. Class balance target

- **Measured, not targeted.** The corpus is 57.6% positive / 37.3% negative / 5.1% neutral, identical across train/val/test (`reports/split_metrics.json` → `label_proportions`). That is the reference: `raay.inference.prediction_drift` reads the prior from `data/processed/train.csv` and PSIs the daily class mix against it, so the *measured* prior is what the drift gate compares to.
- **The original 45/35/20 target was a goal, not an observation, and it was never applied.** It also turns out to be actively harmful as a *reference*: PSI of a clean panel against 45/35/20 is **0.41 (FAIL)**, while the same panel against the real prior is **0.021 (PASS)**. A gate built on the target would fail every night on a healthy model. Both numbers are pinned by `test_clean_panel_passes_against_the_real_prior` (0.0131) and `test_brief_prior_45_35_20_would_fail_a_clean_panel` (0.3726). **Do not "correct" the reference back to 45/35/20.**
- The ~0.02 floor is the model's own Neutral weakness, not drift: it predicts Neutral 2.9% against a 5.1% prior (recall 0.139), so PSI against the true prior is never 0. Do not tighten the threshold below it.
- The residual oversupply is **not** fixed by oversampling synthetic rows (duplication, not signal). Remaining levers are targeted collection of genuinely Neutral review text, or accepting the prior and reporting recall honestly.
- Do not artificially balance the **eval/test sets** — they stay representative so metrics reflect real traffic; only the training set could ever be rebalanced.
- Track and report class balance per data batch in `docs/data_batches.md`.

## 7. Customer-service overrides (Phase 6 step 4)

A support agent's correction is a human opinion formed under time pressure during a dispute, and disputes are often about shipping, refunds or seller conduct rather than sentiment — exactly the scope violation section 2 warns about. So an override is treated as a **proposal**, never as ground truth. Full mechanics in `AGENTS.md`; the policy:

1. **Two distinct agents must agree** on the same review, or an **adjudicator** must have ruled. Nothing trains on a single assertion — a mislabeled override is worse than no label.
2. **Corroboration counts distinct agents, not rows.** An agent who double-posts cannot corroborate themselves, and an agent who asserts two different labels for one review is dropped as self-contradictory rather than counted as a vote.
3. **One review yields one training row.** The second assertion is *evidence for the label*, recorded in `corroborated_by` — not a second copy of the example, which would weight one hard negative 2× for no informational reason.
4. **A confirmation is never training data.** When `model_label == corrected_label` the "corrected" label is the model's own prediction; feeding it back would train the model on itself. Confirmations are the *denominator* of `production_error_rate` instead.
5. **Routing, never gating.** `suspicious_model_score` (0.9) sends a correction to `route=adjudicate_first` — a model that was confident *and wrong* is the hardest case worth collecting, so confidence must not be able to discard a row.
6. **Neutral overrides are capped** at 50 per batch (`max_neutral_per_batch`). CS disputes skew to Neutral and Neutral is already the model's weakest class (recall 0.139); an uncapped batch would skew the training prior away from the reference in section 5.
7. **Test-split overlap is rejected** as `leak`, fuzzily, after the same normalization the corpus got. `promote_model.py` hashes `test.csv` but cannot know the *labels* were also seen, so the merge is the only guard against train-on-test.
8. Adjudicated labels are recorded with the guidelines version that was in force (§6), and the `season=ramadan` row tag promised in §3 is still unimplemented — no row in this corpus carries a timestamp, so whether firing the seasonal retrain trigger before Ramadan helped is unknowable here.

## 6. Versioning

- Guidelines are versioned (`v1.0`, `v1.1`, ...). Any change that could alter existing labels triggers a re-audit of a sample from prior batches.
- Every labeled batch records the guideline version used (`guideline_version` column) so drift in the *labeling process itself* is distinguishable from drift in the *data*.
