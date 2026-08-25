
# Raay — Arabic Review Sentiment: Label Guidelines

## 1. Labels

| Label    | Code  | Definition                                                                                                                                                   |
| -------- | ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Positive | `2` | Reviewer is overall satisfied with the product, seller, or delivery. Praise outweighs complaints.                                                            |
| Neutral  | `1` | Purely factual or descriptive text, mixed sentiment with no clear overall lean, or sentiment unrelated to the product itself (e.g., "Delivered on Tuesday"). |
| Negative | `0` | Reviewer is overall dissatisfied. Complaints outweigh praise.                                                                                                |

Encode as integers `0/1/2` in the dataset; keep the Arabic UI-facing strings (`سلبي` / `محايد` / `إيجابي`) in a separate display-mapping module, not in the label column.

## 2. Scope of "sentiment"

Label the sentiment **toward the product/purchase experience**, not toward the platform, unless the review is explicitly about the platform (e.g. "Noon's app crashes") — in that case still label by the reviewer's expressed satisfaction.

## 3. Edge cases (resolve consistently)

- **Sarcasm** ("رائع، وصل مكسور 👏"): label by true intent (Negative here), not literal words.
- **Mixed reviews** ("المنتج حلو بس التوصيل تأخر كتير"): label by the *dominant* clause; if truly balanced, use Neutral.
- **Dialect & code-switching** (Egyptian, Gulf, Levantine, franco-arabe like "كويس sh no مش عارف"): label normally; do not down-weight dialectal text.
- **Ramadan / seasonal vocabulary** ("مناسب للسحور", "هدية العيد حلوة"): treat as normal positive/negative signal, not a special class — but tag the row with `season=ramadan` in metadata for drift monitoring (see Phase 4).
- **Star rating vs. text mismatch** (5 stars, negative text or vice versa): trust the **text**, not the star rating. Flag mismatches (`rating_text_conflict=true`) for adjudication review.
- **Empty / non-Arabic / emoji-only reviews**: exclude from the labeled set; route to a `filtered_out` bucket with reason code.
- **Short reviews** (<3 tokens, e.g. "تمام", "زبالة"): still labelable — lexical polarity is usually unambiguous. Label normally.

## 4. Annotation process

1. Each review labeled independently by **2 annotators**.
2. Disagreements go to a **3rd senior adjudicator**; adjudicator's label is final.
3. Track inter-annotator agreement with **Cohen's Kappa**; target κ ≥ 0.75. Below that, guidelines are ambiguous — revise this doc, not just retrain annotators.
4. Re-annotate a random 5% sample every batch as a QA spot-check.

## 5. Class balance target

- Natural e-commerce review distribution is heavily skewed positive (~70/15/15 or worse).
- Target labeled-set composition: **Positive 45% / Negative 35% / Neutral 20%**, achieved via *stratified oversampling of Negative/Neutral during annotation selection*, not synthetic duplication.
- Do not artificially balance the **eval/test sets** — keep those representative of production traffic so metrics reflect real-world performance; only rebalance the **training set**.
- Track and report class balance per data batch in `docs/data_batches.md` (created in Phase 1).

## 6. Versioning

- Guidelines are versioned (`v1.0`, `v1.1`, ...). Any change that could alter existing labels triggers a re-audit of a sample from prior batches.
- Every labeled batch records the guideline version used (`guideline_version` column) so drift in the *labeling process itself* is distinguishable from drift in the *data*.
