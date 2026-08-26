
## 1. Project Analysis

### 1.1 Business problem

An Arab e-commerce platform (Noon.com-style, Amazon.eg-style) receives ~50,000 Arabic product reviews/day.
Each review must be classified as **Positive / Negative / Neutral** to power:

- Product ranking
- Seller rating aggregation
- Customer-service ticket prioritization (route negative reviews fast)

### 1.2 ML problem framing

- **Task**: 3-class text classification (multi-class, single-label)
- **Language**: Arabic (MSA + dialects — Egyptian, Gulf, Levantine slang)
- **Input**: Free-text review, variable length, noisy (emojis, Arabizi, elongated letters, diacritics inconsistently used)
- **Output**: `{positive, negative, neutral}` + confidence score
- **Scale**: 50k reviews/day ≈ 0.6 req/s average, but with peaks (flash sales, Ramadan, White Friday) — batch + real-time both matter

### 1.3 Why this project is MLOps-rich

| Concern                     | How it shows up here                                                                                       |
| --------------------------- | ---------------------------------------------------------------------------------------------------------- |
| GPU optimization            | TensorRT + Triton needed to hit latency/cost targets at 50k/day scale                                      |
| Model compression           | 12-layer AraBERT teacher → 6-layer student distillation                                                   |
| Data/concept drift          | Ramadan vocabulary, new slang, dialect shifts over time                                                    |
| Multiple inference patterns | Batch (nightly re-scoring), online/real-time (new review submitted), and streaming (queue-based) all apply |
| Data versioning             | Arabic dialect coverage changes — need reproducible datasets                                              |
| Experiment tracking         | Comparing teacher vs. student, different dialects, tokenizers                                              |

### 1.4 Success metrics

- **Offline**: Macro-F1 ≥ target (weight Neutral class carefully — it's usually the weakest class), per-dialect F1 breakdown
- **Online**: Business proxy metrics — agreement with manual QA sample, negative-review escalation latency
- **Systems**: p95 inference latency, throughput (reviews/sec), GPU utilization, cost per 1k reviews
- **Drift**: weekly PSI/KL-divergence on embedding distribution, vocabulary novelty rate

### 1.5 Constraints & risks

- GPU-dependent (TensorRT build is hardware/driver-specific — pin CUDA/TensorRT versions)
- Arabic dialectal variance → risk of poor generalization to underrepresented dialects
- Label noise (3-way sentiment is subjective, especially Neutral vs. mildly Positive/Negative)
- Seasonal vocabulary drift (Ramadan, Eid, sales events)
