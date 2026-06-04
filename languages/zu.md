# Zulu (`zu`)

## Datasets
IsiZulu NER corpus (South African government domain); MasakhaNER2

## Recommended models
masakhane/afroxlmr-large-ner-masakhaner-1.0_2.0

## Annotation scheme
IOB2 / MasakhaNER format

## Notes
MasakhaNER2 dataset and Masakhane model recommended.
## Evaluation

Models are evaluated using:
- **seqeval** — entity-level precision, recall, F1
- **sklearn** — token-level metrics

See `evaluation.py` for the benchmarking pipeline.

## Benchmarks
See [`benchmarks/zu/`](../benchmarks/zu/) — 2 model evaluation notebook(s).
