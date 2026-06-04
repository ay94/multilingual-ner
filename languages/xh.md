# Xhosa (`xh`)

## Datasets
MasakhaNER2; IsiXhosa NER corpus (South African government domain)

## Recommended models
xlm-roberta-large-finetuned-conll03-english; masakhane/afroxlmr-large-ner-masakhaner-1.0_2.0

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
See [`benchmarks/xh/`](../benchmarks/xh/) — 2 model evaluation notebook(s).
