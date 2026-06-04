# Swahili (`sw`)

## Datasets
Wikiann; MasakhaNER2

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
