# Urdu (`ur`)

## Datasets
Wikiann

## Recommended models
xlm-roberta-large-finetuned-conll03-english

## Annotation scheme
IOB2 — PER, LOC, ORG

## Notes
Low resource language. Limited model options available.
## Evaluation

Models are evaluated using:
- **seqeval** — entity-level precision, recall, F1
- **sklearn** — token-level metrics

See `evaluation.py` for the benchmarking pipeline.
