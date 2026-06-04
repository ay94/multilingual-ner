# Indonesian (`id`)

## Datasets
Wikiann; Polyglot_NER

## Recommended models
xlm-roberta-large-finetuned-conll03-english; cahya/bert-base-indonesian-NER

## Annotation scheme
IOB2 — PER, LOC, ORG

## Evaluation

Models are evaluated using:
- **seqeval** — entity-level precision, recall, F1
- **sklearn** — token-level metrics

See `evaluation.py` for the benchmarking pipeline.
