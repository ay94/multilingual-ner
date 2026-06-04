# Czech (`cs`)

## Datasets
Wikiann; Polyglot_NER

## Recommended models
xlm-roberta-large-finetuned-conll03-english

## Annotation scheme
IOB2 — PER, LOC, ORG

## Evaluation

Models are evaluated using:
- **seqeval** — entity-level precision, recall, F1
- **sklearn** — token-level metrics

See `evaluation.py` for the benchmarking pipeline.

## Benchmarks
See [`benchmarks/cs/`](../benchmarks/cs/) — 2 model evaluation notebook(s).
