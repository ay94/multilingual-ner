# Russian (`ru`)

## Datasets
Wikiann; Babelscape/MultiNERD

## Recommended models
xlm-roberta-large-finetuned-conll03-english; Babelscape/wikineural-multilingual-ner

## Annotation scheme
IOB2 — PER, LOC, ORG

## Evaluation

Models are evaluated using:
- **seqeval** — entity-level precision, recall, F1
- **sklearn** — token-level metrics

See `evaluation.py` for the benchmarking pipeline.
