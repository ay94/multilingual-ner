# Turkish (`tr`)

## Datasets
Wikiann; Polyglot_NER; Turkish NER (300k entities, 25 domains, auto-annotated)

## Recommended models
xlm-roberta-large-finetuned-conll03-english; akdeniz27/bert-base-turkish-cased-ner

## Annotation scheme
IOB2 — PER, LOC, ORG

## Notes
Auto-annotated Turkish NER dataset may require label alignment.
## Evaluation

Models are evaluated using:
- **seqeval** — entity-level precision, recall, F1
- **sklearn** — token-level metrics

See `evaluation.py` for the benchmarking pipeline.
