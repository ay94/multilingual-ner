# Mandarin Chinese (`zh`)

## Datasets
Wikiann (IOB2)

## Recommended models
xlm-roberta-large-finetuned-conll03-english; ckiplab/bert-base-chinese-ner (ZhWiki/CNA/ASBC/OntoNotes)

## Annotation scheme
IOB2 — PER, LOC, ORG

## Evaluation

Models are evaluated using:
- **seqeval** — entity-level precision, recall, F1
- **sklearn** — token-level metrics

See `evaluation.py` for the benchmarking pipeline.
