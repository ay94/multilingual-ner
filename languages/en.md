# English (`en`)

## Datasets
Wikigold (manually annotated Wikipedia); WNUT 17 (Twitter, Stack Overflow, YouTube, Reddit — requires label mapping)

## Recommended models
dslim/bert-base-NER; xlm-roberta-large-finetuned-conll03-english

## Annotation scheme
CoNLL-2003 — PER, LOC, ORG, MISC

## Evaluation

Models are evaluated using:
- **seqeval** — entity-level precision, recall, F1
- **sklearn** — token-level metrics

See `evaluation.py` for the benchmarking pipeline.
