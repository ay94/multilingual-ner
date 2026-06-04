# Portuguese (`pt`)

## Datasets
Wikiann; HAREM (93k words, 129 texts — requires format conversion)

## Recommended models
xlm-roberta-large-finetuned-conll03-english; Babelscape/wikineural-multilingual-ner (non-commercial)

## Annotation scheme
IOB2 — PER, LOC, ORG

## Notes
HAREM requires conversion to CoNLL format before use.
## Evaluation

Models are evaluated using:
- **seqeval** — entity-level precision, recall, F1
- **sklearn** — token-level metrics

See `evaluation.py` for the benchmarking pipeline.
