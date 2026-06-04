# Serbo-Croatian (`sr-hr`)

## Datasets
Wikiann (separate SR/HR codes); Polyglot_NER; Babelscape/wikineural

## Recommended models
xlm-roberta-large-finetuned-conll03-english; Babelscape/wikineural-multilingual-ner (non-commercial)

## Annotation scheme
IOB2 — PER, LOC, ORG

## Notes
Covers Serbian (Cyrillic) and Croatian (Latin) simultaneously. Use separate Wikiann codes (sr/hr).
## Evaluation

Models are evaluated using:
- **seqeval** — entity-level precision, recall, F1
- **sklearn** — token-level metrics

See `evaluation.py` for the benchmarking pipeline.

## Benchmarks
See [`benchmarks/sr-hr/`](../benchmarks/sr-hr/) — 3 model evaluation notebook(s).
