# Arabic (`ar`)

## Datasets
AQMAR (74k tokens, 28 Wikipedia articles — requires label mapping)

## Recommended models
hatmimoha/arabic-ner (378k tokens, 14k sentences); CAMeL (ANERcorp-based)

## Annotation scheme
CoNLL with custom tags — PER, LOC, ORG, MISC variations

## Notes
Label mapping required between dataset and model annotation schemes.
## Evaluation

Models are evaluated using:
- **seqeval** — entity-level precision, recall, F1
- **sklearn** — token-level metrics

See `evaluation.py` for the benchmarking pipeline.

## Benchmarks
See [`benchmarks/ar/`](../benchmarks/ar/) — 1 model evaluation notebook(s).
