# Farsi (`fa`)

## Datasets
Wikiann; Persian_NER (7,682 sentences, 250k tokens, 3-fold split)

## Recommended models
xlm-roberta-large-finetuned-conll03-english; HooshvareLab/albert-fa-zwnj-base-v2-ner (ARMAN/PEYMA/WikiANN)

## Annotation scheme
IOB — PER, LOC, ORG

## Evaluation

Models are evaluated using:
- **seqeval** — entity-level precision, recall, F1
- **sklearn** — token-level metrics

See `evaluation.py` for the benchmarking pipeline.
