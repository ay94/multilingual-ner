# Hausa (`ha`)

## Datasets
Hausa VOA NER (news corpus); MasakhaNER

## Recommended models
xlm-roberta-large-finetuned-conll03-english; mbeukman/xlm-roberta-base-finetuned-hausa-finetuned-ner-hausa

## Annotation scheme
IOB — PER, LOC, ORG

## Notes
Low-resource language. MasakhaNER models preferred.
## Evaluation

Models are evaluated using:
- **seqeval** — entity-level precision, recall, F1
- **sklearn** — token-level metrics

See `evaluation.py` for the benchmarking pipeline.
