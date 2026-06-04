# Twi (Ghana) (`tw`)

## Datasets
MasakhaNER2

## Recommended models
masakhane/afroxlmr-large-ner-masakhaner-1.0_2.0

## Annotation scheme
MasakhaNER format

## Notes
Very low resource. MasakhaNER2 is the primary available resource.
## Evaluation

Models are evaluated using:
- **seqeval** — entity-level precision, recall, F1
- **sklearn** — token-level metrics

See `evaluation.py` for the benchmarking pipeline.
