# Bengali (`bn`)

## Datasets
Wikiann

## Recommended models
xlm-roberta-large-finetuned-conll03-english; Suchandra/bengali_language_NER (fine-tuned on Wikiann)

## Annotation scheme
IOB2 (Wikiann); custom labelling (Bengali model)

## Evaluation

Models are evaluated using:
- **seqeval** — entity-level precision, recall, F1
- **sklearn** — token-level metrics

See `evaluation.py` for the benchmarking pipeline.
