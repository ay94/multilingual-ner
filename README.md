# multilingual-ner

Multilingual Named Entity Recognition toolkit — covering model benchmarking, large-scale extraction, and qualitative validation across 26 languages.

## Components

| Module | Description |
|---|---|
| `evaluation.py` | Benchmark NER models against labelled datasets using seqeval (entity-level) and sklearn (token-level) metrics |
| `extraction.py` | Run NER on unlabelled project data at scale using HuggingFace pipelines with batch processing |
| `validation.py` | Dash app for qualitative review of extraction outputs — colour-coded entity display, mistake and missing entity annotation |

## Evaluation workflow

Three-stage process:

1. **Model identification** — find candidate models via HuggingFace, check licensing and annotation scheme (CoNLL BIO format: PER/LOC/ORG/MISC)
2. **Benchmark ranking** — evaluate against standard datasets using seqeval and sklearn; exclude poor performers
3. **Project-specific validation** — run the validation app on a sample of project extractions; analysts review and annotate mistakes and missing entities

## Languages

26 languages covered. Each has a language card in [`languages/`](languages/) with dataset information and model recommendations.

| Code | Language | Dataset |
|---|---|---|
| `ar` | Arabic | AQMAR Arabic NER Corpus |
| `en` | English | WNUT 17, WikiGold |
| `fr` | French | enp_FR |
| `ru` | Russian | Russian NER test set |
| `af` | Afrikaans | — |
| `bg` | Bulgarian | — |
| `bn` | Bengali | — |
| `cs` | Czech | — |
| `el` | Greek | — |
| `es` | Spanish | — |
| `fa` | Farsi | — |
| `ha` | Hausa | — |
| `hi` | Hindi | — |
| `id` | Indonesian | — |
| `ms` | Malay | — |
| `pt` | Portuguese | — |
| `sk` | Slovak | — |
| `sr-hr` | Serbo-Croatian | — |
| `sw` | Swahili | — |
| `tr` | Turkish | — |
| `tw` | Twi | — |
| `ur` | Urdu | — |
| `vi` | Vietnamese | — |
| `xh` | Xhosa | — |
| `zh` | Mandarin Chinese | — |
| `zu` | Zulu | — |

## Installation

```bash
pip install -e .
```

## Quick start

### Benchmark evaluation

```python
from multilingual_ner.evaluation import ReadNERData

reader = ReadNERData()
sentences, labels = reader.read_ner_file("path/to/ner_data.txt")
```

### Extraction

```python
import pandas as pd
from multilingual_ner import NamedEntityExtractions

df = pd.DataFrame({"text": ["Ahmed visited Cairo last week.", "The UN met in Geneva."]})
ner = NamedEntityExtractions(model_name="dslim/bert-base-NER", project_data=df)
outputs = ner.extract_outputs()
```

### Validation app

```bash
python -m multilingual_ner.validation
# Opens at http://localhost:8050
# Upload a CSV with extraction outputs to review
```
