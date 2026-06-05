# multilingual-ner

Multilingual Named Entity Recognition toolkit — covering model benchmarking, large-scale extraction, and qualitative validation across a range of languages.

## Components

| Module | Description |
|---|---|
| `evaluation.py` | Benchmark NER models against labelled datasets using seqeval (entity-level) and sklearn (token-level) metrics |
| `extraction.py` | Run NER on unlabelled project data at scale using HuggingFace pipelines with batch processing |
| `validation.py` | Dash app for qualitative review of extraction outputs — colour-coded entity display, mistake and missing entity annotation |

## Evaluation workflow

Full methodology: [WORKFLOW.md](WORKFLOW.md)

Three-stage process:

1. **Model identification** — find candidate models via HuggingFace, check licensing and annotation scheme (CoNLL BIO format: PER/LOC/ORG/MISC)
2. **Benchmark ranking** — evaluate against standard datasets using seqeval and sklearn; exclude poor performers
3. **Project-specific validation** — run the validation app on a sample of project extractions; analysts review and annotate mistakes and missing entities

## Languages

Languages this workflow has been applied to:

- Afrikaans
- Arabic
- Bengali
- Bulgarian
- Czech
- Farsi
- French
- German
- Greek
- English
- Hausa
- Hindi
- Indonesian
- Japanese
- Malay
- Mandarin Chinese
- Portuguese
- Romanian
- Russian
- Serbo-Croatian
- Slovak
- Spanish
- Swahili
- Thai
- Turkish
- Twi
- Urdu
- Vietnamese
- Xhosa
- Zulu

## Notebooks

| Notebook | Description |
|---|---|
| [`notebooks/template.ipynb`](notebooks/template.ipynb) | Workflow template — all three stages with placeholders, adapt for any language |
| [`benchmarks/ar/arabic_benchmark.ipynb`](benchmarks/ar/arabic_benchmark.ipynb) | Arabic worked example — WikiANN benchmark, hatmimoha model, dummy extraction with JSON output |

## Installation

```bash
# From GitHub
pip install git+https://github.com/ay94/multilingual-ner.git

# Local development
pip install -e .
```

## Quick start

### Benchmark evaluation

```python
from multilingual_ner.evaluation import ReadNERData, ModelEvaluation

reader = ReadNERData()
words, labels = reader.read_dataset("wikiann", {"O": 0, "B-PER": 1, "I-PER": 2, "B-ORG": 3, "I-ORG": 4, "B-LOC": 5, "I-LOC": 6}, lang="ar")

model = ModelEvaluation("hatmimoha/arabic-ner")
results = model.evaluate_model(words, labels)
print(results.get_classification("Seqeval"))
```

### Extraction

```python
import pandas as pd
from multilingual_ner.extraction import NamedEntityExtractions

df = pd.DataFrame({
    "text": ["Ahmed visited Cairo last week.", "The UN met in Geneva."],
    "message_id": ["msg_1", "msg_2"],
    "accountId": ["acc_1", "acc_1"],
})

extractor = NamedEntityExtractions(model_name="dslim/bert-base-NER", project_data=df)
json_schema, output_df, _, _ = extractor.extract_outputs()
print(output_df[["text", "PER", "LOC", "ORG"]])
```

### Validation app

```bash
python -m multilingual_ner.validation
# Opens at http://localhost:8050
# Upload a CSV with extraction outputs to review
```
