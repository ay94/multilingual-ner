# Benchmarks

Model evaluation notebooks for NER across 15 languages. Each notebook evaluates one or more models against a benchmark dataset using seqeval (entity-level) and sklearn (token-level) metrics.

## Structure

```
benchmarks/
  ar/   Arabic
  cs/   Czech
  de/   German
  el/   Greek
  ha/   Hausa
  hi/   Hindi
  ja/   Japanese
  ro/   Romanian
  sk/   Slovak
  sr-hr/ Serbo-Croatian
  th/   Thai
  tr/   Turkish
  xh/   Xhosa
  zh/   Mandarin Chinese
  zu/   Zulu
```

## Running a benchmark

Each notebook follows the same pattern:

1. Install dependencies (`transformers`, `seqeval`, `datasets`)
2. Load the benchmark dataset for the language
3. Align labels to PER / LOC / ORG / MISC schema
4. Load and run the candidate NER model
5. Compute seqeval and sklearn metrics
6. Compare against baseline (xlm-roberta-large-finetuned-conll03-english)

See `notebooks/ner_demo.ipynb` for a minimal worked example.
